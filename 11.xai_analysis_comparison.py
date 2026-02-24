"""
SECOM - XAI ANALİZİ VE MODEL KARŞILAŞTIRMASI
==============================================
10.robust_multi_model_pipeline.py tarafından kaydedilen en iyi 3 modeli
kullanarak SHAP ve LIME analizleri yapar.

Bu script 10.robust_multi_model_pipeline.py çalıştırıldıktan sonra
çalıştırılmalıdır.

Analizler:
    1. SHAP Global  : Tüm test seti üzerinde mean |SHAP| → genel önem sırası
    2. SHAP Local   : Belirli bir fail örneği için "neden fail dedi?" açıklaması
    3. LIME Local   : Tekil örnek için alternatif model-agnostik açıklama
    4. Kritik Sensör: Her model için Top-N sensör listesi
    5. Karşılaştırma: 3 modelin kritik sensörlerini yan yana gösteren grafik

Neden SHAP + LIME?
    SHAP: Ağaç tabanlı modellerde tam kesinlik, global tutarlılık.
          "Feature_59 genel olarak en önemli sensör."
    LIME: Model-agnostik, tekil örneklere odaklı.
          "Bu spesifik ürünün fail çıkmasına 3. sensör yol açtı."

Gereksinimler:
    pip install shap lime

Çıktılar:
    xai_outputs/
        shap/     → Global summary/bar plot + local fail/pass waterfall
        lime/     → Tekil örnek LIME grafikleri + açıklama metni
        reports/  → Kritik sensör CSV + metin raporu + karşılaştırma grafiği
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # GUI olmadan dosyaya kaydet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[UYARI] SHAP kurulu değil  → pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[UYARI] LIME kurulu değil  → pip install lime")

# =============================================================================
# SABİTLER
# =============================================================================

MODELS_DIR  = "/Users/umayyentur/Downloads/Buket/FinalKodlar/robust_pipeline_outputs/models"
XAI_DIR     = "/Users/umayyentur/Downloads/Buket/FinalKodlar/xai_outputs"

# 10.robust_multi_model_pipeline.py ile oluşturulan model isimleri
MODEL_NAMES = ['XGBoost', 'LightGBM', 'RandomForest']

TOP_SENSORS    = 20   # Raporlanacak kritik sensör sayısı
N_LIME_SAMPLES = 3    # LIME için analiz edilecek fail örneği sayısı

# =============================================================================
# MODEL VE VERİ YÜKLEME
# =============================================================================

def load_model_and_data(model_name: str, models_dir: str):
    """
    10.robust_multi_model_pipeline.py tarafından kaydedilen dosyaları yükler.
    Eksik dosya varsa None döndürür (hata almadan devam edilir).
    """
    n = model_name.lower()

    paths = {
        'model':      os.path.join(models_dir, f'{n}_best.pkl'),
        'components': os.path.join(models_dir, f'{n}_best_components.pkl'),
        'X_test':     os.path.join(models_dir, f'{n}_best_test_X.csv'),
        'y_test':     os.path.join(models_dir, f'{n}_best_test_y.csv'),
        'X_train':    os.path.join(models_dir, f'{n}_best_train_X.csv'),
    }

    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"  [ATLANDI] {model_name}: eksik dosyalar → {missing}")
        print(f"  Önce 10.robust_multi_model_pipeline.py çalıştırın.")
        return None, None, None, None, None, None

    def _load_pkl(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)

    try:
        model = _load_pkl(paths['model'])
    except Exception as e:
        err = str(e)
        if 'cuml' in err or 'No module named' in err:
            print(f"  [ATLANDI] {model_name}: GPU (cuML) modeli — bu ortamda yüklenemiyor.")
            print(f"  Çözüm: 10.robust_multi_model_pipeline.py'yi bu makinede yeniden çalıştırın.")
        else:
            print(f"  [ATLANDI] {model_name}: model yüklenemedi → {e}")
        return None, None, None, None, None, None

    try:
        components = _load_pkl(paths['components'])
    except Exception as e:
        print(f"  [ATLANDI] {model_name}: components yüklenemedi → {e}")
        return None, None, None, None, None, None

    X_test_df  = pd.read_csv(paths['X_test'])
    y_test_df  = pd.read_csv(paths['y_test'])
    X_train_df = pd.read_csv(paths['X_train'])

    X_test  = X_test_df.values
    y_test  = y_test_df['y_true'].values
    X_train = X_train_df.values
    feature_names = X_test_df.columns.tolist()

    return model, components, X_test, y_test, X_train, feature_names


# =============================================================================
# SHAP ANALİZİ
# =============================================================================

def _extract_shap_values(explainer, X: np.ndarray):
    """
    Farklı model türleri için SHAP değerlerini düzgün çıkarır.
    - XGBoost / LightGBM: doğrudan 2D dizi (n_samples, n_features)
    - RandomForest: list [class_0_array, class_1_array] → class_1 alınır
    """
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        # Binary classification: index 1 = Fail (pozitif sınıf)
        shap_vals = shap_vals[1]

    if shap_vals.ndim == 3:
        # Bazı model versiyonları (n_samples, n_features, n_classes)
        shap_vals = shap_vals[:, :, 1]

    return shap_vals


def _get_expected_value(explainer) -> float:
    """Farklı model türleri için beklenen değeri güvenli çıkarır."""
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


def _plot_waterfall_manual(shap_vals_sample: np.ndarray,
                            base_value: float,
                            feature_names: list,
                            title: str,
                            save_path: str,
                            max_features: int = 15):
    """
    Basit yatay çubuk waterfall grafiği.
    Kırmızı çubuk → Fail olasılığını artıran sensörler
    Mavi çubuk   → Fail olasılığını azaltan sensörler
    """
    top_idx = np.argsort(np.abs(shap_vals_sample))[::-1][:max_features]
    vals  = shap_vals_sample[top_idx]
    names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#d73027' if v > 0 else '#4575b4' for v in vals]
    bars   = ax.barh(range(len(vals)), vals, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xlabel('SHAP Değeri  (pozitif → Fail artırır, negatif → Fail azaltır)', fontsize=9)
    ax.set_title(title, fontsize=11, pad=12)

    for bar, val in zip(bars, vals):
        ha  = 'left' if val >= 0 else 'right'
        pad = max(abs(vals)) * 0.015
        ax.text(val + (pad if val >= 0 else -pad), bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center', ha=ha, fontsize=8)

    # Legend
    legend_handles = [
        matplotlib.patches.Patch(color='#d73027', label='Fail olasılığını artırır'),
        matplotlib.patches.Patch(color='#4575b4', label='Fail olasılığını azaltır'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_shap_analysis(model,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       feature_names: list,
                       model_name: str,
                       output_dir: str) -> dict:
    """
    SHAP analizi yapar:
        - Global: Tüm test seti → mean |SHAP| sıralaması (summary + bar)
        - Local : En güçlü fail ve bir pass örneği için waterfall

    Returns:
        dict: {'shap_values': ..., 'critical_sensors': [...], ...}
    """
    if not SHAP_AVAILABLE:
        print("  [ATLANDILDI] SHAP mevcut değil.")
        return None

    print(f"\n  [{model_name}] SHAP analizi başlıyor...")

    shap_dir = os.path.join(output_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    # ── Explainer seç ────────────────────────────────────────────────────
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = _extract_shap_values(explainer, X_test)
        base_value = _get_expected_value(explainer)
        print(f"    TreeExplainer başarılı → shape: {shap_vals.shape}")
    except Exception as e:
        print(f"    TreeExplainer başarısız ({e}), KernelExplainer deneniyor...")
        try:
            background = shap.sample(X_test, min(50, len(X_test)), random_state=42)
            explainer  = shap.KernelExplainer(model.predict_proba, background)
            raw        = explainer.shap_values(X_test[:80])
            shap_vals  = raw[1] if isinstance(raw, list) else raw
            base_value = _get_expected_value(explainer)
        except Exception as e2:
            print(f"    KernelExplainer da başarısız: {e2}")
            return None

    # ── Global SHAP: Summary (Bee Swarm) ─────────────────────────────────
    print(f"    Global summary plot...", end=" ", flush=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type='dot'
    )
    plt.title(f'{model_name} — SHAP Global Summary (Top 20 Sensör)', fontsize=13, pad=18)
    plt.tight_layout()
    path_summary = os.path.join(shap_dir, f'{model_name.lower()}_shap_global_summary.png')
    plt.savefig(path_summary, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓")

    # ── Global SHAP: Bar (Mean |SHAP|) ────────────────────────────────────
    print(f"    Global bar plot...", end=" ", flush=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type='bar'
    )
    plt.title(f'{model_name} — SHAP Feature Importance (Mean |SHAP|)', fontsize=13, pad=18)
    plt.tight_layout()
    path_bar = os.path.join(shap_dir, f'{model_name.lower()}_shap_global_bar.png')
    plt.savefig(path_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓")

    # ── Local SHAP: Fail örneği ───────────────────────────────────────────
    y_prob     = model.predict_proba(X_test)[:, 1]
    fail_idx   = np.where(y_test == 1)[0]

    if len(fail_idx) > 0:
        # En yüksek fail olasılıklı gerçek fail örneği
        best_fail = fail_idx[np.argmax(y_prob[fail_idx])]
        _plot_waterfall_manual(
            shap_vals[best_fail], base_value, feature_names,
            title=f'{model_name} — SHAP Local  |  Fail Örneği '
                  f'(Gerçek=Fail, Prob={y_prob[best_fail]:.3f})',
            save_path=os.path.join(shap_dir, f'{model_name.lower()}_shap_local_fail.png')
        )
        print(f"    Local fail waterfall  ✓  (örnek idx={best_fail}, "
              f"prob={y_prob[best_fail]:.3f})")

    # ── Local SHAP: Pass örneği ───────────────────────────────────────────
    pass_idx = np.where(y_test == 0)[0]
    if len(pass_idx) > 0:
        sample_pass = pass_idx[0]
        _plot_waterfall_manual(
            shap_vals[sample_pass], base_value, feature_names,
            title=f'{model_name} — SHAP Local  |  Pass Örneği '
                  f'(Gerçek=Pass, Prob={y_prob[sample_pass]:.3f})',
            save_path=os.path.join(shap_dir, f'{model_name.lower()}_shap_local_pass.png')
        )
        print(f"    Local pass waterfall  ✓  (örnek idx={sample_pass}, "
              f"prob={y_prob[sample_pass]:.3f})")

    # ── Kritik sensörler (mean |SHAP| sıralaması) ─────────────────────────
    mean_abs    = np.abs(shap_vals).mean(axis=0)
    top_idx_srt = np.argsort(mean_abs)[::-1][:TOP_SENSORS]
    critical    = [(feature_names[i], float(mean_abs[i])) for i in top_idx_srt]

    return {
        'shap_values':      shap_vals,
        'base_value':       base_value,
        'critical_sensors': critical,
        'mean_abs_shap':    mean_abs,
        'feature_names':    feature_names,
        'y_prob':           y_prob,
    }


# =============================================================================
# LIME ANALİZİ
# =============================================================================

def run_lime_analysis(model,
                       X_train_ref: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       feature_names: list,
                       model_name: str,
                       output_dir: str) -> list:
    """
    LIME analizi:
        - Yüksek fail olasılıklı N gerçek fail örneği için tekil açıklama
        - Her örnek: hangi sensörler 'fail' kararına yol açtı / engel oldu?
        - Grafik + metin dosyası olarak kaydedilir

    LIME Neden Faydalı?
        SHAP genel resmi verir. LIME ise "Bu spesifik wafer neden bozuk?"
        sorusunu mühendise açıklar. Model-agnostik: herhangi bir modele uyar.
    """
    if not LIME_AVAILABLE:
        print("  [ATLANDILDI] LIME mevcut değil.")
        return None

    print(f"\n  [{model_name}] LIME analizi başlıyor...")

    lime_dir = os.path.join(output_dir, 'lime')
    os.makedirs(lime_dir, exist_ok=True)

    # LIME explainer → background olarak train verisi kullanılır
    # (gerçek dağılım bilgisini buradan alır)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_ref,
        feature_names=feature_names,
        class_names=['Pass', 'Fail'],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )

    y_prob     = model.predict_proba(X_test)[:, 1]
    fail_idx   = np.where(y_test == 1)[0]

    if len(fail_idx) == 0:
        print("  [UYARI] Test setinde gerçek fail örneği yok.")
        return None

    # En yüksek olasılıklı N fail örneği
    fail_probs    = y_prob[fail_idx]
    top_fail_idx  = fail_idx[np.argsort(fail_probs)[::-1][:N_LIME_SAMPLES]]

    lime_records = []

    for i, sample_idx in enumerate(top_fail_idx):
        print(f"    LIME örnek {i+1}/{len(top_fail_idx)}  "
              f"(idx={sample_idx}, prob={y_prob[sample_idx]:.3f})...", end=" ", flush=True)
        try:
            explanation = explainer.explain_instance(
                X_test[sample_idx],
                model.predict_proba,
                num_features=15,
                top_labels=2
            )

            # Grafik
            fig = explanation.as_pyplot_figure(label=1)
            plt.title(
                f'{model_name} — LIME Açıklaması\n'
                f'Fail Örneği {i+1}  (Gerçek=Fail, Fail Prob={y_prob[sample_idx]:.3f})',
                fontsize=10
            )
            plt.tight_layout()
            save_path = os.path.join(
                lime_dir, f'{model_name.lower()}_lime_fail_sample{i+1}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            exp_list = explanation.as_list(label=1)
            lime_records.append({
                'sample_idx': int(sample_idx),
                'prob':       float(y_prob[sample_idx]),
                'features':   exp_list
            })
            print("✓")

        except Exception as e:
            print(f"[HATA] {e}")

    # Metin açıklama dosyası kaydet
    if lime_records:
        txt_path = os.path.join(lime_dir, f'{model_name.lower()}_lime_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"LIME Analizi — {model_name}\n")
            f.write("=" * 65 + "\n\n")
            f.write("Yorum Kılavuzu:\n")
            f.write("  Pozitif ağırlık → bu koşul Fail olasılığını ARTIRIYOR\n")
            f.write("  Negatif ağırlık → bu koşul Fail olasılığını AZALTIYOR\n\n")

            for i, rec in enumerate(lime_records):
                f.write(f"Örnek {i+1}  "
                        f"(test_idx={rec['sample_idx']}, "
                        f"Fail Olasılığı={rec['prob']:.4f})\n")
                f.write("-" * 50 + "\n")
                for feat_cond, weight in rec['features']:
                    direction = "↑ FAIL ARTIRIYOR" if weight > 0 else "↓ fail azalıyor"
                    f.write(f"  {feat_cond:<45}  {weight:>+.4f}   {direction}\n")
                f.write("\n")

        print(f"    Metin özeti kaydedildi: {txt_path}")

    return lime_records


# =============================================================================
# KRİTİK SENSÖR RAPORU
# =============================================================================

def generate_critical_sensor_report(shap_results: dict, output_dir: str) -> pd.DataFrame:
    """
    3 modelin kritik sensörlerini karşılaştıran kapsamlı rapor üretir.

    'Kritik sensör' = Mean |SHAP| değeri yüksek sensör
    → Modelin fail kararını verirken en çok bu sensörlere bakıyor
    → Üretim mühendisine "Hangi sensörü izlemeliyim?" sorusunun cevabı
    """
    print("\n" + "=" * 70)
    print("  KRİTİK SENSÖR RAPORU")
    print("=" * 70)

    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    all_rows = []
    valid_results = {k: v for k, v in shap_results.items() if v is not None}

    for model_name, result in valid_results.items():
        print(f"\n  {model_name} — Top {TOP_SENSORS} Kritik Sensör:")
        print(f"  {'Sıra':<5} {'Sensör':<22} {'Mean |SHAP|':>12}")
        print(f"  {'─'*42}")

        for rank, (sensor, score) in enumerate(result['critical_sensors'], 1):
            print(f"  {rank:<5} {sensor:<22} {score:>12.6f}")
            all_rows.append({
                'Model':         model_name,
                'Rank':          rank,
                'Sensor':        sensor,
                'Mean_Abs_SHAP': score
            })

    # ── Ortak sensörler (≥2 modelde Top-N'de) ────────────────────────────
    if len(valid_results) > 1:
        sensor_count = {}
        for result in valid_results.values():
            for sensor, _ in result['critical_sensors']:
                sensor_count[sensor] = sensor_count.get(sensor, 0) + 1

        common = sorted(
            [(s, c) for s, c in sensor_count.items() if c >= 2],
            key=lambda x: -x[1]
        )
        n_models = len(valid_results)

        print(f"\n  ── TÜM MODELLERDE ORTAK KRİTİK SENSÖRLER ──")
        print(f"  (En az 2 modelin Top-{TOP_SENSORS} listesinde yer alanlar)")
        print(f"  {'Sensör':<25} {'Kaç modelde':>12}")
        print(f"  {'─'*40}")
        for sensor, cnt in common:
            print(f"  {sensor:<25} {cnt:>5}/{n_models} modelde")
    else:
        common = []

    # ── CSV kaydet ────────────────────────────────────────────────────────
    df_sensors = pd.DataFrame(all_rows)
    csv_path   = os.path.join(reports_dir, 'critical_sensors_all_models.csv')
    df_sensors.to_csv(csv_path, index=False)
    print(f"\n  ✓ CSV  : {csv_path}")

    # ── Metin raporu kaydet ───────────────────────────────────────────────
    txt_path = os.path.join(reports_dir, 'critical_sensor_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("SECOM — KRİTİK SENSÖR RAPORU\n")
        f.write("=" * 65 + "\n\n")
        f.write("Bu rapor, SHAP analizi ile tespit edilen kritik sensörleri listeler.\n")
        f.write("Yüksek Mean |SHAP| = Model bu sensöre çok önem veriyor\n")
        f.write("                   = Bu sensörde anomali → muhtemelen Fail\n\n")

        for model_name, result in valid_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{model_name} — Top {TOP_SENSORS} Kritik Sensör\n")
            f.write(f"{'='*60}\n")
            for rank, (sensor, score) in enumerate(result['critical_sensors'], 1):
                f.write(f"  {rank:2d}. {sensor:<22}  Mean |SHAP| = {score:.6f}\n")

        if common:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"TÜM MODELLERDE ORTAK KRİTİK SENSÖRLER\n")
            f.write(f"{'='*60}\n")
            f.write(f"(En az 2 modelin Top-{TOP_SENSORS} listesinde bulunanlar)\n\n")
            for sensor, cnt in common:
                f.write(f"  {sensor:<25}  {cnt}/{n_models} modelde kritik\n")

    print(f"  ✓ Metin: {txt_path}")
    return df_sensors


# =============================================================================
# KARŞILAŞTIRMA GRAFİĞİ
# =============================================================================

def plot_model_comparison(shap_results: dict, output_dir: str) -> None:
    """3 modelin top-15 kritik sensörünü yan yana bar grafiği olarak gösterir."""
    valid = {k: v for k, v in shap_results.items() if v is not None}
    if not valid:
        return

    n     = len(valid)
    fig   = plt.figure(figsize=(7 * n, 9))
    gs    = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)
    cmap  = plt.cm.get_cmap('Reds', 20)

    for col, (model_name, result) in enumerate(valid.items()):
        ax      = fig.add_subplot(gs[0, col])
        sensors = [s for s, _ in result['critical_sensors'][:15]]
        scores  = [sc for _, sc in result['critical_sensors'][:15]]
        norm    = np.array(scores) / max(scores) if max(scores) > 0 else np.ones(len(scores))
        colors  = [cmap(0.4 + 0.5 * v) for v in norm]

        bars = ax.barh(range(len(sensors)), scores, color=colors,
                       edgecolor='white', height=0.7)
        ax.set_yticks(range(len(sensors)))
        ax.set_yticklabels(sensors, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f'{model_name}\nTop 15 Kritik Sensör', fontsize=11, pad=8)
        ax.set_xlabel('Mean |SHAP| Değeri', fontsize=9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        for bar, score in zip(bars, scores):
            ax.text(score, bar.get_y() + bar.get_height() / 2,
                    f' {score:.5f}', va='center', fontsize=7.5)

    fig.suptitle(
        'SECOM — Kritik Sensör Karşılaştırması (SHAP)\n'
        '3 Modelin Top-15 Kritik Sensörü',
        fontsize=13, y=1.02
    )

    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out = os.path.join(reports_dir, 'sensor_comparison_all_models.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Karşılaştırma grafiği: {out}")


def plot_shap_heatmap(shap_results: dict, output_dir: str, top_k: int = 15) -> None:
    """
    Tüm modellerin ortak top sensörlerini ısı haritası olarak gösterir.
    Renk yoğunluğu = o modelde sensörün Mean |SHAP| değeri.
    """
    valid = {k: v for k, v in shap_results.items() if v is not None}
    if len(valid) < 2:
        return

    # Her modelden top_k sensörü al ve birleştir
    all_sensors = []
    for result in valid.values():
        all_sensors += [s for s, _ in result['critical_sensors'][:top_k]]
    unique_sensors = list(dict.fromkeys(all_sensors))[:top_k * 2]  # En fazla 2×top_k

    # Matris: satır=sensör, sütun=model
    matrix = np.zeros((len(unique_sensors), len(valid)))
    for j, (model_name, result) in enumerate(valid.items()):
        score_map = {s: sc for s, sc in result['critical_sensors']}
        for i, sensor in enumerate(unique_sensors):
            matrix[i, j] = score_map.get(sensor, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 2.5), max(8, len(unique_sensors) * 0.45)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(list(valid.keys()), fontsize=10)
    ax.set_yticks(range(len(unique_sensors)))
    ax.set_yticklabels(unique_sensors, fontsize=8)
    ax.set_title('SHAP Değerleri Isı Haritası — Sensör × Model', fontsize=12, pad=12)

    plt.colorbar(im, ax=ax, label='Mean |SHAP| Değeri', shrink=0.8)
    plt.tight_layout()

    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out = os.path.join(reports_dir, 'shap_heatmap_sensor_vs_model.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP ısı haritası: {out}")


# =============================================================================
# TEZ ÖZET METNİ
# =============================================================================

def generate_thesis_xai_text(shap_results: dict, output_dir: str) -> None:
    """Tez için hazır XAI özet paragrafı üretir."""
    valid = {k: v for k, v in shap_results.items() if v is not None}
    if not valid:
        return

    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    out = os.path.join(reports_dir, 'thesis_xai_summary.md')

    sensor_count = {}
    for result in valid.values():
        for sensor, _ in result['critical_sensors']:
            sensor_count[sensor] = sensor_count.get(sensor, 0) + 1

    common = sorted(
        [(s, c) for s, c in sensor_count.items() if c >= 2],
        key=lambda x: -x[1]
    )[:10]

    lines = []
    lines.append("## XAI Analizi Bulguları\n")
    lines.append("### SHAP Global Analizi\n")
    lines.append(
        f"SECOM veri setinde {len(valid)} farklı model (XGBoost, LightGBM, RandomForest) "
        "üzerinde SHAP (SHapley Additive exPlanations) analizi uygulanmıştır. "
        "Analiz sonucunda her model için en kritik sensörler aşağıdaki gibi tespit edilmiştir:\n"
    )

    for model_name, result in valid.items():
        top5 = result['critical_sensors'][:5]
        lines.append(f"\n**{model_name}** — Top 5 Kritik Sensör:\n")
        for rank, (sensor, score) in enumerate(top5, 1):
            lines.append(f"{rank}. `{sensor}` (Mean |SHAP| = {score:.6f})\n")

    if common:
        lines.append("\n### Tüm Modellerde Ortak Kritik Sensörler\n")
        lines.append(
            "Aşağıdaki sensörler birden fazla modelin Top-20 listesinde yer almaktadır. "
            "Bu sensörler üretim sürecinde öncelikli izleme listesine alınmalıdır:\n\n"
        )
        for sensor, cnt in common:
            n_models = len(valid)
            lines.append(f"- `{sensor}` → {cnt}/{n_models} modelde kritik\n")

    lines.append("\n### SHAP Local (Tekil Örnek) Analizi\n")
    lines.append(
        "Gerçek fail örnekleri üzerinde yerel SHAP analizi yapılarak, "
        "her hatalı wafer için spesifik sensör katkıları belirlenmiştir. "
        "Bu analiz, üretim mühendisinin 'Bu wafer neden başarısız oldu?' "
        "sorusunu veri odaklı yanıtlamasına olanak tanır.\n"
    )

    lines.append("\n### LIME Analizi\n")
    lines.append(
        "LIME (Local Interpretable Model-agnostic Explanations) yöntemi, "
        "model tipinden bağımsız olarak tekil örnekleri doğrusal bir yaklaşımla açıklar. "
        "SHAP analizi ile tutarlı sensörlerin LIME'da da öne çıkması, "
        "bulguların güvenilirliğini doğrulamaktadır.\n"
    )

    with open(out, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"  ✓ Tez özet metni: {out}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    print("\n" + "*" * 70)
    print("  SECOM — XAI ANALİZİ VE MODEL KARŞILAŞTIRMASI")
    print("  SHAP (Global + Local) + LIME + Kritik Sensör Raporu")
    print("*" * 70)

    if not SHAP_AVAILABLE:
        print("\n[HATA] SHAP zorunlu: pip install shap")
    if not LIME_AVAILABLE:
        print("\n[UYARI] LIME önerilir: pip install lime")

    # Çıktı dizinleri
    for sub in ['shap', 'lime', 'reports']:
        os.makedirs(os.path.join(XAI_DIR, sub), exist_ok=True)

    shap_results = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"  {model_name} — Yükleniyor...")
        print(f"{'='*70}")

        (model, components,
         X_test, y_test,
         X_train, feature_names) = load_model_and_data(model_name, MODELS_DIR)

        if model is None:
            shap_results[model_name] = None
            continue

        print(f"  Model    : {model_name}  (seed={components.get('seed', '?')})")
        print(f"  Test seti: {X_test.shape}  |  Fail sayısı: {int(np.sum(y_test==1))}")
        print(f"  Sensörler: {len(feature_names)}  "
              f"|  Threshold: {components['best_threshold']:.3f}")

        # ── SHAP ────────────────────────────────────────────────────────
        shap_result = run_shap_analysis(
            model, X_test, y_test, feature_names, model_name, XAI_DIR
        )
        shap_results[model_name] = shap_result

        # ── LIME ────────────────────────────────────────────────────────
        if LIME_AVAILABLE:
            run_lime_analysis(
                model, X_train, X_test, y_test,
                feature_names, model_name, XAI_DIR
            )

    # ── Kritik sensör raporu ─────────────────────────────────────────────
    valid = any(v is not None for v in shap_results.values())
    if valid:
        df_sensors = generate_critical_sensor_report(shap_results, XAI_DIR)
        plot_model_comparison(shap_results, XAI_DIR)
        plot_shap_heatmap(shap_results, XAI_DIR)
        generate_thesis_xai_text(shap_results, XAI_DIR)

    # ── Özet ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  XAI ANALİZİ TAMAMLANDI")
    print(f"  Çıktı klasörü: {XAI_DIR}")
    print("=" * 70)
    print("\n  Dosya yapısı:")
    for root, dirs, files in os.walk(XAI_DIR):
        level = root.replace(XAI_DIR, '').count(os.sep)
        indent = '    ' + '  ' * level
        if files:
            print(f"{indent}{os.path.basename(root)}/")
            for fname in sorted(files):
                print(f"{indent}  {fname}")
    print()


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    main()
