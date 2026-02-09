"""
SECOM - KALİBRASYON DENEYİ (9.calibration_experiment.py)
=========================================================
4 Farklı Veri Artırım Senaryosu × 3 Model ile Kalibrasyon Analizi

Senaryolar:
    1) Augmentation Yok (Baseline)
    2) SMOTE → azınlık %10 oranına
    3) SMOTE → azınlık %33 oranına
    4) SMOTE → azınlık %50 oranına

Modeller:
    - XGBoost
    - LightGBM
    - Random Forest

Çıktılar:
    - Train / Validation / Test metrikleri (F1_fail, Recall, Precision, ROC-AUC, PR-AUC, Brier Score)
    - Kalibrasyon Eğrileri (Reliability Diagram) → PNG
    - Özet tablo → CSV + LaTeX
"""

import warnings
warnings.filterwarnings('ignore')

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve, auc,
    brier_score_loss
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# =============================================================================
# GLOBAL SEED
# =============================================================================
RANDOM_SEED = 42

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_all_seeds(RANDOM_SEED)

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def calculate_missing_ratio(df):
    return df.isnull().sum() / len(df)

def drop_high_missing_columns(X, threshold=0.40):
    missing_ratios = calculate_missing_ratio(X)
    cols_to_drop = missing_ratios[missing_ratios >= threshold].index.tolist()
    return X.drop(columns=cols_to_drop), cols_to_drop

def drop_constant_columns(X):
    constant_cols = [col for col in X.columns
                     if X[col].dropna().nunique() <= 1 or
                     (pd.notna(X[col].std()) and X[col].std() == 0)]
    return X.drop(columns=constant_cols), constant_cols

def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(recall, precision)

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])

    X_clean, _ = drop_high_missing_columns(X, threshold=0.40)
    X_clean, _ = drop_constant_columns(X_clean)

    y_encoded = (y == 1).astype(int)
    return X_clean, y_encoded

# =============================================================================
# FEATURE IMPORTANCE → TOP-K SEÇİMİ
# =============================================================================

def get_top_features(X, y, top_k=100):
    """ExtraTrees ile feature importance hesapla ve top-k feature seç."""
    from sklearn.ensemble import ExtraTreesClassifier

    print("Feature importance hesaplanıyor...")
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5,
                                      random_state=RANDOM_SEED, n_jobs=1),
        max_iter=5, random_state=RANDOM_SEED
    )
    X_imp = imputer.fit_transform(X)

    # Feature importance için SMOTE kullanılmaz - sadece importance sıralaması
    et = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1)
    # Dengesiz veriyle fit (sadece sıralama için)
    et.fit(X_imp, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': et.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = importance_df['feature'].head(top_k).tolist()
    print(f"  → Top {top_k} feature seçildi (toplam: {X.shape[1]})")
    return top_features, importance_df

# =============================================================================
# MODEL TANIMLARI
# =============================================================================

def get_models():
    """3 model döndürür: XGBoost, LightGBM, RandomForest."""
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1.0,
            random_state=RANDOM_SEED,
            n_jobs=1,
            eval_metric='logloss',
            tree_method='hist'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=1,
            verbose=-1,
            force_col_wise=True
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
    }
    return models

# =============================================================================
# METRİK HESAPLAMA
# =============================================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Verilen y_true ve y_prob için tüm metrikleri hesapla."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'F1_fail': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_fail': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Precision_fail': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'PR_AUC': pr_auc_score(y_true, y_prob),
        'Brier_Score': brier_score_loss(y_true, y_prob),
    }
    return metrics

# =============================================================================
# SMOTE SENARYO UYGULAMA
# =============================================================================

def apply_smote_scenario(X_train, y_train, scenario_name, target_ratio):
    """
    Belirtilen orana göre SMOTE uygular.

    Args:
        target_ratio: Azınlık/Çoğunluk oranı (None = artırım yok)
                      Örn: 0.10 → azınlık, çoğunluğun %10'u kadar olacak
    """
    if target_ratio is None:
        print(f"    [{scenario_name}] Veri artırımı uygulanmadı.")
        return X_train, y_train

    n_majority = np.sum(y_train == 0)
    n_minority_current = np.sum(y_train == 1)
    n_minority_target = int(n_majority * target_ratio)

    if n_minority_target <= n_minority_current:
        print(f"    [{scenario_name}] Mevcut azınlık ({n_minority_current}) zaten hedeften ({n_minority_target}) büyük. Artırım yapılmadı.")
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy={1: n_minority_target},
        random_state=RANDOM_SEED
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"    [{scenario_name}] SMOTE: {n_minority_current} → {n_minority_target} "
          f"(çoğunluk: {n_majority}, oran: {target_ratio:.0%})")
    return X_res, y_res

# =============================================================================
# KALİBRASYON EĞRİSİ ÇİZİMİ
# =============================================================================

def plot_calibration_curves(calibration_data, output_path='calibration_curves.png'):
    """
    Tüm model × senaryo kombinasyonları için kalibrasyon eğrilerini çizer.

    Layout: 3 satır (model) × 4 sütun (senaryo) = 12 subplot
    """
    model_names = list(calibration_data.keys())
    scenario_names = list(calibration_data[model_names[0]].keys())

    n_models = len(model_names)
    n_scenarios = len(scenario_names)

    fig, axes = plt.subplots(n_models, n_scenarios, figsize=(5 * n_scenarios, 5 * n_models))
    fig.suptitle('Kalibrasyon Eğrileri (Reliability Diagram)\nTest Seti Üzerinde',
                 fontsize=16, fontweight='bold', y=1.02)

    for i, model_name in enumerate(model_names):
        for j, scenario_name in enumerate(scenario_names):
            ax = axes[i, j] if n_models > 1 else axes[j]

            data = calibration_data[model_name][scenario_name]
            y_true = data['y_true']
            y_prob = data['y_prob']
            brier = data['brier_score']

            # Kalibrasyon eğrisi
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )

                ax.plot(mean_predicted_value, fraction_of_positives,
                        marker='o', linewidth=2, color='#2196F3', label='Model')
            except Exception:
                ax.text(0.5, 0.5, 'Yetersiz veri', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)

            # Mükemmel kalibrasyon çizgisi
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
                    linewidth=1, label='Mükemmel Kalibrasyon')

            # Histogram (olasılık dağılımı)
            ax2 = ax.twinx()
            ax2.hist(y_prob, bins=20, range=(0, 1), alpha=0.15,
                     color='#FF9800', edgecolor='#FF9800', linewidth=0.5)
            ax2.set_ylabel('Frekans', fontsize=8, color='#FF9800')
            ax2.tick_params(axis='y', labelsize=7, colors='#FF9800')
            ax2.set_ylim(0, max(len(y_prob) * 0.5, 10))

            # Başlıklar ve etiketler
            ax.set_title(f'{model_name}\n{scenario_name}\nBrier={brier:.4f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Tahmin Edilen Olasılık', fontsize=9)
            ax.set_ylabel('Gerçek Pozitif Oranı', fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Kalibrasyon eğrileri kaydedildi: {output_path}")

# =============================================================================
# ÖZET KALİBRASYON KARŞILAŞTIRMA GRAFİĞİ
# =============================================================================

def plot_brier_score_comparison(results_df, output_path='brier_score_comparison.png'):
    """Brier Score karşılaştırma bar grafiği."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Brier Score Karşılaştırması (Düşük = Daha İyi Kalibrasyon)',
                 fontsize=14, fontweight='bold')

    colors = ['#E8E8E8', '#AED581', '#64B5F6', '#FF8A65']
    sets_to_plot = ['Validation', 'Test']

    model_names = results_df['Model'].unique()
    scenario_names = results_df['Scenario'].unique()

    split_map = {'Train': 'train', 'Validation': 'val', 'Test': 'test'}
    for idx, (split_label, split_suffix) in enumerate(split_map.items()):
        ax = axes[idx]
        col = f'Brier_Score_{split_suffix}'

        x = np.arange(len(model_names))
        width = 0.18

        for s_idx, scenario in enumerate(scenario_names):
            subset = results_df[results_df['Scenario'] == scenario]
            values = []
            for model in model_names:
                row = subset[subset['Model'] == model]
                values.append(row[col].values[0] if len(row) > 0 else 0)

            bars = ax.bar(x + s_idx * width, values, width,
                          label=scenario, color=colors[s_idx],
                          edgecolor='gray', linewidth=0.5)

            # Değerleri bar üstüne yaz
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Brier Score', fontsize=11)
        ax.set_title(f'{split_label} Seti', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Brier Score karşılaştırma grafiği kaydedildi: {output_path}")

# =============================================================================
# METRİK KARŞILAŞTIRMA GRAFİĞİ
# =============================================================================

def plot_metric_comparison(results_df, output_path='metric_comparison.png'):
    """F1_fail, Recall, Precision, ROC-AUC, PR-AUC karşılaştırma grafiği (Test seti)."""
    metrics_to_plot = ['F1_fail_test', 'Recall_fail_test', 'Precision_fail_test',
                       'ROC_AUC_test', 'PR_AUC_test']
    metric_labels = ['F1 (Fail)', 'Recall (Fail)', 'Precision (Fail)',
                     'ROC-AUC', 'PR-AUC']

    model_names = results_df['Model'].unique()
    scenario_names = results_df['Scenario'].unique()
    colors = ['#E8E8E8', '#AED581', '#64B5F6', '#FF8A65']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))
    fig.suptitle('Test Seti Metrikleri – Senaryo Karşılaştırması',
                 fontsize=14, fontweight='bold')

    for m_idx, (metric_col, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[m_idx]
        x = np.arange(len(model_names))
        width = 0.18

        for s_idx, scenario in enumerate(scenario_names):
            subset = results_df[results_df['Scenario'] == scenario]
            values = []
            for model in model_names:
                row = subset[subset['Model'] == model]
                values.append(row[metric_col].values[0] if len(row) > 0 else 0)

            ax.bar(x + s_idx * width, values, width,
                   label=scenario, color=colors[s_idx],
                   edgecolor='gray', linewidth=0.5)

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrik karşılaştırma grafiği kaydedildi: {output_path}")

# =============================================================================
# LaTeX TABLO
# =============================================================================

def create_calibration_latex_table(results_df, output_path='calibration_latex.tex'):
    """Kalibrasyon sonuçları için LaTeX tablosu."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Kalibrasyon Deney Sonuçları – Test Seti Metrikleri}
\label{tab:calibration_experiment}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|ccccc|c}
\toprule
\textbf{Senaryo} & \textbf{Model} & \textbf{F1$_{fail}$} & \textbf{Recall} & \textbf{Precision} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{Brier Score} \\
\midrule
"""

    scenarios = results_df['Scenario'].unique()
    for s_idx, scenario in enumerate(scenarios):
        subset = results_df[results_df['Scenario'] == scenario]
        for r_idx, (_, row) in enumerate(subset.iterrows()):
            scenario_label = scenario if r_idx == 0 else ""
            latex += f"{scenario_label} & {row['Model']} & "
            latex += f"{row['F1_fail_test']:.4f} & "
            latex += f"{row['Recall_fail_test']:.4f} & "
            latex += f"{row['Precision_fail_test']:.4f} & "
            latex += f"{row['ROC_AUC_test']:.4f} & "
            latex += f"{row['PR_AUC_test']:.4f} & "
            latex += f"{row['Brier_Score_test']:.4f} \\\\\n"

        if s_idx < len(scenarios) - 1:
            latex += r"\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✓ LaTeX tablosu kaydedildi: {output_path}")
    return latex

# =============================================================================
# ANA DENEY FONKSİYONU
# =============================================================================

def run_calibration_experiment(X, y, top_features, output_dir='.'):
    """
    4 senaryo × 3 model kalibrasyon deneyi çalıştırır.

    Adımlar:
        1) Veriyi Train (%60), Validation (%20), Test (%20) olarak böl
        2) Imputation + Scaling (train üzerinde fit, val/test üzerinde transform)
        3) Her senaryo için SMOTE uygula (sadece train setine)
        4) Model eğit
        5) Train / Validation / Test metrikleri hesapla
        6) Kalibrasyon eğrisi verisi topla
    """

    print("\n" + "=" * 70)
    print("  KALİBRASYON DENEYİ")
    print("  4 Senaryo × 3 Model")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Feature seçimi ve veri bölme
    # ------------------------------------------------------------------
    X_selected = X[top_features]
    X_array = X_selected.values
    y_array = y.values if hasattr(y, 'values') else y

    # İlk bölme: Train+Val (%80) vs Test (%20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_array, y_array, test_size=0.20, stratify=y_array, random_state=RANDOM_SEED
    )
    # İkinci bölme: Train (%60 of total = 75% of trainval) vs Val (%20 of total = 25% of trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_SEED
    )

    print(f"\nVeri Bölümü:")
    print(f"  Train      : {X_train.shape[0]:>5} örnek  (Fail: {np.sum(y_train==1):>4}, "
          f"Pass: {np.sum(y_train==0):>4}, Fail oranı: {np.mean(y_train):.2%})")
    print(f"  Validation : {X_val.shape[0]:>5} örnek  (Fail: {np.sum(y_val==1):>4}, "
          f"Pass: {np.sum(y_val==0):>4}, Fail oranı: {np.mean(y_val):.2%})")
    print(f"  Test       : {X_test.shape[0]:>5} örnek  (Fail: {np.sum(y_test==1):>4}, "
          f"Pass: {np.sum(y_test==0):>4}, Fail oranı: {np.mean(y_test):.2%})")

    # ------------------------------------------------------------------
    # 2) Imputation + Scaling (data leakage yok - sadece train'de fit)
    # ------------------------------------------------------------------
    print("\nImputation & Scaling...")
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5,
                                      random_state=RANDOM_SEED, n_jobs=1),
        max_iter=5, random_state=RANDOM_SEED
    )
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print("  ✓ Imputation ve Scaling tamamlandı (train üzerinde fit)")

    # ------------------------------------------------------------------
    # 3) Senaryolar
    # ------------------------------------------------------------------
    scenarios = {
        '1_NoAugmentation': None,    # SMOTE yok
        '2_SMOTE_10pct':    0.10,    # Azınlık → çoğunluğun %10'u
        '3_SMOTE_33pct':    0.33,    # Azınlık → çoğunluğun %33'ü
        '4_SMOTE_50pct':    0.50,    # Azınlık → çoğunluğun %50'si
    }

    models_dict = get_models()
    all_results = []
    calibration_data = {m: {} for m in models_dict.keys()}

    # ------------------------------------------------------------------
    # 4) Her senaryo × her model
    # ------------------------------------------------------------------
    for scenario_name, smote_ratio in scenarios.items():
        print(f"\n{'─' * 60}")
        print(f"  SENARYO: {scenario_name}")
        print(f"{'─' * 60}")

        # SMOTE uygula (sadece train setine)
        X_train_aug, y_train_aug = apply_smote_scenario(
            X_train_scaled, y_train, scenario_name, smote_ratio
        )

        for model_name, model_template in models_dict.items():
            from sklearn.base import clone
            model = clone(model_template)

            print(f"\n    Model: {model_name}")
            t_start = time.time()

            # Eğit
            model.fit(X_train_aug, y_train_aug)

            # Tahminler (olasılık)
            y_prob_train = model.predict_proba(X_train_scaled)[:, 1]  # Orijinal train
            y_prob_val   = model.predict_proba(X_val_scaled)[:, 1]
            y_prob_test  = model.predict_proba(X_test_scaled)[:, 1]

            # Metrikler
            metrics_train = compute_metrics(y_train, y_prob_train)
            metrics_val   = compute_metrics(y_val, y_prob_val)
            metrics_test  = compute_metrics(y_test, y_prob_test)

            elapsed = time.time() - t_start

            # Sonuçları birleştir
            result = {
                'Scenario': scenario_name,
                'Model': model_name,
                'Time_sec': elapsed,
            }
            for key, val in metrics_train.items():
                result[f'{key}_train'] = val
            for key, val in metrics_val.items():
                result[f'{key}_val'] = val
            for key, val in metrics_test.items():
                result[f'{key}_test'] = val

            all_results.append(result)

            # Kalibrasyon verisi sakla (test seti)
            calibration_data[model_name][scenario_name] = {
                'y_true': y_test,
                'y_prob': y_prob_test,
                'brier_score': metrics_test['Brier_Score']
            }

            print(f"      Train  → F1_fail: {metrics_train['F1_fail']:.4f}  "
                  f"Recall: {metrics_train['Recall_fail']:.4f}  "
                  f"Brier: {metrics_train['Brier_Score']:.4f}")
            print(f"      Val    → F1_fail: {metrics_val['F1_fail']:.4f}  "
                  f"Recall: {metrics_val['Recall_fail']:.4f}  "
                  f"Brier: {metrics_val['Brier_Score']:.4f}")
            print(f"      Test   → F1_fail: {metrics_test['F1_fail']:.4f}  "
                  f"Recall: {metrics_test['Recall_fail']:.4f}  "
                  f"Brier: {metrics_test['Brier_Score']:.4f}")
            print(f"      Süre: {elapsed:.2f}s")

    results_df = pd.DataFrame(all_results)

    # ------------------------------------------------------------------
    # 5) Çıktılar
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SONUÇLARI KAYDETME")
    print("=" * 70)

    # CSV
    csv_path = f'{output_dir}/calibration_experiment_results.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ CSV kaydedildi: {csv_path}")

    # Kalibrasyon eğrileri
    plot_calibration_curves(
        calibration_data,
        output_path=f'{output_dir}/calibration_curves.png'
    )

    # Brier Score karşılaştırma
    plot_brier_score_comparison(
        results_df,
        output_path=f'{output_dir}/brier_score_comparison.png'
    )

    # Metrik karşılaştırma
    plot_metric_comparison(
        results_df,
        output_path=f'{output_dir}/metric_comparison.png'
    )

    # LaTeX
    create_calibration_latex_table(
        results_df,
        output_path=f'{output_dir}/calibration_latex.tex'
    )

    # ------------------------------------------------------------------
    # 6) Özet tablo yazdır
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ÖZET TABLO (Test Seti)")
    print("=" * 70)

    summary_cols = ['Scenario', 'Model', 'F1_fail_test', 'Recall_fail_test',
                    'Precision_fail_test', 'ROC_AUC_test', 'PR_AUC_test',
                    'Brier_Score_test', 'Time_sec']
    print(results_df[summary_cols].to_string(index=False, float_format='%.4f'))

    print("\n" + "=" * 70)
    print("  ÖZET TABLO (Validation Seti)")
    print("=" * 70)

    summary_cols_val = ['Scenario', 'Model', 'F1_fail_val', 'Recall_fail_val',
                        'Precision_fail_val', 'ROC_AUC_val', 'PR_AUC_val',
                        'Brier_Score_val']
    print(results_df[summary_cols_val].to_string(index=False, float_format='%.4f'))

    print("\n" + "=" * 70)
    print("  ÖZET TABLO (Train Seti)")
    print("=" * 70)

    summary_cols_train = ['Scenario', 'Model', 'F1_fail_train', 'Recall_fail_train',
                          'Precision_fail_train', 'ROC_AUC_train', 'PR_AUC_train',
                          'Brier_Score_train']
    print(results_df[summary_cols_train].to_string(index=False, float_format='%.4f'))

    # ------------------------------------------------------------------
    # 7) En iyi model/senaryo analizi
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EN İYİ SONUÇLAR")
    print("=" * 70)

    # En düşük Brier Score (test)
    best_brier = results_df.loc[results_df['Brier_Score_test'].idxmin()]
    print(f"\n  En İyi Kalibrasyon (Düşük Brier Score):")
    print(f"    → {best_brier['Model']} + {best_brier['Scenario']}")
    print(f"      Brier Score: {best_brier['Brier_Score_test']:.4f}")

    # En yüksek F1_fail (test)
    best_f1 = results_df.loc[results_df['F1_fail_test'].idxmax()]
    print(f"\n  En Yüksek F1_fail (Test):")
    print(f"    → {best_f1['Model']} + {best_f1['Scenario']}")
    print(f"      F1_fail: {best_f1['F1_fail_test']:.4f}  "
          f"Recall: {best_f1['Recall_fail_test']:.4f}")

    # En yüksek Recall (test)
    best_recall = results_df.loc[results_df['Recall_fail_test'].idxmax()]
    print(f"\n  En Yüksek Recall (Test):")
    print(f"    → {best_recall['Model']} + {best_recall['Scenario']}")
    print(f"      Recall: {best_recall['Recall_fail_test']:.4f}  "
          f"Precision: {best_recall['Precision_fail_test']:.4f}")

    return results_df, calibration_data


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Buket/uci-secom.csv"

    print("\n")
    print("*" * 70)
    print("  SECOM - KALİBRASYON DENEYİ")
    print("  4 Senaryo × 3 Model (XGBoost, LightGBM, RandomForest)")
    print("*" * 70)

    # 1) Veri hazırla
    X_clean, y = load_and_prepare_data(filepath)
    print(f"\nVeri boyutu: {X_clean.shape}")
    print(f"Sınıf dağılımı: Pass={np.sum(y==0)}, Fail={np.sum(y==1)} "
          f"(Fail oranı: {np.mean(y):.2%})")

    # 2) Top-K feature seç
    top_features, importance_df = get_top_features(X_clean, y, top_k=100)

    # 3) Deneyi çalıştır
    results_df, calibration_data = run_calibration_experiment(
        X_clean, y, top_features, output_dir='.'
    )

    print("\n" + "*" * 70)
    print("  DENEY TAMAMLANDI!")
    print("  Çıktılar:")
    print("    • calibration_experiment_results.csv")
    print("    • calibration_curves.png")
    print("    • brier_score_comparison.png")
    print("    • metric_comparison.png")
    print("    • calibration_latex.tex")
    print("*" * 70)