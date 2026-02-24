"""
SECOM - ROBUST MULTI-MODEL PIPELINE (20 Seed × 3 Model) 
=========================================================
Akademik Danışman Revizyonları Doğrultusunda Hazırlanmıştır.

Pipeline Sırası (Revizyona Göre):
    Imputer → Scaler → Feature Selection (Top-100) → SMOTE → Model

GPU Desteği:
    ✅ XGBoost   → device='cuda' 
    ✅ LightGBM  → device='gpu'  
    ✅ RandomForest → cuML (GPU) varsa kullanılır, yoksa sklearn (CPU) fallback
    ℹ️  SimpleImputer (median) kullanılır — IterativeImputer SECOM boyutunda çok yavaş

Kritik Akademik Kurallar:
    ✅ SMOTE SADECE ve SADECE train setine uygulanır → data leakage önlemi
    ✅ Feature importance SADECE train verisiyle hesaplanır → data leakage önlemi
    ✅ 3 Model: XGBoost, LightGBM, RandomForest
    ✅ 20 farklı random seed → istatistiksel güvenilirlik (kararlılık kanıtı)
    ✅ Her modelin en iyi versiyonu kaydedilir (XAI analizi için)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import pickle
import random
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve, auc
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================================================
# GPU TESPİTİ
# =============================================================================

def detect_gpu() -> dict:
    """
    Mevcut GPU'yu tespit eder.
    Döndürdüğü dict:
        'available'      : bool  — en az 1 CUDA GPU var mı?
        'device_count'   : int   — GPU sayısı
        'name'           : str   — ilk GPU'nun adı
        'cuda_version'   : str   — CUDA sürümü (varsa)
        'cuml_available' : bool  — cuML kurulu mu? (RF GPU için)
    """
    info = {
        'available': False,
        'device_count': 0,
        'name': 'N/A',
        'cuda_version': 'N/A',
        'cuml_available': False,
    }

    # nvidia-smi ile hızlı kontrol
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            info['available']     = True
            info['device_count']  = len(lines)
            info['name']          = lines[0].split(',')[0].strip()
    except Exception:
        pass

    # CUDA sürümünü nvcc'den al
    if info['available']:
        try:
            r = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
            for line in r.stdout.split('\n'):
                if 'release' in line.lower():
                    info['cuda_version'] = line.strip().split('release')[-1].split(',')[0].strip()
                    break
        except Exception:
            pass

        # cuML kontrolü (RandomForest GPU desteği için)
        try:
            import cuml  # noqa
            info['cuml_available'] = True
        except ImportError:
            pass

    return info


GPU_INFO = detect_gpu()


def print_gpu_info():
    """Başlangıçta GPU durumunu yazdırır."""
    print("\n" + "─" * 70)
    print("  GPU DURUMU")
    print("─" * 70)
    if GPU_INFO['available']:
        print(f"  ✅ GPU BULUNDU: {GPU_INFO['name']}")
        print(f"     Adet       : {GPU_INFO['device_count']}")
        print(f"     CUDA       : {GPU_INFO['cuda_version']}")
        print(f"     cuML (RF)  : {'✅ Kurulu → RF GPU\'da çalışır' if GPU_INFO['cuml_available'] else '❌ Kurulu değil → RF CPU\'da çalışır'}")
        print(f"\n  XGBoost   → device='cuda' ✅")
        print(f"  LightGBM  → device='gpu'  ✅")
        print(f"  RandomForest → {'cuML GPU ✅' if GPU_INFO['cuml_available'] else 'sklearn CPU (cuML yok)'}")
    else:
        print("  ⚠️  GPU BULUNAMADI — Tüm modeller CPU'da çalışacak.")
        print("      CUDA GPU'nuz varsa 'nvidia-smi' ile kontrol edin.")
    print("─" * 70)


# =============================================================================
# LightGBM KONTROLÜ
# =============================================================================

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[UYARI] LightGBM kurulu değil → pip install lightgbm")
    print("        LightGBM atlanacak, XGBoost ve RandomForest çalışacak.\n")

# =============================================================================
# SABİTLER
# =============================================================================

SEEDS           = list(range(1, 21))   # 20 farklı seed: 1'den 20'ye
TOP_K_FEATURES  = 175                  # Seçilecek özellik sayısı (Top-175)
SMOTE_RATIO     = 0.5                  # Azınlık → çoğunluğun %50'sine
TEST_SIZE       = 0.20                 # %20 test
VAL_SIZE        = 0.20                 # Train'in %20'si → validation

# ── Dosya Yolları ──────────────────────────────────────────────────────────
FILEPATH        = "uci-secom.csv"
BASE_OUTPUT_DIR = "robust_pipeline_outputs"

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def drop_high_missing_columns(X: pd.DataFrame, threshold: float = 0.40):
    ratios = X.isnull().sum() / len(X)
    cols   = ratios[ratios >= threshold].index.tolist()
    return X.drop(columns=cols), cols


def drop_constant_columns(X: pd.DataFrame):
    constant = [
        col for col in X.columns
        if X[col].dropna().nunique() <= 1 or
           (pd.notna(X[col].std()) and X[col].std() == 0)
    ]
    return X.drop(columns=constant), constant


def pr_auc_score(y_true, y_prob) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(rec, prec)


def find_best_threshold(y_true, y_prob) -> tuple:
    """Validation seti üzerinde F1-Score'u maksimize eden threshold'u bulur."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def calc_scale_pos_weight(y) -> float:
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    return n_neg / n_pos if n_pos > 0 else 1.0


# =============================================================================
# VERİ HAZIRLAMA
# =============================================================================

def load_and_prepare_data(filepath: str):
    print("=" * 70)
    print("VERİ HAZIRLAMA")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"[1] Yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")

    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])

    X, dropped_missing = drop_high_missing_columns(X, threshold=0.40)
    print(f"[2] %40+ eksik → {len(dropped_missing)} sütun kaldırıldı "
          f"→ {X.shape[1]} feature kaldı")

    X, dropped_const = drop_constant_columns(X)
    print(f"[3] Sabit sütunlar → {len(dropped_const)} kaldırıldı "
          f"→ {X.shape[1]} feature kaldı")

    y_enc = (y == 1).astype(int)
    print(f"[4] Sınıf dağılımı: Pass={int((y_enc==0).sum())}, "
          f"Fail={int((y_enc==1).sum())} "
          f"({float(y_enc.mean()):.2%} fail)")

    return X, y_enc


# =============================================================================
# FEATURE SELECTION — SADECE TRAIN VERİSİ KULLANILIR
# =============================================================================

def compute_top_k_feature_indices(X_train_scaled: np.ndarray,
                                   y_train: np.ndarray,
                                   seed: int,
                                   top_k: int = 100) -> np.ndarray:
    """
    XGBoost feature importance ile Top-K özellik indekslerini döndürür.
    GPU varsa device='cuda' kullanır.

    ÖNEMLİ:
        Bu fonksiyon SADECE X_train_scaled kullanır → data leakage yok.
    """
    spw = calc_scale_pos_weight(y_train)

    xgb_params = dict(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1 if not GPU_INFO['available'] else 1,  # GPU modunda n_jobs=1
        eval_metric='logloss',
        tree_method='hist',
        verbosity=0,
    )
    if GPU_INFO['available']:
        xgb_params['device'] = 'cuda'

    importance_model = XGBClassifier(**xgb_params)
    importance_model.fit(X_train_scaled, y_train)

    importances   = importance_model.feature_importances_
    top_k_indices = np.argsort(importances)[::-1][:top_k]
    return top_k_indices


# =============================================================================
# MODEL TANIMLARI 
# =============================================================================

def get_models(seed: int, y_train_for_spw: np.ndarray = None) -> dict:
    """
    3 modeli döndürür. GPU varsa ilgili parametreler eklenir.

    XGBoost:
        GPU → device='cuda', tree_method='hist'
        CPU → tree_method='hist' (CPU hist)

    LightGBM:
        GPU → device='gpu'
        CPU → device='cpu' (varsayılan)

    RandomForest:
        GPU → cuML RandomForestClassifier (cuml.ensemble)
        CPU → sklearn RandomForestClassifier (fallback)
    """
    use_gpu = GPU_INFO['available']
    spw     = calc_scale_pos_weight(y_train_for_spw) if y_train_for_spw is not None else 1.0
    models  = {}

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_params = dict(
        n_estimators=400,       # 100 → 400 (daha güçlü ensemble)
        max_depth=8,            # 6 → 8 (daha derin ağaçlar)
        learning_rate=0.05,     # n_estimators arttı → lr düşürüldü (daha stabil)
        subsample=0.8,          # overfitting koruması
        colsample_bytree=0.8,   # overfitting koruması
        min_child_weight=3,     # küçük yaprak koruması
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=1 if use_gpu else -1,
        eval_metric='logloss',
        tree_method='hist',
        verbosity=0,
    )
    if use_gpu:
        xgb_params['device'] = 'cuda'

    models['XGBoost'] = XGBClassifier(**xgb_params)

    # ── LightGBM ─────────────────────────────────────────────────────────
    if LGBM_AVAILABLE:
        lgbm_params = dict(
            n_estimators=400,       # 100 → 400
            max_depth=8,            # 6 → 8
            learning_rate=0.05,     # n_estimators arttı → lr düşürüldü
            num_leaves=63,          # 2^(max_depth-1) → daha zengin ağaç
            subsample=0.8,          # overfitting koruması
            colsample_bytree=0.8,   # overfitting koruması
            min_child_samples=10,   # küçük yaprak koruması
            random_state=seed,
            n_jobs=1 if use_gpu else -1,
            verbose=-1,
        )
        if use_gpu:
            lgbm_params['device'] = 'gpu'
            # LightGBM GPU'da n_jobs dikkate alınmaz ama açıkça belirtelim
            lgbm_params['gpu_use_dp'] = False   # single precision → daha hızlı

        models['LightGBM'] = LGBMClassifier(**lgbm_params)

    # ── RandomForest ─────────────────────────────────────────────────────
    if use_gpu and GPU_INFO['cuml_available']:
        # cuML → GPU destekli RandomForest
        from cuml.ensemble import RandomForestClassifier as cuRF
        models['RandomForest'] = cuRF(
            n_estimators=400,       # 100 → 400
            max_depth=24,           # 16 → 24 (cuML sonsuz derinliği desteklemez)
            min_samples_leaf=2,
            random_state=seed,
            n_streams=1,            # kararlı sonuç için
        )
    else:
        # sklearn CPU fallback
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=400,       # 100 → 400
            max_depth=None,         # sınırsız derinlik (sklearn destekler)
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    return models


# =============================================================================
# TEK SEED DENEYİ
# =============================================================================

def run_single_seed(X: pd.DataFrame,
                    y: pd.Series,
                    seed: int,
                    feature_names: list) -> dict:
    """
    Tek bir seed için tam pipeline'ı çalıştırır.

    Pipeline adımları:
        1. Train / Val / Test Split  (stratified)
        2. SimpleImputer (median)    → fit: SADECE train  [CPU, <1sn]
        3. RobustScaler              → fit: SADECE train
        4. Feature Selection Top-100 → importance: SADECE train [GPU/CPU]
        5. SMOTE                     → SADECE train (KRİTİK KURAL!)
        6. Her model için eğitim     [GPU/CPU]
        7. Threshold optimizasyonu   → validation üzerinde
        8. Test değerlendirmesi
    """
    set_seed(seed)
    print(f"\n{'─'*65}")
    print(f"  SEED {seed:>2}/20")
    print(f"{'─'*65}")

    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y

    # ── 1. Split ──────────────────────────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_arr, y_arr, test_size=TEST_SIZE, random_state=seed, stratify=y_arr
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=seed, stratify=y_tv
    )
    print(f"  Split  → Train:{len(y_train)}, Val:{len(y_val)}, Test:{len(y_test)}")
    print(f"  Fail   → Train:{int(np.sum(y_train==1))}, "
          f"Val:{int(np.sum(y_val==1))}, Test:{int(np.sum(y_test==1))}")

    # ── 2. Imputation (fit: SADECE TRAIN) — CPU ───────────────────────────
    # SimpleImputer → median ile eksik değerleri doldurur.
    # IterativeImputer (ExtraTreesRegressor) SECOM'un 444 feature'ı için
    # seed başına 5-10 dakika sürer; SimpleImputer <1 saniyede biter.
    # Akademik literatürde median imputation standart ve savunulabilir bir yaklaşımdır.
    print(f"  [2/5] SimpleImputer/median (fit: sadece train)...", end=" ", flush=True)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)
    print("✓")

    # ── 3. Scaling (fit: SADECE TRAIN) ────────────────────────────────────
    print(f"  [3/5] RobustScaler...", end=" ", flush=True)
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc   = scaler.transform(X_val_imp)
    X_test_sc  = scaler.transform(X_test_imp)
    print("✓")

    # ── 4. Feature Selection (importance: SADECE TRAIN) ───────────────────
    print(f"  [4/5] Feature Selection Top-{TOP_K_FEATURES} "
          f"({'GPU' if GPU_INFO['available'] else 'CPU'})...", end=" ", flush=True)

    top_k_idx = compute_top_k_feature_indices(X_train_sc, y_train, seed, top_k=TOP_K_FEATURES)

    X_train_fs = X_train_sc[:, top_k_idx]
    X_val_fs   = X_val_sc[:, top_k_idx]
    X_test_fs  = X_test_sc[:, top_k_idx]

    selected_names = [feature_names[i] for i in top_k_idx]
    print(f"✓ ({len(top_k_idx)} özellik)")

    # ── 5. SMOTE ── SADECE TRAIN — KRİTİK DATA LEAKAGE KURALI ────────────
    print(f"  [5/5] SMOTE (SADECE train — val/test'e DOKUNULMAZ)...", end=" ", flush=True)

    n_majority = int(np.sum(y_train == 0))
    n_minority = int(np.sum(y_train == 1))
    n_target   = int(n_majority * SMOTE_RATIO)

    if n_target > n_minority:
        smote = SMOTE(sampling_strategy={1: n_target}, random_state=seed)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_fs, y_train)
        print(f"✓ Fail: {n_minority} → {n_target} (+{n_target - n_minority} sentetik)")
    else:
        X_train_sm, y_train_sm = X_train_fs, y_train
        print(f"✓ (atlandı — mevcut azınlık {n_minority} ≥ hedef {n_target})")

    # ── 6. Modelleri Eğit ve Değerlendir ──────────────────────────────────
    models = get_models(seed, y_train_for_spw=y_train_sm)
    seed_results = {}

    for model_name, model in models.items():
        device_tag = _model_device_tag(model_name)
        print(f"\n  [{model_name}] Eğitiliyor ({device_tag})...", end=" ", flush=True)
        model.fit(X_train_sm, y_train_sm)
        print("✓")

        # cuML predict_proba çıktısı numpy array döndürür ama sütun adlandırması farklı olabilir
        y_val_prob  = _safe_predict_proba(model, X_val_fs)
        best_thr, best_val_f1 = find_best_threshold(y_val, y_val_prob)

        y_test_prob = _safe_predict_proba(model, X_test_fs)
        y_test_pred = (y_test_prob >= best_thr).astype(int)

        f1    = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        rec   = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        prec  = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
        roc   = roc_auc_score(y_test, y_test_prob)
        prauc = pr_auc_score(y_test, y_test_prob)

        print(f"  [{model_name}] Thr={best_thr:.2f} | "
              f"F1={f1:.4f} | Rec={rec:.4f} | Prec={prec:.4f} | ROC={roc:.4f}")

        seed_results[model_name] = {
            'seed':            seed,
            'f1_fail':         f1,
            'recall_fail':     rec,
            'precision_fail':  prec,
            'roc_auc':         roc,
            'pr_auc':          prauc,
            'best_threshold':  best_thr,
            'val_f1':          best_val_f1,
            '_model':                  model,
            '_imputer':                imputer,
            '_scaler':                 scaler,
            '_top_k_idx':              top_k_idx,
            '_selected_feature_names': selected_names,
            '_X_train_fs':             X_train_fs,
            '_X_test_fs':              X_test_fs,
            '_y_test':                 y_test,
        }

    return seed_results


def _model_device_tag(model_name: str) -> str:
    """Log için model cihaz etiketi."""
    if not GPU_INFO['available']:
        return "CPU"
    if model_name == 'XGBoost':
        return "GPU/CUDA"
    if model_name == 'LightGBM':
        return "GPU"
    if model_name == 'RandomForest':
        return "GPU/cuML" if GPU_INFO['cuml_available'] else "CPU (cuML yok)"
    return "?"


def _safe_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    predict_proba'nın sonucunu güvenli biçimde numpy float64 dizisine çevirir.
    cuML modelleri bazen cupy array döndürebilir.
    """
    prob = model.predict_proba(X)
    # cupy array → numpy
    try:
        import cupy as cp
        if isinstance(prob, cp.ndarray):
            prob = cp.asnumpy(prob)
    except ImportError:
        pass
    prob = np.asarray(prob, dtype=np.float64)
    return prob[:, 1]


# =============================================================================
# ÖZET RAPOR
# =============================================================================

def print_and_collect_summary(all_seed_results: list, model_names: list) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("20 SEED ÖZET SONUÇLARI (Mean ± Std)")
    print("=" * 70)

    rows = []
    for model_name in model_names:
        records = [r[model_name] for r in all_seed_results if model_name in r]
        if not records:
            continue

        f1s   = [r['f1_fail']        for r in records]
        recs  = [r['recall_fail']    for r in records]
        precs = [r['precision_fail'] for r in records]
        rocs  = [r['roc_auc']        for r in records]

        print(f"\n  {model_name}:")
        print(f"    F1-Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
              f"[min={np.min(f1s):.4f}, max={np.max(f1s):.4f}]")
        print(f"    Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}  "
              f"[min={np.min(recs):.4f}, max={np.max(recs):.4f}]")
        print(f"    Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}  "
              f"[min={np.min(precs):.4f}, max={np.max(precs):.4f}]")
        print(f"    ROC-AUC   : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")

        rows.append({
            'Model':            model_name,
            'F1_mean':          np.mean(f1s),
            'F1_std':           np.std(f1s),
            'Recall_mean':      np.mean(recs),
            'Recall_std':       np.std(recs),
            'Precision_mean':   np.mean(precs),
            'Precision_std':    np.std(precs),
            'ROC_AUC_mean':     np.mean(rocs),
            'ROC_AUC_std':      np.std(rocs),
        })

    return pd.DataFrame(rows)


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    t_start = time.time()

    print_gpu_info()

    print("\n" + "*" * 70)
    print("  SECOM - ROBUST MULTI-MODEL PIPELINE")
    print(f"  {len(SEEDS)} Seed × 3 Model (XGBoost / LightGBM / RandomForest)")
    print(f"  Top-{TOP_K_FEATURES} Feature Selection  |  SMOTE Ratio={SMOTE_RATIO:.0%}")
    print("  SMOTE → SADECE train setine uygulanır (data leakage koruması)")
    print("*" * 70)

    models_dir  = os.path.join(BASE_OUTPUT_DIR, 'models')
    results_dir = os.path.join(BASE_OUTPUT_DIR, 'results')
    for d in [models_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    X, y = load_and_prepare_data(FILEPATH)
    feature_names = X.columns.tolist()

    model_names = ['XGBoost']
    if LGBM_AVAILABLE:
        model_names.append('LightGBM')
    model_names.append('RandomForest')

    best_info = {m: {'f1': -1.0, 'data': None} for m in model_names}
    all_seed_results = []
    per_model_records = {m: [] for m in model_names}

    print(f"\n{'='*70}")
    print(f"  20 SEED DÖNGÜSÜ BAŞLIYOR")
    print(f"{'='*70}")

    for seed in SEEDS:
        seed_results = run_single_seed(X, y, seed, feature_names)
        all_seed_results.append(seed_results)

        for model_name in model_names:
            if model_name not in seed_results:
                continue
            result = seed_results[model_name]
            record = {k: v for k, v in result.items() if not k.startswith('_')}
            per_model_records[model_name].append(record)

            if result['f1_fail'] > best_info[model_name]['f1']:
                best_info[model_name]['f1']   = result['f1_fail']
                best_info[model_name]['data']  = result

    print(f"\n{'='*70}")
    print("  SONUÇLAR KAYDEDİLİYOR...")
    print(f"{'='*70}")

    for model_name in model_names:
        df_model  = pd.DataFrame(per_model_records[model_name])
        csv_path  = os.path.join(results_dir, f'{model_name.lower()}_20seeds.csv')
        df_model.to_csv(csv_path, index=False)
        print(f"  ✓ {csv_path}")

    print(f"\n  En iyi modeller kaydediliyor (XAI için)...")

    for model_name in model_names:
        bd = best_info[model_name]['data']
        if bd is None:
            continue

        # cuML modeli sklearn gibi pickle'lanamayabilir → joblib kullan
        model_path = os.path.join(models_dir, f'{model_name.lower()}_best.pkl')
        _save_model(bd['_model'], model_path)

        components = {
            'imputer':               bd['_imputer'],
            'scaler':                bd['_scaler'],
            'top_k_indices':         bd['_top_k_idx'],
            'selected_feature_names': bd['_selected_feature_names'],
            'best_threshold':        bd['best_threshold'],
            'smote_ratio':           SMOTE_RATIO,
            'seed':                  bd['seed'],
            'model_name':            model_name,
            'gpu_used':              GPU_INFO['available'],
        }
        comp_path = os.path.join(models_dir, f'{model_name.lower()}_best_components.pkl')
        with open(comp_path, 'wb') as f:
            pickle.dump(components, f)

        X_test_df = pd.DataFrame(bd['_X_test_fs'], columns=bd['_selected_feature_names'])
        X_test_df.to_csv(
            os.path.join(models_dir, f'{model_name.lower()}_best_test_X.csv'), index=False)
        pd.DataFrame({'y_true': bd['_y_test']}).to_csv(
            os.path.join(models_dir, f'{model_name.lower()}_best_test_y.csv'), index=False)

        X_train_df = pd.DataFrame(bd['_X_train_fs'], columns=bd['_selected_feature_names'])
        X_train_df.to_csv(
            os.path.join(models_dir, f'{model_name.lower()}_best_train_X.csv'), index=False)

        print(f"  ✓ {model_name}: Seed={bd['seed']}, "
              f"F1={best_info[model_name]['f1']:.4f}")
        print(f"     Model     : {model_path}")
        print(f"     Bileşenler: {comp_path}")

    summary_df = print_and_collect_summary(all_seed_results, model_names)
    summary_path = os.path.join(results_dir, 'all_models_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✓ Genel özet: {summary_path}")

    print("\n" + "=" * 70)
    print("  TEZ İÇİN SONUÇ TABLOSU (Markdown Formatı)")
    print("=" * 70)
    print()
    print("| Model        | F1-Score (mean ± std) | Recall (mean ± std) | Precision (mean ± std) |")
    print("|--------------|----------------------|---------------------|------------------------|")
    for _, row in summary_df.iterrows():
        print(f"| {row['Model']:<12} | {row['F1_mean']:.4f} ± {row['F1_std']:.4f}      | "
              f"{row['Recall_mean']:.4f} ± {row['Recall_std']:.4f}    | "
              f"{row['Precision_mean']:.4f} ± {row['Precision_std']:.4f}       |")

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  TOPLAM SÜRE: {t_total:.1f}s ({t_total/60:.1f} dakika)")
    print(f"  Çıktı klasörü: {BASE_OUTPUT_DIR}")
    print(f"  GPU kullanıldı: {'EVET' if GPU_INFO['available'] else 'HAYIR (CPU)'}")
    print(f"{'='*70}")

    return best_info, summary_df


def _save_model(model, path: str):
    """
    cuML modeli için joblib, diğerleri için pickle kullanır.
    İkisi de aynı arayüzle yüklenebilir.
    """
    try:
        import joblib
        joblib.dump(model, path)
    except Exception:
        with open(path, 'wb') as f:
            pickle.dump(model, f)


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    best_info, summary_df = main()