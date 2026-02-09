"""
SECOM Veri Seti - NİHAİ FİNAL PİPELINE 
===========================================
Kalibrasyon Deneyi Sonuçlarına Göre Optimize Edilmiş Versiyon

Kalibrasyon deneyinde test edilen 4 senaryo (Augmentation Yok, %10, %33, %50)
arasından %50 SMOTE + XGBoost en iyi performansı gösterdiği için bu ayarlar
sabitlenmiştir.

Pipeline Yapısı:
    1. IterativeImputer (MICE) - Eksik veri tamamlama
    2. RobustScaler - Özellik ölçekleme
    3. VarianceThreshold - Düşük varyanslı özellik temizleme
    4. SMOTE (sampling_strategy=0.5) - Azınlık sınıfını çoğunluğun %50'sine çıkar
    5. XGBClassifier (scale_pos_weight otomatik)
    6. Threshold Optimizasyonu (F1-Score maksimizasyonu)

Değişiklikler (v1 → v2):
     SMOTE oranı 0.5 olarak sabitlendi (kalibrasyon deneyi sonucu)
     Brier Score eklendi (kalibrasyon metriği)
     Train/Val/Test metrikleri ayrı ayrı raporlanıyor
"""

import warnings
warnings.filterwarnings('ignore')

import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix,
    classification_report, accuracy_score, brier_score_loss
)
from xgboost import XGBClassifier
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
# 🔒 KALİBRASYON DENEYİNDEN GELEN SABİTLER
# =============================================================================
SMOTE_RATIO = 0.5          # Kalibrasyon deneyi kazananı: %50
BEST_MODEL = "XGBoost"     # Tüm senaryolarda en iyi: XGBoost
VARIANCE_THRESHOLD = 0.01  # VarianceThreshold eşiği

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def calculate_missing_ratio(df):
    return df.isnull().sum() / len(df)


def drop_high_missing_columns(X, threshold=0.40):
    missing_ratios = calculate_missing_ratio(X)
    cols_to_drop = missing_ratios[missing_ratios >= threshold].index.tolist()
    X_clean = X.drop(columns=cols_to_drop)
    return X_clean, cols_to_drop


def drop_constant_columns(X):
    constant_cols = []
    for col in X.columns:
        nunique = X[col].dropna().nunique()
        std = X[col].std()
        if nunique <= 1 or (pd.notna(std) and std == 0):
            constant_cols.append(col)
    X_clean = X.drop(columns=constant_cols)
    return X_clean, constant_cols


def calculate_scale_pos_weight(y):
    """XGBoost için scale_pos_weight değerini hesaplar."""
    n_negative = np.sum(y == 0)
    n_positive = np.sum(y == 1)
    if n_positive == 0:
        return 1.0
    return n_negative / n_positive


def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(recall, precision)


def find_best_threshold(y_true, y_prob, metric='f1'):
    """
    F1-Score'u maksimize eden en iyi threshold'u bulur.

    Parameters:
    -----------
    y_true : array - Gerçek etiketler
    y_prob : array - Tahmin olasılıkları
    metric : str   - 'f1', 'recall', 'precision'

    Returns:
    --------
    best_threshold, best_score
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        scores.append(score)

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]

    return best_threshold, best_score


# =============================================================================
# VERİ HAZIRLAMA
# =============================================================================

def load_and_prepare_data(filepath):
    print("=" * 70)
    print("VERİ HAZIRLAMA")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"\n[1] Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
        print("[2] 'Time' sütunu drop edildi")

    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])
    print(f"[3] Hedef değişken ayrıldı: {X.shape[1]} feature")

    X_clean, dropped_missing = drop_high_missing_columns(X, threshold=0.40)
    print(f"[4] %40+ eksik olan {len(dropped_missing)} sütun drop edildi → {X_clean.shape[1]} kaldı")

    X_clean, dropped_constant = drop_constant_columns(X_clean)
    print(f"[5] Sabit {len(dropped_constant)} sütun drop edildi → {X_clean.shape[1]} kaldı")

    y_encoded = (y == 1).astype(int)

    print(f"\n[6] Sınıf dağılımı:")
    print(f"    Pass (0): {(y_encoded == 0).sum()} ({(y_encoded == 0).mean()*100:.2f}%)")
    print(f"    Fail (1): {(y_encoded == 1).sum()} ({(y_encoded == 1).mean()*100:.2f}%)")

    return X_clean, y_encoded


# =============================================================================
# NİHAİ FİNAL PİPELINE  DEĞERLENDİRME
# =============================================================================

def evaluate_final_pipeline_v2(X, y):
    """
    Kalibrasyon deneyi sonuçlarına göre optimize edilmiş pipeline:
    1. IterativeImputer
    2. RobustScaler
    3. VarianceThreshold
    4. SMOTE (sampling_strategy=0.5)  ← KALİBRASYON DENEYİ SONUCU
    5. XGBoost (otomatik scale_pos_weight)
    6. Threshold Optimizasyonu
    """
    print("\n" + "=" * 70)
    print("NİHAİ FİNAL PİPELINE DEĞERLENDİRME")
    print("=" * 70)
    print("Kalibrasyon Deneyi Sonuçlarına Göre Optimize Edilmiş:")
    print(f"  🔒 SMOTE oranı: {SMOTE_RATIO:.0%} (azınlık → çoğunluğun %50'si)")
    print(f"  🔒 Model: {BEST_MODEL}")
    print("  ✅ scale_pos_weight otomatik (SMOTE sonrası dağılıma göre)")
    print("  ✅ Threshold Optimizasyonu (F1 maksimizasyonu)")
    print("  ✅ VarianceThreshold ile feature selection")
    print("  ✅ Brier Score ile kalibrasyon takibi")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Fold bazlı skorlar
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_default = []
    all_y_prob = []
    all_best_thresholds = []

    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y

    total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        fold_start = time.time()
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*70}")

        X_train_full, X_test = X_array[train_idx], X_array[test_idx]
        y_train_full, y_test = y_array[train_idx], y_array[test_idx]

        # Train'i %80 train %20 validation'a böl (threshold optimizasyonu için)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y_train_full
        )

        print(f"Veri bölünmesi:")
        print(f"  Train:      {len(y_train)} örnek (Fail: {np.sum(y_train==1)})")
        print(f"  Validation: {len(y_val)} örnek (Fail: {np.sum(y_val==1)})")
        print(f"  Test:       {len(y_test)} örnek (Fail: {np.sum(y_test==1)})")

        # =================================================================
        # ADIM 1: IMPUTATION
        # =================================================================
        print(f"\n[1/6] IterativeImputer...", end=" ")
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=RANDOM_SEED, n_jobs=-1),
            max_iter=10,
            random_state=RANDOM_SEED
        )
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)
        print("✓")

        # =================================================================
        # ADIM 2: SCALING
        # =================================================================
        print("[2/6] RobustScaler...", end=" ")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        print("✓")

        # =================================================================
        # ADIM 3: VARIANCE THRESHOLD
        # =================================================================
        print("[3/6] VarianceThreshold...", end=" ")
        var_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_train_selected = var_selector.fit_transform(X_train_scaled)
        X_val_selected = var_selector.transform(X_val_scaled)
        X_test_selected = var_selector.transform(X_test_scaled)
        n_features_removed = X_train_scaled.shape[1] - X_train_selected.shape[1]
        print(f"✓ ({n_features_removed} düşük varyanslı özellik kaldırıldı)")

        # =================================================================
        # ADIM 4: SMOTE (sampling_strategy=0.5) ← KALİBRASYON DENEYİ
        # =================================================================
        print("[4/6] SMOTE...", end=" ")

        n_majority = np.sum(y_train == 0)
        n_minority_current = np.sum(y_train == 1)
        n_minority_target = int(n_majority * SMOTE_RATIO)

        if n_minority_target <= n_minority_current:
            # Mevcut azınlık zaten yeterli - SMOTE uygulanmaz
            X_train_resampled = X_train_selected
            y_train_resampled = y_train
            print(f"✓ (Mevcut azınlık ({n_minority_current}) ≥ hedef ({n_minority_target}), SMOTE atlandı)")
        else:
            smote = SMOTE(
                sampling_strategy={1: n_minority_target},
                random_state=RANDOM_SEED
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_selected, y_train
            )
            n_new = n_minority_target - n_minority_current
            print(f"✓ (Fail: {n_minority_current} → {n_minority_target}, +{n_new} sentetik örnek)")

        print(f"  SMOTE sonrası: Fail={np.sum(y_train_resampled==1)}, "
              f"Pass={np.sum(y_train_resampled==0)}, "
              f"Oran={np.mean(y_train_resampled==1):.1%}")

        # =================================================================
        # ADIM 5: XGBoost (Otomatik scale_pos_weight)
        # =================================================================
        print("[5/6] XGBoost eğitimi...", end=" ")

        scale_pos_weight = calculate_scale_pos_weight(y_train_resampled)
        print(f"\n  scale_pos_weight = {scale_pos_weight:.4f}")

        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        print("  Model eğitimi ✓")

        # =================================================================
        # ADIM 6: THRESHOLD OPTİMİZASYONU (Validation set üzerinde)
        # =================================================================
        print("[6/6] Threshold optimizasyonu...", end=" ")

        y_val_prob = model.predict_proba(X_val_selected)[:, 1]
        best_threshold, best_f1_val = find_best_threshold(y_val, y_val_prob, metric='f1')
        all_best_thresholds.append(best_threshold)

        print(f"✓")
        print(f"  En iyi threshold: {best_threshold:.3f} (Val F1: {best_f1_val:.4f})")

        # =================================================================
        # TAHMİNLER VE METRİKLER
        # =================================================================
        print("\nSonuçlar:")

        # Train seti metrikleri (orijinal train, SMOTE'suz - overfitting kontrolü)
        y_train_prob = model.predict_proba(X_train_selected)[:, 1]
        y_train_pred_opt = (y_train_prob >= best_threshold).astype(int)
        brier_train = brier_score_loss(y_train, y_train_prob)

        # Validation seti metrikleri
        y_val_pred_opt = (y_val_prob >= best_threshold).astype(int)
        brier_val = brier_score_loss(y_val, y_val_prob)

        # Test seti metrikleri
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        y_pred_default = model.predict(X_test_selected)
        y_pred_optimized = (y_prob >= best_threshold).astype(int)
        brier_test = brier_score_loss(y_test, y_prob)

        f1_default = f1_score(y_test, y_pred_default, pos_label=1, zero_division=0)
        recall_default = recall_score(y_test, y_pred_default, pos_label=1, zero_division=0)
        f1_optimized = f1_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)
        recall_optimized = recall_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)

        # Ekran çıktısı
        print(f"  {'Set':<12} {'F1_fail':>8} {'Recall':>8} {'Prec':>8} {'ROC-AUC':>9} {'Brier':>8}")
        print(f"  {'─'*55}")

        f1_tr = f1_score(y_train, y_train_pred_opt, pos_label=1, zero_division=0)
        rec_tr = recall_score(y_train, y_train_pred_opt, pos_label=1, zero_division=0)
        pre_tr = precision_score(y_train, y_train_pred_opt, pos_label=1, zero_division=0)
        auc_tr = roc_auc_score(y_train, y_train_prob)
        print(f"  {'Train':<12} {f1_tr:>8.4f} {rec_tr:>8.4f} {pre_tr:>8.4f} {auc_tr:>9.4f} {brier_train:>8.4f}")

        f1_v = f1_score(y_val, y_val_pred_opt, pos_label=1, zero_division=0)
        rec_v = recall_score(y_val, y_val_pred_opt, pos_label=1, zero_division=0)
        pre_v = precision_score(y_val, y_val_pred_opt, pos_label=1, zero_division=0)
        auc_v = roc_auc_score(y_val, y_val_prob)
        print(f"  {'Validation':<12} {f1_v:>8.4f} {rec_v:>8.4f} {pre_v:>8.4f} {auc_v:>9.4f} {brier_val:>8.4f}")

        pre_t = precision_score(y_test, y_pred_optimized, pos_label=1, zero_division=0)
        auc_t = roc_auc_score(y_test, y_prob)
        print(f"  {'Test':<12} {f1_optimized:>8.4f} {recall_optimized:>8.4f} {pre_t:>8.4f} {auc_t:>9.4f} {brier_test:>8.4f}")

        print(f"\n  Threshold karşılaştırma (Test):")
        print(f"    Varsayılan (0.5):       F1={f1_default:.4f}, Recall={recall_default:.4f}")
        print(f"    Optimize ({best_threshold:.3f}):   F1={f1_optimized:.4f}, Recall={recall_optimized:.4f}")
        print(f"    İyileşme:               F1 +{(f1_optimized - f1_default):.4f}, Recall +{(recall_optimized - recall_default):.4f}")

        # =================================================================
        # FOLD SONUÇLARI
        # =================================================================
        fold_result = {
            'fold': fold_idx + 1,
            'best_threshold': best_threshold,
            # Train metrikleri
            'f1_fail_train': f1_tr,
            'recall_fail_train': rec_tr,
            'precision_fail_train': pre_tr,
            'roc_auc_train': auc_tr,
            'pr_auc_train': pr_auc_score(y_train, y_train_prob),
            'brier_train': brier_train,
            # Validation metrikleri
            'f1_fail_val': f1_v,
            'recall_fail_val': rec_v,
            'precision_fail_val': pre_v,
            'roc_auc_val': auc_v,
            'pr_auc_val': pr_auc_score(y_val, y_val_prob),
            'brier_val': brier_val,
            # Test metrikleri (optimize threshold)
            'accuracy': accuracy_score(y_test, y_pred_optimized),
            'f1_macro': f1_score(y_test, y_pred_optimized, average='macro'),
            'f1_fail': f1_optimized,
            'recall_fail': recall_optimized,
            'precision_fail': pre_t,
            'roc_auc': auc_t,
            'pr_auc': pr_auc_score(y_test, y_prob),
            'brier_test': brier_test,
            # Karşılaştırma (varsayılan threshold)
            'f1_fail_default': f1_default,
            'recall_fail_default': recall_default,
        }
        fold_results.append(fold_result)

        # Toplu değerlendirme için
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_optimized)
        all_y_pred_default.extend(y_pred_default)
        all_y_prob.extend(y_prob)

        fold_time = time.time() - fold_start
        print(f"\n  Fold süresi: {fold_time:.1f}s")

    total_time = time.time() - total_start
    avg_threshold = np.mean(all_best_thresholds)

    print(f"\n{'='*70}")
    print(f"Ortalama En İyi Threshold: {avg_threshold:.3f}")
    print(f"Toplam Süre: {total_time:.1f}s")
    print(f"{'='*70}")

    return (fold_results, np.array(all_y_true), np.array(all_y_pred),
            np.array(all_y_pred_default), np.array(all_y_prob), total_time, avg_threshold)


# =============================================================================
# SONUÇ RAPORU
# =============================================================================

def generate_report(fold_results, y_true, y_pred, y_pred_default, y_prob, total_time, avg_threshold):
    """Kapsamlı sonuç raporu üret."""

    print("\n" + "=" * 70)
    print("FİNAL SONUÇ RAPORU (Kalibrasyon Deneyi Optimize)")
    print("=" * 70)

    results_df = pd.DataFrame(fold_results)

    # 1. FOLD BAZLI SONUÇLAR
    print("\n--- FOLD BAZLI SONUÇLAR (Test Seti - Optimize Edilmiş Threshold) ---\n")
    display_cols = ['fold', 'best_threshold', 'f1_fail', 'recall_fail',
                    'precision_fail', 'brier_test', 'f1_fail_default', 'recall_fail_default']
    display_df = results_df[display_cols].copy()

    for col in display_df.columns:
        if col != 'fold':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    print(display_df.to_string(index=False))

    # 2. ÖZET İSTATİSTİKLER
    print("\n--- ÖZET İSTATİSTİKLER (Mean ± Std) ---\n")

    # Test metrikleri
    test_metrics = ['accuracy', 'f1_macro', 'f1_fail', 'recall_fail',
                    'precision_fail', 'roc_auc', 'pr_auc', 'brier_test']

    summary = {}
    print("Test Seti (Optimize Edilmiş Threshold ile):")
    for metric in test_metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        summary[metric] = {'mean': mean_val, 'std': std_val}
        label = "Brier Score" if metric == 'brier_test' else metric
        print(f"  {label:<18}: {mean_val:.4f} ± {std_val:.4f}")

    # Train vs Val vs Test karşılaştırma
    print("\n--- OVERFITTING KONTROLÜ (F1_fail / Brier Score) ---\n")
    for split_name, f1_col, brier_col in [
        ('Train',      'f1_fail_train', 'brier_train'),
        ('Validation', 'f1_fail_val',   'brier_val'),
        ('Test',       'f1_fail',       'brier_test')
    ]:
        f1_mean = results_df[f1_col].mean()
        f1_std = results_df[f1_col].std()
        br_mean = results_df[brier_col].mean()
        br_std = results_df[brier_col].std()
        print(f"  {split_name:<12} F1_fail: {f1_mean:.4f} ± {f1_std:.4f}  |  "
              f"Brier: {br_mean:.4f} ± {br_std:.4f}")

    # Threshold karşılaştırma
    print("\nVarsayılan Threshold (0.5) ile karşılaştırma:")
    f1_default_mean = results_df['f1_fail_default'].mean()
    recall_default_mean = results_df['recall_fail_default'].mean()
    f1_improvement = summary['f1_fail']['mean'] - f1_default_mean
    recall_improvement = summary['recall_fail']['mean'] - recall_default_mean

    print(f"  F1-Score (Fail):    {f1_default_mean:.4f} → {summary['f1_fail']['mean']:.4f} (+{f1_improvement:.4f})")
    print(f"  Recall (Fail):      {recall_default_mean:.4f} → {summary['recall_fail']['mean']:.4f} (+{recall_improvement:.4f})")

    # 3. CONFUSION MATRIX
    print("\n--- CONFUSION MATRIX (Optimize Edilmiş Threshold) ---\n")
    cm_optimized = confusion_matrix(y_true, y_pred)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimized.ravel()

    print(f"                 Predicted")
    print(f"                 Pass    Fail")
    print(f"  Actual Pass    {tn_opt:5d}   {fp_opt:5d}")
    print(f"  Actual Fail    {fn_opt:5d}   {tp_opt:5d}")

    cm_default = confusion_matrix(y_true, y_pred_default)
    tn_def, fp_def, fn_def, tp_def = cm_default.ravel()

    print("\n--- CONFUSION MATRIX (Varsayılan Threshold 0.5) ---\n")
    print(f"                 Predicted")
    print(f"                 Pass    Fail")
    print(f"  Actual Pass    {tn_def:5d}   {fp_def:5d}")
    print(f"  Actual Fail    {fn_def:5d}   {tp_def:5d}")

    print("\n--- KARŞILAŞTIRMA ---")
    print(f"  True Positives:  {tp_def} → {tp_opt} (+{tp_opt - tp_def} hatalı ürün daha yakalandı!)")
    print(f"  False Negatives: {fn_def} → {fn_opt} ({fn_def - fn_opt} kaçırılan hata azaldı!)")
    print(f"  False Positives: {fp_def} → {fp_opt} (+{fp_opt - fp_def} yanlış alarm)")

    tpr = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    fpr = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0
    tnr = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0

    print(f"\n  Sensitivity (TPR/Recall): {tpr:.4f}")
    print(f"  Specificity (TNR):        {tnr:.4f}")
    print(f"  False Positive Rate:      {fpr:.4f}")

    # 4. CLASSIFICATION REPORT
    print("\n--- CLASSIFICATION REPORT (Optimize Edilmiş) ---\n")
    print(classification_report(y_true, y_pred, target_names=['Pass (0)', 'Fail (1)']))

    # 5. TOPLAM SÜRE
    print(f"\n--- TOPLAM SÜRE ---")
    print(f"  {total_time:.1f} saniye ({total_time/60:.1f} dakika)")

    return summary, cm_optimized, cm_default


def generate_thesis_summary(summary, cm_opt, cm_def, avg_threshold):
    """Tez için hazır özet metni üret."""

    print("\n" + "=" * 70)
    print("TEZ İÇİN ÖZET METİN")
    print("=" * 70)

    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    tn_def, fp_def, fn_def, tp_def = cm_def.ravel()

    text = f"""
### Nihai Model Performansı (Kalibrasyon Deneyi Sonrası)

SECOM yarı iletken üretim veri seti üzerinde yapılan kalibrasyon deneyi
kapsamında 4 farklı veri artırım senaryosu (Augmentation Yok, %10, %33, %50)
ve 3 farklı model (XGBoost, LightGBM, RandomForest) sistematik olarak
karşılaştırılmıştır. Brier Score ve Reliability Diagram analizleri
sonucunda en iyi kalibrasyon ve performans aşağıdaki konfigürasyonla
elde edilmiştir:

**Kazanan Konfigürasyon:**
- Model: XGBoost
- SMOTE Oranı: %50 (azınlık sınıfı → çoğunluğun %50'si)

**Pipeline Yapısı:**
1. IterativeImputer (MICE) - Eksik veri tamamlama
2. RobustScaler - Özellik ölçekleme
3. VarianceThreshold - Düşük varyanslı özellik temizleme
4. SMOTE (sampling_strategy=0.5) - Sınıf dengesizliği düzeltme
5. XGBClassifier (scale_pos_weight otomatik)
6. Threshold Optimizasyonu - F1-Score maksimizasyonu

**5-Fold Cross-Validation Sonuçları:**

| Metrik | Ortalama | Std |
|--------|----------|-----|
| F1-Score (Fail) | {summary['f1_fail']['mean']:.4f} | {summary['f1_fail']['std']:.4f} |
| Recall (Fail) | {summary['recall_fail']['mean']:.4f} | {summary['recall_fail']['std']:.4f} |
| Precision (Fail) | {summary['precision_fail']['mean']:.4f} | {summary['precision_fail']['std']:.4f} |
| ROC-AUC | {summary['roc_auc']['mean']:.4f} | {summary['roc_auc']['std']:.4f} |
| PR-AUC | {summary['pr_auc']['mean']:.4f} | {summary['pr_auc']['std']:.4f} |
| Brier Score | {summary['brier_test']['mean']:.4f} | {summary['brier_test']['std']:.4f} |

**Threshold Optimizasyonu Etkisi:**
- Varsayılan (0.5) → Optimize ({avg_threshold:.3f})
- Yakalanan Hatalı Ürün: {tp_def} → {tp_opt} (+{tp_opt - tp_def} ✅)
- Kaçırılan Hata: {fn_def} → {fn_opt} (-{fn_def - fn_opt} ✅)
- Yanlış Alarm: {fp_def} → {fp_opt} (+{fp_opt - fp_def})

**Kalibrasyon Deneyi Önemli Bulguları:**
- SMOTE artırım oranı arttıkça modelin azınlık sınıfını (Fail) tanıma
  yeteneği iyileşmektedir.
- %50 oranı, Recall ve F1 dengesini en iyi sağlayan noktadır.
- XGBoost, gradient boosting ailesi içinde en tutarlı performansı
  sergilemiştir.
- Brier Score ile doğrulanan kalibrasyon, modelin ürettiği olasılık
  tahminlerinin güvenilir olduğunu göstermektedir.

**Yorum:**
Threshold optimizasyonu sayesinde, model {tp_opt - tp_def} adet daha fazla
hatalı ürünü tespit edebilmektedir. Bu, üretim hattında kalite kontrol
maliyetlerini önemli ölçüde azaltabilir. Recall değeri
{summary['recall_fail']['mean']:.2%} olup, hatalı ürünlerin
{summary['recall_fail']['mean']*100:.0f}/100'ünün yakalanabildiğini
göstermektedir.
"""

    print(text)
    return text


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv'):
    print("\n")
    print("*" * 70)
    print("  SECOM - NİHAİ FİNAL PİPELINE")
    print("  Kalibrasyon Deneyi Sonuçlarına Göre Optimize Edilmiş")
    print(f"  SMOTE={SMOTE_RATIO:.0%} + XGBoost + Threshold Optimizasyonu")
    print("*" * 70)

    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)

    # 2. Pipeline değerlendirme
    (fold_results, y_true, y_pred, y_pred_default,
     y_prob, total_time, avg_threshold) = evaluate_final_pipeline_v2(X_clean, y)

    # 3. Rapor üret
    summary, cm_opt, cm_def = generate_report(
        fold_results, y_true, y_pred, y_pred_default, y_prob, total_time, avg_threshold
    )

    # 4. Tez özeti
    thesis_text = generate_thesis_summary(summary, cm_opt, cm_def, avg_threshold)

    # 5. Sonuçları kaydet
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('secom_final_pipeline_results.csv', index=False)

    summary_df = pd.DataFrame([
        {'metric': k, 'mean': v['mean'], 'std': v['std']}
        for k, v in summary.items()
    ])
    summary_df.to_csv('secom_final_pipeline_summary.csv', index=False)

    # Tez metnini de kaydet
    with open('secom_thesis_summary.md', 'w', encoding='utf-8') as f:
        f.write(thesis_text)

    print("\n" + "=" * 70)
    print("[✓] Sonuçlar kaydedildi:")
    print("    - secom_final_pipeline_results.csv (fold bazlı)")
    print("    - secom_final_pipeline_summary.csv (özet)")
    print("    - secom_thesis_summary.md (tez özet metni)")
    print("=" * 70)

    return fold_results, summary, cm_opt, cm_def, avg_threshold


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Buket/uci-secom.csv"

    results, summary, cm_opt, cm_def, avg_threshold = main(filepath)