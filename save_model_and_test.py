"""
SECOM - Model ve Test Verisi Kaydetme
==========================================
Final Pipeline yapısını birebir uygulayarak modeli ve test verilerini
XAI (Açıklanabilir Yapay Zeka) analizi için kaydeder.

Pipeline Yapısı (Kalibrasyon Deneyi Sonucu):
    1. IterativeImputer (MICE)
    2. RobustScaler
    3. VarianceThreshold (threshold=0.01)
    4. SMOTE (sampling_strategy=0.5) → azınlık = çoğunluğun %50'si
    5. XGBClassifier (scale_pos_weight otomatik)
    6. Threshold Optimizasyonu (F1-Score maksimizasyonu)

Kaydedilen Dosyalar:
    - xai_final_model.pkl          : Eğitilmiş XGBoost modeli
    - xai_pipeline_components.pkl  : Imputer, Scaler, VarianceSelector, threshold
    - xai_test_X.csv               : Test feature'ları (processed, top-100)
    - xai_test_y.csv               : Test hedef değişkeni
    - xai_test_X_original.csv      : Test feature'ları (raw, referans)
    - xai_test_predictions.csv     : Tahminler (y_true, y_pred, y_prob)
    - xai_top100_features.txt      : Top-100 feature listesi
    - xai_feature_names.txt        : Kullanılan feature isimleri
    - xai_train_info.txt           : Eğitim bilgileri ve metrikler
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix,
    classification_report, brier_score_loss
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================================================
# GLOBAL SEED & SABİTLER 
# =============================================================================
RANDOM_SEED = 42
SMOTE_RATIO = 0.5           # Kalibrasyon deneyi kazananı
VARIANCE_THRESHOLD = 0.01   # VarianceThreshold eşiği

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
    constant_cols = []
    for col in X.columns:
        nunique = X[col].dropna().nunique()
        std = X[col].std()
        if nunique <= 1 or (pd.notna(std) and std == 0):
            constant_cols.append(col)
    return X.drop(columns=constant_cols), constant_cols

def calculate_scale_pos_weight(y):
    n_negative = np.sum(y == 0)
    n_positive = np.sum(y == 1)
    if n_positive == 0:
        return 1.0
    return n_negative / n_positive

def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(recall, precision)

def find_best_threshold(y_true, y_prob, metric='f1'):
    """F1-Score'u maksimize eden en iyi threshold'u bulur."""
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
    return thresholds[best_idx], scores[best_idx]


# =============================================================================
# ANA FONKSİYON: MODEL VE TEST VERİSİ KAYDETME
# =============================================================================

def save_model_and_test_data(filepath, output_dir='save_model_and_test_outputs/'):
    """
    Final Pipeline yapısını birebir uygulayarak modeli ve test
    verilerini XAI analizi için kaydeder.

    Veri bölümü:
        Train (%64) → Validation (%16) → Test (%20)
        (Önce %80/%20 train-test, sonra train'in %80/%20'si train-val)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 70)
    print("  MODEL VE TEST VERİSİ KAYDETME (Pipeline)")
    print("=" * 70)
    print(f"  SMOTE Oranı: {SMOTE_RATIO:.0%}")
    print(f"  VarianceThreshold: {VARIANCE_THRESHOLD}")
    print(f"  Çıktı dizini: {output_dir}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # ADIM 1: VERİ YÜKLEME VE TEMİZLEME
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/9] Veri yükleniyor ve temizleniyor...")
    df = pd.read_csv(filepath)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])

    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])

    X_clean, dropped_missing = drop_high_missing_columns(X, threshold=0.40)
    X_clean, dropped_constant = drop_constant_columns(X_clean)
    y_encoded = (y == 1).astype(int)

    print(f"       Orijinal: {X.shape[1]} feature")
    print(f"       Eksik veri temizliği: -{len(dropped_missing)} sütun")
    print(f"       Sabit sütun temizliği: -{len(dropped_constant)} sütun")
    print(f"       Temiz veri: {X_clean.shape}")
    print(f"       Sınıf dağılımı: Pass={np.sum(y_encoded==0)}, Fail={np.sum(y_encoded==1)} "
          f"({np.mean(y_encoded):.2%})")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 2: TRAIN / VAL / TEST SPLIT
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/9] Veri bölünüyor (Train %64 / Val %16 / Test %20)...")

    # Önce %80 train+val, %20 test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_clean, y_encoded,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    # Sonra train+val'ı %80 train, %20 val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_trainval
    )

    print(f"       Train:      {X_train.shape[0]:>5} örnek (Fail: {np.sum(y_train==1)})")
    print(f"       Validation: {X_val.shape[0]:>5} örnek (Fail: {np.sum(y_val==1)})")
    print(f"       Test:       {X_test.shape[0]:>5} örnek (Fail: {np.sum(y_test==1)})")

    # Orijinal test verisini kaydet (referans)
    X_test.to_csv(f'{output_dir}xai_test_X_original.csv', index=False)
    print(f"       ✓ Orijinal test verisi kaydedildi")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 3: IMPUTATION
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/9] IterativeImputer uygulanıyor...")
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=RANDOM_SEED, n_jobs=-1),
        max_iter=10,
        random_state=RANDOM_SEED
    )
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    print("       ✓ Imputation tamamlandı")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 4: SCALING
    # ─────────────────────────────────────────────────────────────────
    print("\n[4/9] RobustScaler uygulanıyor...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    print("       ✓ Scaling tamamlandı")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 5: VARIANCE THRESHOLD
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[5/9] VarianceThreshold (threshold={VARIANCE_THRESHOLD})...")
    var_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_selected = var_selector.fit_transform(X_train_scaled)
    X_val_selected = var_selector.transform(X_val_scaled)
    X_test_selected = var_selector.transform(X_test_scaled)

    # Hangi feature'lar kaldı?
    selected_mask = var_selector.get_support()
    selected_features = X_clean.columns[selected_mask].tolist()
    n_removed = X_train_scaled.shape[1] - X_train_selected.shape[1]
    print(f"       {n_removed} düşük varyanslı özellik kaldırıldı → {len(selected_features)} kaldı")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 6: SMOTE (sampling_strategy=0.5)
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[6/9] SMOTE uygulanıyor (oran: {SMOTE_RATIO:.0%})...")
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

    n_majority = np.sum(y_train_arr == 0)
    n_minority_current = np.sum(y_train_arr == 1)
    n_minority_target = int(n_majority * SMOTE_RATIO)

    if n_minority_target <= n_minority_current:
        X_train_resampled = X_train_selected
        y_train_resampled = y_train_arr
        print(f"       Mevcut azınlık ({n_minority_current}) ≥ hedef ({n_minority_target}), SMOTE atlandı")
    else:
        smote = SMOTE(
            sampling_strategy={1: n_minority_target},
            random_state=RANDOM_SEED
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_selected, y_train_arr
        )
        n_new = n_minority_target - n_minority_current
        print(f"       Fail: {n_minority_current} → {n_minority_target} (+{n_new} sentetik örnek)")

    print(f"       SMOTE sonrası: Fail={np.sum(y_train_resampled==1)}, "
          f"Pass={np.sum(y_train_resampled==0)}")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 7: XGBoost EĞİTİMİ
    # ─────────────────────────────────────────────────────────────────
    print("\n[7/9] XGBoost eğitiliyor...")
    scale_pos_weight = calculate_scale_pos_weight(y_train_resampled)
    print(f"       scale_pos_weight = {scale_pos_weight:.4f}")

    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist'
    )
    final_model.fit(X_train_resampled, y_train_resampled)
    print("       ✓ Model eğitildi")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 8: THRESHOLD OPTİMİZASYONU (Validation seti üzerinde)
    # ─────────────────────────────────────────────────────────────────
    print("\n[8/9] Threshold optimizasyonu (validation seti üzerinde)...")
    y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val
    y_val_prob = final_model.predict_proba(X_val_selected)[:, 1]
    best_threshold, best_f1_val = find_best_threshold(y_val_arr, y_val_prob, metric='f1')
    print(f"       En iyi threshold: {best_threshold:.3f} (Val F1_fail: {best_f1_val:.4f})")

    # ─────────────────────────────────────────────────────────────────
    # ADIM 9: TEST PERFORMANSI VE KAYDETME
    # ─────────────────────────────────────────────────────────────────
    print("\n[9/9] Test performansı hesaplanıyor ve dosyalar kaydediliyor...")
    y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

    # Tahminler
    y_prob_test = final_model.predict_proba(X_test_selected)[:, 1]
    y_pred_default = (y_prob_test >= 0.5).astype(int)
    y_pred_optimized = (y_prob_test >= best_threshold).astype(int)

    # Metrikler - Varsayılan threshold
    f1_def = f1_score(y_test_arr, y_pred_default, pos_label=1, zero_division=0)
    rec_def = recall_score(y_test_arr, y_pred_default, pos_label=1, zero_division=0)
    pre_def = precision_score(y_test_arr, y_pred_default, pos_label=1, zero_division=0)

    # Metrikler - Optimize threshold
    f1_opt = f1_score(y_test_arr, y_pred_optimized, pos_label=1, zero_division=0)
    rec_opt = recall_score(y_test_arr, y_pred_optimized, pos_label=1, zero_division=0)
    pre_opt = precision_score(y_test_arr, y_pred_optimized, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test_arr, y_prob_test)
    pr_auc = pr_auc_score(y_test_arr, y_prob_test)
    brier = brier_score_loss(y_test_arr, y_prob_test)

    print(f"\n       {'Metrik':<20} {'Default(0.5)':>12} {'Optimize({:.2f})':>12}".format(best_threshold))
    print(f"       {'─'*46}")
    print(f"       {'F1_fail':<20} {f1_def:>12.4f} {f1_opt:>12.4f}")
    print(f"       {'Recall_fail':<20} {rec_def:>12.4f} {rec_opt:>12.4f}")
    print(f"       {'Precision_fail':<20} {pre_def:>12.4f} {pre_opt:>12.4f}")
    print(f"       {'ROC-AUC':<20} {roc_auc:>12.4f} {'':>12}")
    print(f"       {'PR-AUC':<20} {pr_auc:>12.4f} {'':>12}")
    print(f"       {'Brier Score':<20} {brier:>12.4f} {'':>12}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_arr, y_pred_optimized)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n       Confusion Matrix (threshold={best_threshold:.2f}):")
    print(f"                        Predicted")
    print(f"                        Pass    Fail")
    print(f"       Actual Pass      {tn:5d}   {fp:5d}")
    print(f"       Actual Fail      {fn:5d}   {tp:5d}")

    # ─── DOSYALARI KAYDET ────────────────────────────────────────────

    print(f"\n       Dosyalar kaydediliyor ({output_dir})...")

    # 1. Model
    with open(f'{output_dir}xai_final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print(f"       ✓ xai_final_model.pkl")

    # 2. Pipeline bileşenleri (imputer, scaler, var_selector, threshold)
    pipeline_components = {
        'imputer': imputer,
        'scaler': scaler,
        'var_selector': var_selector,
        'selected_features_after_variance': selected_features,
        'best_threshold': best_threshold,
        'smote_ratio': SMOTE_RATIO,
        'scale_pos_weight': scale_pos_weight,
        'random_seed': RANDOM_SEED,
    }
    with open(f'{output_dir}xai_pipeline_components.pkl', 'wb') as f:
        pickle.dump(pipeline_components, f)
    print(f"       ✓ xai_pipeline_components.pkl")

    # 3. Test X (processed - VarianceThreshold sonrası)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)
    X_test_df.to_csv(f'{output_dir}xai_test_X.csv', index=False)
    print(f"       ✓ xai_test_X.csv ({X_test_df.shape})")

    # 4. Test y
    y_test_df = pd.DataFrame({'Pass_Fail': y_test_arr})
    y_test_df.to_csv(f'{output_dir}xai_test_y.csv', index=False)
    print(f"       ✓ xai_test_y.csv")

    # 5. Tahminler
    predictions_df = pd.DataFrame({
        'y_true': y_test_arr,
        'y_prob': y_prob_test,
        'y_pred_default_05': y_pred_default,
        'y_pred_optimized': y_pred_optimized,
        'threshold_used': best_threshold
    })
    predictions_df.to_csv(f'{output_dir}xai_test_predictions.csv', index=False)
    print(f"       ✓ xai_test_predictions.csv")

    # 6. Feature isimleri
    with open(f'{output_dir}xai_feature_names.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"       ✓ xai_feature_names.txt ({len(selected_features)} feature)")

    # 7. Eğitim bilgileri özet dosyası
    train_info = f"""SECOM - Final Model Eğitim Bilgileri (Pipeline)
{'='*50}
Tarih: Kalibrasyon Deneyi Sonrası Nihai Model

VERİ BÖLÜMÜ:
  Train:      {X_train.shape[0]} örnek (Fail: {np.sum(y_train_arr==1)})
  Validation: {X_val.shape[0]} örnek (Fail: {np.sum(y_val_arr==1)})
  Test:       {X_test.shape[0]} örnek (Fail: {np.sum(y_test_arr==1)})

PİPELINE ADIMLARI:
  1. IterativeImputer (MICE, max_iter=10)
  2. RobustScaler
  3. VarianceThreshold (threshold={VARIANCE_THRESHOLD}) → {n_removed} feature kaldırıldı → {len(selected_features)} kaldı
  4. SMOTE (ratio={SMOTE_RATIO:.0%}) → Fail: {n_minority_current} → {n_minority_target}
  5. XGBClassifier (scale_pos_weight={scale_pos_weight:.4f})
  6. Threshold Optimizasyonu → best_threshold={best_threshold:.3f}

MODEL PARAMETRELERİ:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  scale_pos_weight: {scale_pos_weight:.4f}
  tree_method: hist
  random_state: {RANDOM_SEED}

TEST SONUÇLARI (threshold={best_threshold:.3f}):
  F1_fail:       {f1_opt:.4f}
  Recall_fail:   {rec_opt:.4f}
  Precision_fail:{pre_opt:.4f}
  ROC-AUC:       {roc_auc:.4f}
  PR-AUC:        {pr_auc:.4f}
  Brier Score:   {brier:.4f}

CONFUSION MATRIX:
  TP={tp}, FP={fp}, FN={fn}, TN={tn}

KARŞILAŞTIRMA (Default 0.5 vs Optimize {best_threshold:.2f}):
  F1_fail:   {f1_def:.4f} → {f1_opt:.4f}
  Recall:    {rec_def:.4f} → {rec_opt:.4f}
  Precision: {pre_def:.4f} → {pre_opt:.4f}

NOT:
  5-Fold CV sonuçları: F1_fail={0.1581:.4f} ± {0.0499:.4f}
  Bu model tek bir split üzerinde eğitilmiştir.
  XAI analizi bu model üzerinden yapılacaktır.
"""
    with open(f'{output_dir}xai_train_info.txt', 'w', encoding='utf-8') as f:
        f.write(train_info)
    print(f"       ✓ xai_train_info.txt")

    # ─── ÖZET ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✓ TÜM DOSYALAR BAŞARIYLA KAYDEDİLDİ!")
    print("=" * 70)
    print(f"\n  Dizin: {output_dir}")
    print(f"  {'─'*50}")
    print(f"  1. xai_final_model.pkl          → Eğitilmiş XGBoost modeli")
    print(f"  2. xai_pipeline_components.pkl  → Imputer, Scaler, VarSelector, Threshold")
    print(f"  3. xai_test_X.csv               → Test feature ({X_test_df.shape})")
    print(f"  4. xai_test_y.csv               → Test etiketleri")
    print(f"  5. xai_test_X_original.csv      → Ham test verisi (referans)")
    print(f"  6. xai_test_predictions.csv     → Tahminler + olasılıklar")
    print(f"  7. xai_feature_names.txt        → {len(selected_features)} feature ismi")
    print(f"  8. xai_train_info.txt           → Eğitim detayları")
    print(f"\n  Bu dosyaları 'xai_analysis.py' scripti ile kullanabilirsiniz.")
    print("=" * 70)

    return final_model, X_test_selected, y_test_arr, selected_features, best_threshold


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":

    filepath = "Buket/uci-secom.csv"
    
    output_dir = "Buket/FinalKodlar/save_model_and_test_outputs/"

    model, X_test, y_test, features, threshold = save_model_and_test_data(
        filepath, output_dir=output_dir
    )