"""
SECOM Veri Seti - FİNAL PİPELINE (AŞAMA 2)
============================================
Seçilen yapı:
    1. IterativeImputer (MICE)
    2. RobustScaler
    3. SMOTE
    4. XGBClassifier

Bu kod:
    - 5-Fold CV ile final performansı raporlar
    - Detaylı metrikler ve confusion matrix çıkarır
    - Fold bazlı skorları gösterir
    - Tez için hazır özet üretir
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, 
    classification_report, accuracy_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


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


def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(recall, precision)


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
# FİNAL PİPELINE DEĞERLENDİRME
# =============================================================================

def evaluate_final_pipeline(X, y):
    """
    Final pipeline: IterativeImputer → RobustScaler → SMOTE → XGBoost
    """
    print("\n" + "=" * 70)
    print("FİNAL PİPELINE DEĞERLENDİRME")
    print("Pipeline: IterativeImputer → RobustScaler → SMOTE → XGBoost")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost parametreleri (SMOTE ile scale_pos_weight=1)
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': 1.0,  # SMOTE dengelediği için
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    # Fold bazlı skorlar
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    total_start = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        fold_start = time.time()
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # 1. Imputation
        print("  [1] IterativeImputer...", end=" ")
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
            max_iter=10,
            random_state=42
        )
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        print("✓")
        
        # 2. Scaling
        print("  [2] RobustScaler...", end=" ")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        print("✓")
        
        # 3. SMOTE (sadece train üzerinde)
        print("  [3] SMOTE...", end=" ")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"✓ (Train: {len(y_train)} → {len(y_train_resampled)})")
        
        # 4. Model eğitimi
        print("  [4] XGBoost eğitimi...", end=" ")
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        print("✓")
        
        # 5. Tahminler
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Fold sonuçları
        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_fail': f1_score(y_test, y_pred, pos_label=1),
            'recall_fail': recall_score(y_test, y_pred, pos_label=1),
            'precision_fail': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': pr_auc_score(y_test, y_prob)
        }
        fold_results.append(fold_result)
        
        # Toplu değerlendirme için
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        fold_time = time.time() - fold_start
        print(f"  Süre: {fold_time:.1f}s | F1_fail: {fold_result['f1_fail']:.4f} | Recall: {fold_result['recall_fail']:.4f}")
    
    total_time = time.time() - total_start
    
    return fold_results, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob), total_time


# =============================================================================
# SONUÇ RAPORU
# =============================================================================

def generate_report(fold_results, y_true, y_pred, y_prob, total_time):
    """Kapsamlı sonuç raporu üret."""
    
    print("\n" + "=" * 70)
    print("FİNAL SONUÇ RAPORU")
    print("=" * 70)
    
    # DataFrame oluştur
    results_df = pd.DataFrame(fold_results)
    
    # 1. FOLD BAZLI SONUÇLAR
    print("\n--- FOLD BAZLI SONUÇLAR ---\n")
    display_df = results_df.copy()
    for col in display_df.columns:
        if col != 'fold':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    print(display_df.to_string(index=False))
    
    # 2. ÖZET İSTATİSTİKLER
    print("\n--- ÖZET İSTATİSTİKLER (Mean ± Std) ---\n")
    metrics = ['accuracy', 'f1_macro', 'f1_fail', 'recall_fail', 'precision_fail', 'roc_auc', 'pr_auc']
    
    summary = {}
    for metric in metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        summary[metric] = {'mean': mean_val, 'std': std_val}
        print(f"  {metric:<18}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 3. CONFUSION MATRIX
    print("\n--- CONFUSION MATRIX (Tüm Fold'lar Birleşik) ---\n")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"                 Predicted")
    print(f"                 Pass    Fail")
    print(f"  Actual Pass    {tn:5d}   {fp:5d}")
    print(f"  Actual Fail    {fn:5d}   {tp:5d}")
    
    print(f"\n  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    
    # Oranlar
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n  Sensitivity (TPR/Recall): {tpr:.4f}")
    print(f"  Specificity (TNR):        {tnr:.4f}")
    print(f"  False Positive Rate:      {fpr:.4f}")
    print(f"  False Negative Rate:      {fnr:.4f}")
    
    # 4. CLASSIFICATION REPORT
    print("\n--- CLASSIFICATION REPORT ---\n")
    print(classification_report(y_true, y_pred, target_names=['Pass (0)', 'Fail (1)']))
    
    # 5. TOPLAM SÜRE
    print(f"\n--- TOPLAM SÜRE ---")
    print(f"  {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
    
    return summary, cm


def generate_thesis_summary(summary, cm):
    """Tez için hazır özet metni üret."""
    
    print("\n" + "=" * 70)
    print("TEZ İÇİN ÖZET METİN")
    print("=" * 70)
    
    tn, fp, fn, tp = cm.ravel()
    
    text = f"""
### Final Model Performansı

SECOM yarı iletken üretim veri seti üzerinde yapılan kapsamlı deneyler sonucunda,
en iyi performans gösteren pipeline aşağıdaki bileşenlerden oluşmaktadır:

**Pipeline Yapısı:**
1. IterativeImputer (MICE) - Eksik veri tamamlama
2. RobustScaler - Özellik ölçekleme
3. SMOTE - Sınıf dengesizliği düzeltme
4. XGBClassifier - Sınıflandırma

**5-Fold Cross-Validation Sonuçları:**

| Metrik | Ortalama | Std |
|--------|----------|-----|
| Accuracy | {summary['accuracy']['mean']:.4f} | {summary['accuracy']['std']:.4f} |
| F1-Score (Macro) | {summary['f1_macro']['mean']:.4f} | {summary['f1_macro']['std']:.4f} |
| F1-Score (Fail) | {summary['f1_fail']['mean']:.4f} | {summary['f1_fail']['std']:.4f} |
| Recall (Fail) | {summary['recall_fail']['mean']:.4f} | {summary['recall_fail']['std']:.4f} |
| Precision (Fail) | {summary['precision_fail']['mean']:.4f} | {summary['precision_fail']['std']:.4f} |
| ROC-AUC | {summary['roc_auc']['mean']:.4f} | {summary['roc_auc']['std']:.4f} |
| PR-AUC | {summary['pr_auc']['mean']:.4f} | {summary['pr_auc']['std']:.4f} |

**Confusion Matrix (Toplam):**
- True Negatives: {tn} | False Positives: {fp}
- False Negatives: {fn} | True Positives: {tp}

**Yorum:**
Model, 1567 gözlemden oluşan ve %6.64 oranında hatalı ürün (Fail) içeren 
dengesiz veri setinde, {tp} hatalı ürünü doğru tespit etmiştir. 
Recall değeri {summary['recall_fail']['mean']:.2%} olup, üretim hattındaki 
hatalı ürünlerin yaklaşık {summary['recall_fail']['mean']*100:.0f}/100'ünün 
yakalanabildiğini göstermektedir.
"""
    
    print(text)
    return text


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv'):
    print("\n")
    print("*" * 70)
    print("  SECOM - FİNAL PİPELINE (AŞAMA 2)")
    print("  IterativeImputer → RobustScaler → SMOTE → XGBoost")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Final pipeline değerlendirme
    fold_results, y_true, y_pred, y_prob, total_time = evaluate_final_pipeline(X_clean, y)
    
    # 3. Rapor üret
    summary, cm = generate_report(fold_results, y_true, y_pred, y_prob, total_time)
    
    # 4. Tez özeti
    thesis_text = generate_thesis_summary(summary, cm)
    
    # 5. Sonuçları kaydet
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('secom_final_pipeline_results.csv', index=False)
    
    # Özet metrikleri de kaydet
    summary_df = pd.DataFrame([
        {'metric': k, 'mean': v['mean'], 'std': v['std']} 
        for k, v in summary.items()
    ])
    summary_df.to_csv('secom_final_pipeline_summary.csv', index=False)
    
    print("\n" + "=" * 70)
    print("[✓] Sonuçlar kaydedildi:")
    print("    - secom_final_pipeline_results.csv (fold bazlı)")
    print("    - secom_final_pipeline_summary.csv (özet)")
    print("=" * 70)
    
    return fold_results, summary, cm


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    results, summary, cm = main(filepath)