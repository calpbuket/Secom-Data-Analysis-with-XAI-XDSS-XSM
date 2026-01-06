"""
SECOM Veri Seti - Sınıf Dengesizliği Stratejileri Karşılaştırması (AŞAMA 1)
===========================================================================
IterativeImputer sabitlenmiş, farklı resampling yöntemleri karşılaştırılır.

Gerekli paketler:
    pip install imbalanced-learn

Senaryolar:
    1. Baseline: scale_pos_weight=14.07 (mevcut)
    2. SMOTE + scale_pos_weight=1
    3. ADASYN + scale_pos_weight=1
    4. RandomUnderSampler + scale_pos_weight=1
    5. SMOTE + Tomek Links (combined) + scale_pos_weight=1
"""

import warnings
warnings.filterwarnings('ignore')

import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier

# imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# =============================================================================
# GLOBAL SEED AYARI - REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

def set_all_seeds(seed=42):
    """Tüm random seed'leri ayarlar - tam reproducibility için."""
    random.seed(seed)
    np.random.seed(seed)
    # XGBoost kendi içinde random_state parametresi kullanıyor
    print(f"[SEED] Tüm random seed'ler {seed} olarak ayarlandı")

# Seed'leri hemen set et
set_all_seeds(RANDOM_SEED)

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def calculate_missing_ratio(df):
    """Her sütun için eksik değer oranını hesaplar."""
    return df.isnull().sum() / len(df)


def drop_high_missing_columns(X, threshold=0.40):
    """Eksik oranı threshold'dan yüksek olan sütunları drop eder."""
    missing_ratios = calculate_missing_ratio(X)
    cols_to_drop = missing_ratios[missing_ratios >= threshold].index.tolist()
    X_clean = X.drop(columns=cols_to_drop)
    return X_clean, cols_to_drop


def drop_constant_columns(X):
    """Standart sapması 0 olan veya tek unique değeri olan sütunları drop eder."""
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
    return n_negative / n_positive


def pr_auc_score(y_true, y_prob):
    """PR-AUC skorunu hesaplar."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return auc(recall, precision)


# =============================================================================
# VERİ HAZIRLAMA
# =============================================================================

def load_and_prepare_data(filepath):
    """Veriyi yükler ve temel hazırlıkları yapar."""
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
    print(f"[4] %40+ eksik olan {len(dropped_missing)} sütun drop edildi")
    print(f"    Kalan sütun sayısı: {X_clean.shape[1]}")
    
    X_clean, dropped_constant = drop_constant_columns(X_clean)
    print(f"[5] Sabit {len(dropped_constant)} sütun drop edildi")
    print(f"    Final sütun sayısı: {X_clean.shape[1]}")
    
    # Encode: -1 (Pass) -> 0, 1 (Fail) -> 1
    y_encoded = (y == 1).astype(int)
    
    print(f"\n[6] Sınıf dağılımı (encoded):")
    for cls in [0, 1]:
        label = "Pass" if cls == 0 else "Fail"
        count = (y_encoded == cls).sum()
        print(f"    {label} ({cls}): {count} ({count/len(y_encoded)*100:.2f}%)")
    
    return X_clean, y_encoded


# =============================================================================
# RESAMPLING STRATEJİLERİ
# =============================================================================

def get_resampling_strategies():
    """
    Farklı resampling stratejilerini döndürür.
    None = resampling yok (baseline)
    """
    return {
        'baseline': None,  # scale_pos_weight kullanılacak
        'smote': SMOTE(random_state=42),
        'adasyn': ADASYN(random_state=42),
        'undersampling': RandomUnderSampler(random_state=42),
        'smote_tomek': SMOTETomek(random_state=42)
    }


# =============================================================================
# SENARYO DEĞERLENDİRME
# =============================================================================

def evaluate_scenario(X, y, resampler, scenario_name, scale_pos_weight_baseline):
    """
    Tek bir senaryo için manuel cross-validation yapar.
    IterativeImputer sabit olarak kullanılır.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline için scale_pos_weight, diğerleri için 1
    if scenario_name == 'baseline':
        spw = scale_pos_weight_baseline
    else:
        spw = 1.0
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': spw,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    # Fold bazlı skorlar
    fold_scores = {
        'f1_macro': [],
        'f1_fail': [],
        'recall_fail': [],
        'precision_fail': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    # Confusion matrix toplamları için
    total_cm = np.zeros((2, 2))
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        fold_start = time.time()
        
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # 1. Imputation (IterativeImputer - sabit)
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
            max_iter=10,
            random_state=42
        )
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # 2. Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # 3. Resampling (sadece train set üzerinde!)
        if resampler is not None:
            try:
                X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)
            except Exception as e:
                print(f"      [!] Resampling hatası fold {fold_idx+1}: {str(e)[:50]}")
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
        # 4. Model eğitimi
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 5. Tahminler
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 6. Metrikler
        fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1))
        fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1))
        fold_scores['precision_fail'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
        fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        total_cm += cm
        
        fold_time = time.time() - fold_start
        print(f"    Fold {fold_idx+1}/5 tamamlandı ({fold_time:.1f}s)")
    
    return fold_scores, total_cm


def evaluate_all_scenarios(X, y, scale_pos_weight_baseline):
    """Tüm sınıf dengesizliği senaryolarını değerlendirir."""
    print("\n" + "=" * 70)
    print("SINIF DENGESİZLİĞİ STRATEJİLERİ DEĞERLENDİRME")
    print("=" * 70)
    
    strategies = get_resampling_strategies()
    
    strategy_display_names = {
        'baseline': 'scale_pos_weight (14.07)',
        'smote': 'SMOTE',
        'adasyn': 'ADASYN',
        'undersampling': 'Random Undersampling',
        'smote_tomek': 'SMOTE + Tomek Links'
    }
    
    results = []
    confusion_matrices = {}
    
    for name, resampler in strategies.items():
        print(f"\n[{name.upper()}] Değerlendiriliyor...")
        scenario_start = time.time()
        
        try:
            fold_scores, total_cm = evaluate_scenario(
                X, y, resampler, name, scale_pos_weight_baseline
            )
            
            scenario_time = time.time() - scenario_start
            
            result = {
                'Senaryo': name,
                'Strateji': strategy_display_names[name],
                'F1_macro_mean': np.mean(fold_scores['f1_macro']),
                'F1_macro_std': np.std(fold_scores['f1_macro']),
                'F1_fail_mean': np.mean(fold_scores['f1_fail']),
                'F1_fail_std': np.std(fold_scores['f1_fail']),
                'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
                'Recall_fail_std': np.std(fold_scores['recall_fail']),
                'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
                'Precision_fail_std': np.std(fold_scores['precision_fail']),
                'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
                'ROC_AUC_std': np.std(fold_scores['roc_auc']),
                'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
                'PR_AUC_std': np.std(fold_scores['pr_auc']),
                'Süre_s': scenario_time
            }
            
            results.append(result)
            confusion_matrices[name] = total_cm
            
            print(f"    ✓ Tamamlandı ({scenario_time:.1f}s)")
            print(f"      F1_macro: {result['F1_macro_mean']:.4f} | F1_fail: {result['F1_fail_mean']:.4f} | Recall_fail: {result['Recall_fail_mean']:.4f}")
            
        except Exception as e:
            print(f"    ✗ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results), confusion_matrices


# =============================================================================
# SONUÇ TABLOSU VE ANALİZ
# =============================================================================

def format_results_table(results_df):
    """Sonuç tablosunu formatla ve yazdır."""
    print("\n" + "=" * 70)
    print("SONUÇ TABLOSU")
    print("=" * 70)
    
    display_cols = [
        'Senaryo', 'Strateji',
        'F1_macro_mean', 'F1_fail_mean', 'Recall_fail_mean', 
        'Precision_fail_mean', 'ROC_AUC_mean', 'PR_AUC_mean', 'Süre_s'
    ]
    
    display_df = results_df[display_cols].copy()
    
    # Sayısal sütunları formatla
    for col in display_df.columns:
        if col not in ['Senaryo', 'Strateji']:
            if col == 'Süre_s':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print("\n")
    print(display_df.to_string(index=False))


def analyze_results(results_df, confusion_matrices):
    """Sonuçları analiz et ve karşılaştırma yap."""
    print("\n" + "=" * 70)
    print("ANALİZ VE KARŞILAŞTIRMA")
    print("=" * 70)
    
    # Baseline değerlerini al
    baseline = results_df[results_df['Senaryo'] == 'baseline'].iloc[0]
    
    print("\n--- Baseline'a Göre Değişimler ---\n")
    print(f"{'Strateji':<30} {'ΔF1_fail':>12} {'ΔRecall_fail':>14} {'ΔPR_AUC':>12} {'ΔROC_AUC':>12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        if row['Senaryo'] == 'baseline':
            continue
        
        delta_f1_fail = row['F1_fail_mean'] - baseline['F1_fail_mean']
        delta_recall = row['Recall_fail_mean'] - baseline['Recall_fail_mean']
        delta_pr_auc = row['PR_AUC_mean'] - baseline['PR_AUC_mean']
        delta_roc_auc = row['ROC_AUC_mean'] - baseline['ROC_AUC_mean']
        
        # İşaret ekle
        sign_f1 = "+" if delta_f1_fail >= 0 else ""
        sign_recall = "+" if delta_recall >= 0 else ""
        sign_pr = "+" if delta_pr_auc >= 0 else ""
        sign_roc = "+" if delta_roc_auc >= 0 else ""
        
        print(f"{row['Strateji']:<30} {sign_f1}{delta_f1_fail:>11.4f} {sign_recall}{delta_recall:>13.4f} {sign_pr}{delta_pr_auc:>11.4f} {sign_roc}{delta_roc_auc:>11.4f}")
    
    # En iyi sonuçlar
    print("\n" + "-" * 70)
    print("EN İYİ SONUÇLAR:")
    print("-" * 70)
    
    metrics = ['F1_macro_mean', 'F1_fail_mean', 'Recall_fail_mean', 'Precision_fail_mean', 'ROC_AUC_mean', 'PR_AUC_mean']
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_scenario = results_df.loc[best_idx, 'Strateji']
        best_value = results_df.loc[best_idx, metric]
        baseline_value = baseline[metric]
        diff = best_value - baseline_value
        sign = "+" if diff >= 0 else ""
        print(f"  {metric}: {best_scenario} ({best_value:.4f}) [baseline'dan {sign}{diff:.4f}]")
    
    # Confusion Matrix özeti
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX ÖZETİ (5 Fold Toplamı):")
    print("-" * 70)
    print(f"{'Strateji':<30} {'TN':>8} {'FP':>8} {'FN':>8} {'TP':>8} {'TPR':>8} {'FPR':>8}")
    print("-" * 70)
    
    for name, cm in confusion_matrices.items():
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        strategy_name = results_df[results_df['Senaryo'] == name]['Strateji'].values[0]
        print(f"{strategy_name:<30} {int(tn):>8} {int(fp):>8} {int(fn):>8} {int(tp):>8} {tpr:>8.3f} {fpr:>8.3f}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv'):
    """Ana fonksiyon."""
    print("\n")
    print("*" * 70)
    print("  SECOM - SINIF DENGESİZLİĞİ STRATEJİLERİ KARŞILAŞTIRMASI (AŞAMA 1)")
    print("  İmputer: IterativeImputer (MICE) - SABİT")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Baseline scale_pos_weight hesapla
    spw = calculate_scale_pos_weight(y)
    print(f"\n[7] Baseline scale_pos_weight: {spw:.2f}")
    
    # 3. Tüm senaryoları değerlendir
    results_df, confusion_matrices = evaluate_all_scenarios(X_clean, y, spw)
    
    # 4. Sonuçları göster
    format_results_table(results_df)
    
    # 5. Analiz
    analyze_results(results_df, confusion_matrices)
    
    # 6. CSV kaydet
    results_df.to_csv('secom_imbalance_results.csv', index=False)
    print("\n" + "=" * 70)
    print("[8] Sonuçlar 'secom_imbalance_results.csv' dosyasına kaydedildi.")
    
    return results_df, confusion_matrices


if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    results, cms = main(filepath)