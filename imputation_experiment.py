"""
SECOM Yarı İletken Üretim Veri Seti - Eksik Veri Stratejileri Karşılaştırması
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score,
    precision_recall_curve, auc
)
from xgboost import XGBClassifier
import time


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
# IMPUTER TANIMLARI (HIZLANDIRILMIŞ)
# =============================================================================

def get_imputers():
    """Üç farklı imputer döndürür - HIZLANDIRILMIŞ PARAMETRELER."""
    return {
        'median': SimpleImputer(strategy='median'),
        
        # KNN: n_neighbors azaltıldı (5->3) - HIZLANDIRILDI
        'knn': KNNImputer(n_neighbors=3),
        
        # Iterative: estimator sayısı ve max_iter azaltıldı - BÜYÜK HIZLANMA
        'iterative': IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=5,      # 10 -> 5 (HIZLANDIRILDI)
                max_depth=5,         # Eklendi - daha hızlı
                random_state=RANDOM_SEED,
                n_jobs=1             # -1 yerine 1 (Mac uyumluluğu)
            ),
            max_iter=5,              # 10 -> 5 (HIZLANDIRILDI)
            random_state=RANDOM_SEED,
            verbose=0
        )
    }


# =============================================================================
# MANUEL CROSS-VALIDATION (HIZLANDIRILMIŞ)
# =============================================================================

def evaluate_scenario(X, y, imputer, scale_pos_weight, scenario_name):
    """
    Tek bir senaryo için manuel cross-validation yapar.
    Pipeline yerine adımları manuel uygular.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scaler = RobustScaler()
    
    # XGBoost parametreleri - HIZLANDIRILDI
    xgb_params = {
        'n_estimators': 50,          # 100 -> 50 (HIZLANDIRILDI)
        'max_depth': 4,              # 6 -> 4 (HIZLANDIRILDI)
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'n_jobs': 1,                 # -1 -> 1 (Mac uyumluluğu)
        'eval_metric': 'logloss',
        'tree_method': 'hist'        # Eklendi - daha hızlı
    }
    
    # Fold bazlı skorlar
    fold_scores = {
        'f1_macro': [],
        'f1_fail': [],
        'recall_fail': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
        fold_start = time.time()
        
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # 1. Imputation (train'de fit, test'e transform)
        imputer_clone = clone(imputer)
        X_train_imp = imputer_clone.fit_transform(X_train)
        X_test_imp  = imputer_clone.transform(X_test)
        
        # 2. Scaling
        scaler_clone = RobustScaler()
        X_train_scaled = scaler_clone.fit_transform(X_train_imp)
        X_test_scaled = scaler_clone.transform(X_test_imp)
        
        # 3. Model eğitimi
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # 4. Tahminler
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 5. Metrikler
        fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1))
        fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1))
        fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
        fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
        
        fold_time = time.time() - fold_start
        print(f"    Fold {fold_idx+1}/5 tamamlandı ({fold_time:.1f}s)")
    
    return fold_scores


def evaluate_all_scenarios(X, y, scale_pos_weight):
    """Tüm senaryoları değerlendirir."""
    print("\n" + "=" * 70)
    print("MODEL DEĞERLENDİRME")
    print("=" * 70)
    
    imputers = get_imputers()
    imputer_display_names = {
        'median': 'SimpleImputer',
        'knn': 'KNNImputer',
        'iterative': 'IterativeImputer'
    }
    
    results = []
    
    for name, imputer in imputers.items():
        print(f"\n[{name.upper()}] Pipeline değerlendiriliyor...")
        scenario_start = time.time()
        
        try:
            fold_scores = evaluate_scenario(X, y, imputer, scale_pos_weight, name)
            
            result = {
                'Senaryo': name,
                'Imputer': imputer_display_names[name],
                'F1_macro_mean': np.mean(fold_scores['f1_macro']),
                'F1_macro_std': np.std(fold_scores['f1_macro']),
                'F1_fail_mean': np.mean(fold_scores['f1_fail']),
                'F1_fail_std': np.std(fold_scores['f1_fail']),
                'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
                'Recall_fail_std': np.std(fold_scores['recall_fail']),
                'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
                'ROC_AUC_std': np.std(fold_scores['roc_auc']),
                'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
                'PR_AUC_std': np.std(fold_scores['pr_auc'])
            }
            
            results.append(result)
            scenario_time = time.time() - scenario_start
            print(f"    ✓ Tamamlandı ({scenario_time:.1f}s) - F1_macro: {result['F1_macro_mean']:.4f} (±{result['F1_macro_std']:.4f})")
            
        except Exception as e:
            print(f"    ✗ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results)


# =============================================================================
# SONUÇ TABLOSU
# =============================================================================

def format_results_table(results_df):
    """Sonuç tablosunu formatla ve yazdır."""
    print("\n" + "=" * 70)
    print("SONUÇ TABLOSU")
    print("=" * 70)
    
    display_cols = [
        'Senaryo', 'Imputer',
        'F1_macro_mean', 'F1_macro_std',
        'F1_fail_mean', 'Recall_fail_mean',
        'ROC_AUC_mean', 'PR_AUC_mean'
    ]
    
    display_df = results_df[display_cols].copy()
    
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print("\n")
    print(display_df.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("EN İYİ SONUÇLAR:")
    print("-" * 70)
    
    metrics = ['F1_macro_mean', 'F1_fail_mean', 'Recall_fail_mean', 'ROC_AUC_mean', 'PR_AUC_mean']
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_scenario = results_df.loc[best_idx, 'Senaryo']
        best_value = results_df.loc[best_idx, metric]
        print(f"  {metric}: {best_scenario} ({best_value:.4f})")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv', seed=None):
    """Ana fonksiyon."""
    # Eğer seed verilmişse global seed'i değiştir
    if seed is not None:
        global RANDOM_SEED
        RANDOM_SEED = seed
        set_all_seeds(seed)
    
    total_start = time.time()
    
    print("\n")
    print("*" * 70)
    print("  SECOM - EKSİK VERİ STRATEJİLERİ KARŞILAŞTIRMASI (HIZLANDIRILMIŞ)")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Scale pos weight hesapla
    spw = calculate_scale_pos_weight(y)
    print(f"\n[7] scale_pos_weight hesaplandı: {spw:.2f}")
    
    # 3. Değerlendirme
    results_df = evaluate_all_scenarios(X_clean, y, spw)
    
    # 4. Sonuçları göster
    format_results_table(results_df)
    
    # 5. CSV kaydet
    results_df.to_csv('secom_imputation_results.csv', index=False)
    print("\n[8] Sonuçlar 'secom_imputation_results.csv' dosyasına kaydedildi.")
    
    total_time = time.time() - total_start
    print(f"\n[9] Toplam süre: {total_time/60:.1f} dakika")
    
    return results_df


if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    
    
    results = main(filepath)