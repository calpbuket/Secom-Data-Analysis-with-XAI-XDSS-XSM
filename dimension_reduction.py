"""
SECOM Veri Seti - BOYUT AZALTMA ANALİZİ (AŞAMA 3)
==================================================
Mevcut pipeline: IterativeImputer → RobustScaler → SMOTE → XGBoost (SABİT)

Deneyler:
    A) PCA: %90, %95, %99 varyans seviyeleri
    B) Feature Selection: Top-50, Top-100, Top-150 features (XGBoost importance)

Gerekli paketler:
    pip install imbalanced-learn xgboost
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
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


def load_and_prepare_data(filepath):
    """Veriyi yükle ve hazırla."""
    print("=" * 70)
    print("VERİ HAZIRLAMA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"\n[1] Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])
    
    X_clean, _ = drop_high_missing_columns(X, threshold=0.40)
    X_clean, _ = drop_constant_columns(X_clean)
    
    y_encoded = (y == 1).astype(int)
    
    print(f"[2] Final boyut: {X_clean.shape[1]} feature")
    print(f"[3] Sınıf dağılımı: Pass={sum(y_encoded==0)}, Fail={sum(y_encoded==1)}")
    
    return X_clean, y_encoded


# =============================================================================
# BASELINE SONUÇLARI (Karşılaştırma için)
# =============================================================================

BASELINE_RESULTS = {
    'F1_fail': 0.1285,
    'Recall_fail': 0.0771,
    'Precision_fail': 0.4705,
    'PR_AUC': 0.1954,
    'ROC_AUC': 0.7054
}


# =============================================================================
# A) PCA DENEYİ
# =============================================================================

def run_pca_experiment(X, y, variance_levels=[0.90, 0.95, 0.99]):
    """
    PCA ile boyut azaltma deneyi.
    Pipeline: IterativeImputer → RobustScaler → PCA → SMOTE → XGBoost
    """
    print("\n" + "=" * 70)
    print("A) PCA DENEYİ")
    print("Pipeline: Imputer → Scaler → PCA → SMOTE → XGBoost")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    results = []
    
    for var_level in variance_levels:
        print(f"\n--- PCA Varyans: {var_level*100:.0f}% ---")
        start_time = time.time()
        
        fold_scores = {
            'f1_fail': [], 'recall_fail': [], 'precision_fail': [],
            'pr_auc': [], 'roc_auc': [], 'f1_macro': []
        }
        n_components_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            # 1. Imputation
            imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
                max_iter=10, random_state=42
            )
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            # 2. Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            # 3. PCA
            pca = PCA(n_components=var_level, random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            n_components_list.append(pca.n_components_)
            
            # 4. SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)
            
            # 5. Model
            model = XGBClassifier(**xgb_params)
            model.fit(X_train_resampled, y_train_resampled)
            
            # Tahminler
            y_pred = model.predict(X_test_pca)
            y_prob = model.predict_proba(X_test_pca)[:, 1]
            
            # Metrikler
            fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1))
            fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1))
            fold_scores['precision_fail'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
            fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            
            print(f"  Fold {fold_idx+1}/5 ✓ (n_components={pca.n_components_})")
        
        elapsed = time.time() - start_time
        avg_components = int(np.mean(n_components_list))
        
        result = {
            'PCA_Varyans': f"{var_level*100:.0f}%",
            'Bileşen_Sayısı': avg_components,
            'F1_fail_mean': np.mean(fold_scores['f1_fail']),
            'F1_fail_std': np.std(fold_scores['f1_fail']),
            'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
            'Recall_fail_std': np.std(fold_scores['recall_fail']),
            'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
            'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
            'PR_AUC_std': np.std(fold_scores['pr_auc']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'F1_macro_mean': np.mean(fold_scores['f1_macro']),
            'Süre_s': elapsed
        }
        results.append(result)
        
        print(f"  → F1_fail: {result['F1_fail_mean']:.4f} | Recall: {result['Recall_fail_mean']:.4f} | PR-AUC: {result['PR_AUC_mean']:.4f}")
        print(f"  → Süre: {elapsed:.1f}s")
    
    return pd.DataFrame(results)


# =============================================================================
# B) FEATURE SELECTION DENEYİ
# =============================================================================

def get_feature_importance(X, y):
    """
    Tüm veri üzerinde XGBoost eğitip feature importance çıkar.
    Bu importance değerleri feature selection için kullanılacak.
    """
    print("\n[*] Feature importance hesaplanıyor...")
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    
    # Imputation
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
        max_iter=10, random_state=42
    )
    X_imp = imputer.fit_transform(X_array)
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_array)
    
    # Model
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=1.0, random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    model.fit(X_resampled, y_resampled)
    
    # Feature importance
    importance = model.feature_importances_
    
    # Sıralama
    feature_names = X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X_array.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"[*] Top 10 feature:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def run_feature_selection_experiment(X, y, importance_df, top_k_values=[50, 100, 150]):
    """
    Feature Selection deneyi.
    Pipeline: IterativeImputer → RobustScaler → Feature Selection → SMOTE → XGBoost
    """
    print("\n" + "=" * 70)
    print("B) FEATURE SELECTION DENEYİ")
    print("Pipeline: Imputer → Scaler → Top-K Features → SMOTE → XGBoost")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else list(range(X_array.shape[1]))
    
    results = []
    
    for top_k in top_k_values:
        print(f"\n--- Top-{top_k} Features ---")
        start_time = time.time()
        
        # Seçilecek feature'ların indexlerini bul
        top_features = importance_df.head(top_k)['feature'].tolist()
        feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]
        
        if len(feature_indices) < top_k:
            print(f"  [!] Uyarı: {top_k} yerine {len(feature_indices)} feature bulundu")
        
        fold_scores = {
            'f1_fail': [], 'recall_fail': [], 'precision_fail': [],
            'pr_auc': [], 'roc_auc': [], 'f1_macro': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            # 1. Imputation (tüm feature'lar üzerinde)
            imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
                max_iter=10, random_state=42
            )
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            # 2. Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            # 3. Feature Selection (scaled data üzerinde)
            X_train_selected = X_train_scaled[:, feature_indices]
            X_test_selected = X_test_scaled[:, feature_indices]
            
            # 4. SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            
            # 5. Model
            model = XGBClassifier(**xgb_params)
            model.fit(X_train_resampled, y_train_resampled)
            
            # Tahminler
            y_pred = model.predict(X_test_selected)
            y_prob = model.predict_proba(X_test_selected)[:, 1]
            
            # Metrikler
            fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1))
            fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1))
            fold_scores['precision_fail'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
            fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            
            print(f"  Fold {fold_idx+1}/5 ✓")
        
        elapsed = time.time() - start_time
        
        result = {
            'Feature_Sayısı': top_k,
            'F1_fail_mean': np.mean(fold_scores['f1_fail']),
            'F1_fail_std': np.std(fold_scores['f1_fail']),
            'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
            'Recall_fail_std': np.std(fold_scores['recall_fail']),
            'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
            'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
            'PR_AUC_std': np.std(fold_scores['pr_auc']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'F1_macro_mean': np.mean(fold_scores['f1_macro']),
            'Süre_s': elapsed
        }
        results.append(result)
        
        print(f"  → F1_fail: {result['F1_fail_mean']:.4f} | Recall: {result['Recall_fail_mean']:.4f} | PR-AUC: {result['PR_AUC_mean']:.4f}")
        print(f"  → Süre: {elapsed:.1f}s")
    
    return pd.DataFrame(results)


# =============================================================================
# SONUÇ RAPORLAMA
# =============================================================================

def generate_comparison_report(pca_results, fs_results):
    """PCA vs Feature Selection karşılaştırma raporu."""
    
    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMA RAPORU")
    print("=" * 70)
    
    # Baseline değerleri
    print("\n--- BASELINE (442 feature, PCA/FS yok) ---")
    print(f"  F1_fail: {BASELINE_RESULTS['F1_fail']:.4f}")
    print(f"  Recall_fail: {BASELINE_RESULTS['Recall_fail']:.4f}")
    print(f"  PR-AUC: {BASELINE_RESULTS['PR_AUC']:.4f}")
    
    # PCA Sonuçları
    print("\n--- A) PCA SONUÇLARI ---\n")
    pca_display = pca_results[['PCA_Varyans', 'Bileşen_Sayısı', 'F1_fail_mean', 'Recall_fail_mean', 'PR_AUC_mean', 'Süre_s']].copy()
    pca_display.columns = ['Varyans', 'Bileşen', 'F1_fail', 'Recall', 'PR-AUC', 'Süre(s)']
    print(pca_display.to_string(index=False))
    
    # En iyi PCA
    best_pca_idx = pca_results['PR_AUC_mean'].idxmax()
    best_pca = pca_results.loc[best_pca_idx]
    print(f"\n  ★ En iyi PCA: {best_pca['PCA_Varyans']} ({best_pca['Bileşen_Sayısı']} bileşen)")
    print(f"    PR-AUC: {best_pca['PR_AUC_mean']:.4f} (Baseline'dan {best_pca['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']:+.4f})")
    
    # Feature Selection Sonuçları
    print("\n--- B) FEATURE SELECTION SONUÇLARI ---\n")
    fs_display = fs_results[['Feature_Sayısı', 'F1_fail_mean', 'Recall_fail_mean', 'PR_AUC_mean', 'Süre_s']].copy()
    fs_display.columns = ['Features', 'F1_fail', 'Recall', 'PR-AUC', 'Süre(s)']
    print(fs_display.to_string(index=False))
    
    # En iyi Feature Selection
    best_fs_idx = fs_results['PR_AUC_mean'].idxmax()
    best_fs = fs_results.loc[best_fs_idx]
    print(f"\n  ★ En iyi Feature Selection: Top-{int(best_fs['Feature_Sayısı'])} features")
    print(f"    PR-AUC: {best_fs['PR_AUC_mean']:.4f} (Baseline'dan {best_fs['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']:+.4f})")
    
    # Genel Karşılaştırma
    print("\n" + "-" * 70)
    print("GENEL KARŞILAŞTIRMA")
    print("-" * 70)
    
    comparison_data = [
        {
            'Yöntem': 'Baseline (442 feat)',
            'Boyut': 442,
            'F1_fail': BASELINE_RESULTS['F1_fail'],
            'Recall': BASELINE_RESULTS['Recall_fail'],
            'PR-AUC': BASELINE_RESULTS['PR_AUC'],
            'ΔPR-AUC': 0.0
        },
        {
            'Yöntem': f"PCA {best_pca['PCA_Varyans']}",
            'Boyut': best_pca['Bileşen_Sayısı'],
            'F1_fail': best_pca['F1_fail_mean'],
            'Recall': best_pca['Recall_fail_mean'],
            'PR-AUC': best_pca['PR_AUC_mean'],
            'ΔPR-AUC': best_pca['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']
        },
        {
            'Yöntem': f"Top-{int(best_fs['Feature_Sayısı'])} Features",
            'Boyut': int(best_fs['Feature_Sayısı']),
            'F1_fail': best_fs['F1_fail_mean'],
            'Recall': best_fs['Recall_fail_mean'],
            'PR-AUC': best_fs['PR_AUC_mean'],
            'ΔPR-AUC': best_fs['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n")
    print(comparison_df.to_string(index=False))
    
    # Kazanan
    print("\n" + "-" * 70)
    
    if best_pca['PR_AUC_mean'] > best_fs['PR_AUC_mean'] and best_pca['PR_AUC_mean'] > BASELINE_RESULTS['PR_AUC']:
        winner = f"PCA {best_pca['PCA_Varyans']}"
        winner_prauc = best_pca['PR_AUC_mean']
    elif best_fs['PR_AUC_mean'] > BASELINE_RESULTS['PR_AUC']:
        winner = f"Feature Selection (Top-{int(best_fs['Feature_Sayısı'])})"
        winner_prauc = best_fs['PR_AUC_mean']
    else:
        winner = "Baseline (boyut azaltma fayda sağlamadı)"
        winner_prauc = BASELINE_RESULTS['PR_AUC']
    
    print(f"🏆 KAZANAN: {winner}")
    print(f"   PR-AUC: {winner_prauc:.4f}")
    
    return comparison_df


def generate_thesis_text(pca_results, fs_results, comparison_df):
    """Tez için özet paragraf."""
    
    print("\n" + "=" * 70)
    print("TEZ İÇİN ÖZET PARAGRAF")
    print("=" * 70)
    
    best_pca = pca_results.loc[pca_results['PR_AUC_mean'].idxmax()]
    best_fs = fs_results.loc[fs_results['PR_AUC_mean'].idxmax()]
    
    text = f"""
### 4.X.X Boyut Azaltma Tekniklerinin Değerlendirilmesi

SECOM veri setinin yüksek boyutluluğu (442 feature) nedeniyle, model performansı 
üzerindeki etkisini değerlendirmek amacıyla iki farklı boyut azaltma tekniği 
incelenmiştir: Principal Component Analysis (PCA) ve XGBoost tabanlı Feature Selection.

**PCA Analizi:**
Üç farklı varyans açıklama oranı (%90, %95, %99) test edilmiştir. 
{best_pca['PCA_Varyans']} varyans seviyesi {int(best_pca['Bileşen_Sayısı'])} bileşen ile 
en iyi sonucu vermiş, PR-AUC değeri {best_pca['PR_AUC_mean']:.4f} olarak ölçülmüştür.
Baseline'a göre değişim: {best_pca['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']:+.4f}

**Feature Selection Analizi:**
XGBoost feature importance skorlarına göre en önemli 50, 100 ve 150 feature 
seçilerek deneyler yapılmıştır. Top-{int(best_fs['Feature_Sayısı'])} feature ile 
PR-AUC {best_fs['PR_AUC_mean']:.4f} değerine ulaşılmıştır.
Baseline'a göre değişim: {best_fs['PR_AUC_mean'] - BASELINE_RESULTS['PR_AUC']:+.4f}

**Sonuç:**
{'Boyut azaltma teknikleri baseline performansını iyileştirmiştir.' if max(best_pca['PR_AUC_mean'], best_fs['PR_AUC_mean']) > BASELINE_RESULTS['PR_AUC'] else 'Boyut azaltma teknikleri baseline performansını iyileştirmemiştir.'}
{'PCA' if best_pca['PR_AUC_mean'] > best_fs['PR_AUC_mean'] else 'Feature Selection'} yöntemi 
daha yüksek PR-AUC değeri elde etmiştir. Ancak, dengesiz veri setlerinde küçük 
performans farklarının istatistiksel anlamlılığı göz önünde bulundurulmalıdır.

Feature selection yönteminin yorumlanabilirlik avantajı, üretim ortamında hangi 
sensörlerin kritik olduğunun belirlenmesi açısından değerlidir. Bu nedenle, 
operasyonel uygulamalar için feature selection tercih edilebilir.
"""
    
    print(text)
    return text


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv'):
    print("\n")
    print("*" * 70)
    print("  SECOM - BOYUT AZALTMA ANALİZİ (AŞAMA 3)")
    print("  Pipeline: IterativeImputer → RobustScaler → [PCA/FS] → SMOTE → XGBoost")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Feature importance hesapla (Feature Selection için)
    importance_df = get_feature_importance(X_clean, y)
    importance_df.to_csv('secom_feature_importance.csv', index=False)
    print("[*] Feature importance kaydedildi: secom_feature_importance.csv")
    
    # 3. PCA Deneyi
    pca_results = run_pca_experiment(X_clean, y, variance_levels=[0.90, 0.95, 0.99])
    pca_results.to_csv('secom_pca_results.csv', index=False)
    
    # 4. Feature Selection Deneyi
    fs_results = run_feature_selection_experiment(X_clean, y, importance_df, top_k_values=[50, 100, 150])
    fs_results.to_csv('secom_feature_selection_results.csv', index=False)
    
    # 5. Karşılaştırma Raporu
    comparison_df = generate_comparison_report(pca_results, fs_results)
    comparison_df.to_csv('secom_dimension_reduction_comparison.csv', index=False)
    
    # 6. Tez Özeti
    thesis_text = generate_thesis_text(pca_results, fs_results, comparison_df)
    
    # Tez metnini kaydet
    with open('secom_dimension_reduction_thesis.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text)
    
    print("\n" + "=" * 70)
    print("[✓] TÜM SONUÇLAR KAYDEDİLDİ:")
    print("    - secom_feature_importance.csv")
    print("    - secom_pca_results.csv")
    print("    - secom_feature_selection_results.csv")
    print("    - secom_dimension_reduction_comparison.csv")
    print("    - secom_dimension_reduction_thesis.txt")
    print("=" * 70)
    
    return pca_results, fs_results, importance_df


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    pca_results, fs_results, importance_df = main(filepath)
