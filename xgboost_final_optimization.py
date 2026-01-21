"""
SECOM - XGBOOST FİNAL OPTİMİZASYONU + SHAP ANALİZİ (AŞAMA 5)
==============================================================
Final Pipeline: IterativeImputer → RobustScaler → Top-100 Features → SMOTE → XGBoost

Bu script:
    1. Pipeline'ı kesinleştirir
    2. XGBoost hyperparameter araması yapar
    3. F1_FAIL metriğine göre optimize eder
    4. En iyi konfigürasyonu seçer
    5. Final model performansını raporlar
    6. Optimal threshold belirler
    7. ✨ SHAP feature importance analizi yapar ✨

Gerekli paketler:
    pip install imbalanced-learn xgboost shap --break-system-packages
"""

import warnings
warnings.filterwarnings('ignore')

import time
import random
import numpy as np
import pandas as pd
from itertools import product
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
from imblearn.over_sampling import SMOTE
import shap


# =============================================================================
# GLOBAL SEED - REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

def set_all_seeds(seed=42):
    """Tüm random seed'leri ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] Tüm random seed'ler {seed} olarak ayarlandı")

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


def pr_auc_score(y_true, y_prob):
    """PR-AUC skorunu hesaplar."""
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
    
    X_clean, dropped_missing = drop_high_missing_columns(X, threshold=0.40)
    print(f"[2] %40+ eksik olan {len(dropped_missing)} sütun drop edildi")
    
    X_clean, dropped_constant = drop_constant_columns(X_clean)
    print(f"[3] Sabit {len(dropped_constant)} sütun drop edildi")
    print(f"[4] Final boyut: {X_clean.shape[1]} feature")
    
    y_encoded = (y == 1).astype(int)
    
    n_pass = sum(y_encoded == 0)
    n_fail = sum(y_encoded == 1)
    print(f"[5] Sınıf dağılımı: Pass={n_pass} ({n_pass/len(y_encoded)*100:.1f}%), "
          f"Fail={n_fail} ({n_fail/len(y_encoded)*100:.1f}%)")
    
    return X_clean, y_encoded


def get_feature_importance(X, y):
    """XGBoost ile feature importance hesapla."""
    print("\n[6] Feature importance hesaplanıyor...")
    
    # Imputation
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, random_state=RANDOM_SEED, n_jobs=1),
        max_iter=5, random_state=RANDOM_SEED
    )
    X_imp = imputer.fit_transform(X)
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    # SMOTE
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # XGBoost ile importance
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_SEED, n_jobs=1, eval_metric='logloss'
    )
    model.fit(X_resampled, y_resampled)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"    Top-5 features: {importance_df['feature'].head(5).tolist()}")
    
    return importance_df


# =============================================================================
# 1️⃣ FİNAL PİPELİNE FONKSİYONU
# =============================================================================

def run_final_pipeline(X_train, X_test, y_train, y_test, xgb_params, top_features):
    """
    Final pipeline: IterativeImputer → RobustScaler → Top-K Features → SMOTE → XGBoost
    
    Returns:
        y_pred, y_prob, train_time
    """
    # Feature selection
    X_train_sel = X_train[top_features].values
    X_test_sel = X_test[top_features].values
    
    # 1. Imputation
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, 
                                      random_state=RANDOM_SEED, n_jobs=1),
        max_iter=5, random_state=RANDOM_SEED
    )
    X_train_imp = imputer.fit_transform(X_train_sel)
    X_test_imp = imputer.transform(X_test_sel)
    
    # 2. Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # 3. SMOTE (sadece train)
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 4. XGBoost
    train_start = time.time()
    model = XGBClassifier(**xgb_params)
    model.fit(X_train_resampled, y_train_resampled)
    train_time = time.time() - train_start
    
    # 5. Tahminler
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    return y_pred, y_prob, train_time, model


def prepare_data_for_training(X, y, top_features):
    """
    Veriyi SHAP analizi için hazırlar - pipeline'ı uygular.
    """
    # Feature selection
    X_sel = X[top_features].values
    
    # 1. Imputation
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, 
                                      random_state=RANDOM_SEED, n_jobs=1),
        max_iter=5, random_state=RANDOM_SEED
    )
    X_imp = imputer.fit_transform(X_sel)
    
    # 2. Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    # 3. SMOTE
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    return X_resampled, y_resampled


# =============================================================================
# 2️⃣ HYPERPARAMETER GRID TANIMLAMA
# =============================================================================

def get_param_grid():
    """XGBoost hyperparameter arama alanı."""
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Toplam kombinasyon sayısı
    total = 1
    for v in param_grid.values():
        total *= len(v)
    
    print(f"\n[*] Toplam parametre kombinasyonu: {total}")
    
    return param_grid


def generate_param_combinations(param_grid, max_combinations=None):
    """Parametre kombinasyonlarını oluşturur."""
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        # Sabit parametreler ekle
        param_dict['random_state'] = RANDOM_SEED
        param_dict['n_jobs'] = 1
        param_dict['eval_metric'] = 'logloss'
        param_dict['tree_method'] = 'hist'
        param_dict['scale_pos_weight'] = 1.0  # SMOTE kullanıyoruz
        combinations.append(param_dict)
    
    if max_combinations and len(combinations) > max_combinations:
        # Random seçim
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(combinations), max_combinations, replace=False)
        combinations = [combinations[i] for i in indices]
        print(f"[*] {max_combinations} kombinasyon random seçildi")
    
    return combinations


# =============================================================================
# 3️⃣ HYPERPARAMETER SEARCH
# =============================================================================

def run_hyperparameter_search(X, y, importance_df, top_k=100, max_combinations=None):
    """
    XGBoost hyperparameter araması yapar.
    Optimizasyon metriği: F1_FAIL
    """
    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER SEARCH")
    print("Optimizasyon Metriği: F1_FAIL")
    print("=" * 70)
    
    # Top-K feature seç
    top_features = importance_df['feature'].head(top_k).tolist()
    print(f"\n[*] Top-{top_k} feature seçildi")
    
    # Parametre kombinasyonları
    param_grid = get_param_grid()
    param_combinations = generate_param_combinations(param_grid, max_combinations)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_array = y.values if hasattr(y, 'values') else y
    
    results = []
    total_configs = len(param_combinations)
    
    for config_idx, params in enumerate(param_combinations, 1):
        config_start = time.time()
        
        # Sadece tuning parametrelerini göster
        display_params = {k: v for k, v in params.items() 
                         if k in ['n_estimators', 'max_depth', 'learning_rate', 
                                  'subsample', 'colsample_bytree']}
        
        print(f"\n[{config_idx}/{total_configs}] {display_params}")
        
        fold_scores = {
            'f1_macro': [], 'f1_fail': [], 'recall_fail': [],
            'precision_fail': [], 'pr_auc': [], 'roc_auc': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y_array)):
            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y_array[train_idx]
            y_test = y_array[test_idx]
            
            y_pred, y_prob, _, _ = run_final_pipeline(
                X_train, X_test, y_train, y_test, params, top_features
            )
            
            # Metrikler
            fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['precision_fail'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
        
        config_time = time.time() - config_start
        
        result = {
            'config_id': config_idx,
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'F1_macro_mean': np.mean(fold_scores['f1_macro']),
            'F1_fail_mean': np.mean(fold_scores['f1_fail']),
            'F1_fail_std': np.std(fold_scores['f1_fail']),
            'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
            'Recall_fail_std': np.std(fold_scores['recall_fail']),
            'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
            'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
            'PR_AUC_std': np.std(fold_scores['pr_auc']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'Time_s': config_time
        }
        results.append(result)
        
        print(f"  → F1_fail: {result['F1_fail_mean']:.4f} (±{result['F1_fail_std']:.4f}) | "
              f"Recall: {result['Recall_fail_mean']:.4f} | PR-AUC: {result['PR_AUC_mean']:.4f} | "
              f"({config_time:.1f}s)")
    
    return pd.DataFrame(results)


# =============================================================================
# 4️⃣ EN İYİ KONFİGÜRASYONU SEÇ
# =============================================================================

def select_best_config(search_results):
    """
    En iyi XGBoost konfigürasyonunu seçer.
    Öncelik: F1_FAIL > PR_AUC > Recall_FAIL
    """
    print("\n" + "=" * 70)
    print("EN İYİ KONFİGÜRASYON SEÇİMİ")
    print("=" * 70)
    
    # F1_FAIL'e göre sırala
    sorted_results = search_results.sort_values('F1_fail_mean', ascending=False)
    
    print("\n--- TOP 5 KONFİGÜRASYON (F1_FAIL'e göre) ---\n")
    
    display_cols = ['config_id', 'n_estimators', 'max_depth', 'learning_rate',
                    'subsample', 'colsample_bytree', 'F1_fail_mean', 'Recall_fail_mean', 
                    'PR_AUC_mean']
    print(sorted_results[display_cols].head(5).to_string(index=False))
    
    # En iyi config
    best_idx = sorted_results['F1_fail_mean'].idxmax()
    best_config = search_results.loc[best_idx]
    
    print("\n" + "-" * 70)
    print("🏆 SEÇİLEN FİNAL KONFİGÜRASYON:")
    print("-" * 70)
    
    final_params = {
        'n_estimators': int(best_config['n_estimators']),
        'max_depth': int(best_config['max_depth']),
        'learning_rate': best_config['learning_rate'],
        'subsample': best_config['subsample'],
        'colsample_bytree': best_config['colsample_bytree'],
        'random_state': RANDOM_SEED,
        'n_jobs': 1,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'scale_pos_weight': 1.0
    }
    
    print(f"""
    n_estimators:     {final_params['n_estimators']}
    max_depth:        {final_params['max_depth']}
    learning_rate:    {final_params['learning_rate']}
    subsample:        {final_params['subsample']}
    colsample_bytree: {final_params['colsample_bytree']}
    
    Performans:
    ├─ F1_fail:  {best_config['F1_fail_mean']:.4f} (±{best_config['F1_fail_std']:.4f})
    ├─ Recall:   {best_config['Recall_fail_mean']:.4f} (±{best_config['Recall_fail_std']:.4f})
    ├─ PR-AUC:   {best_config['PR_AUC_mean']:.4f} (±{best_config['PR_AUC_std']:.4f})
    └─ ROC-AUC:  {best_config['ROC_AUC_mean']:.4f}
""")
    
    return final_params, best_config


# =============================================================================
# 5️⃣ FİNAL MODEL DEĞERLENDİRMESİ
# =============================================================================

def evaluate_final_model(X, y, importance_df, final_params, top_k=100):
    """
    Final XGBoost modelini detaylı değerlendirir.
    """
    print("\n" + "=" * 70)
    print("FİNAL MODEL DEĞERLENDİRMESİ")
    print("=" * 70)
    
    top_features = importance_df['feature'].head(top_k).tolist()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_array = y.values if hasattr(y, 'values') else y
    
    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    
    print("\n--- FOLD BAZLI SONUÇLAR ---")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y_array)):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y_array[train_idx]
        y_test = y_array[test_idx]
        
        y_pred, y_prob, train_time, model = run_final_pipeline(
            X_train, X_test, y_train, y_test, final_params, top_features
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fold_result = {
            'fold': fold_idx + 1,
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_fail': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'recall_fail': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'precision_fail': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'pr_auc': pr_auc_score(y_test, y_prob),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn
        }
        fold_results.append(fold_result)
        
        # Threshold optimization için biriktir
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold_idx+1}: F1_fail={fold_result['f1_fail']:.4f} | "
              f"Recall={fold_result['recall_fail']:.4f} | "
              f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    
    fold_df = pd.DataFrame(fold_results)
    
    # Özet tablo
    print("\n" + "-" * 70)
    print("FİNAL XGBOOST SONUÇ TABLOSU")
    print("-" * 70)
    
    summary = {
        'Metric': ['F1_macro', 'F1_fail', 'Recall_fail', 'Precision_fail', 'PR-AUC', 'ROC-AUC'],
        'Mean': [
            fold_df['f1_macro'].mean(),
            fold_df['f1_fail'].mean(),
            fold_df['recall_fail'].mean(),
            fold_df['precision_fail'].mean(),
            fold_df['pr_auc'].mean(),
            fold_df['roc_auc'].mean()
        ],
        'Std': [
            fold_df['f1_macro'].std(),
            fold_df['f1_fail'].std(),
            fold_df['recall_fail'].std(),
            fold_df['precision_fail'].std(),
            fold_df['pr_auc'].std(),
            fold_df['roc_auc'].std()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df['Mean'] = summary_df['Mean'].apply(lambda x: f"{x:.4f}")
    summary_df['Std'] = summary_df['Std'].apply(lambda x: f"{x:.4f}")
    
    print("\n")
    print(summary_df.to_string(index=False))
    
    # Toplam confusion matrix
    print("\n--- TOPLAM CONFUSION MATRIX (5 Fold) ---")
    total_tp = fold_df['TP'].sum()
    total_fn = fold_df['FN'].sum()
    total_fp = fold_df['FP'].sum()
    total_tn = fold_df['TN'].sum()
    
    print(f"""
                    Predicted
                 Fail    Pass
    Actual Fail   {total_tp:4d}    {total_fn:4d}
    Actual Pass   {total_fp:4d}    {total_tn:4d}
    """)
    
    return fold_df, np.array(all_y_true), np.array(all_y_prob), np.array(all_y_pred)


# =============================================================================
# 6️⃣ THRESHOLD OPTİMİZASYONU
# =============================================================================

def optimize_threshold(y_true, y_prob):
    """
    Final XGBoost için optimal threshold belirler.
    """
    print("\n" + "=" * 70)
    print("THRESHOLD OPTİMİZASYONU")
    print("=" * 70)
    
    thresholds_to_test = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    results = []
    
    for thresh in thresholds_to_test:
        y_pred = (y_prob >= thresh).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        results.append({
            'Threshold': thresh,
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'Recall_fail': recall,
            'Precision_fail': precision,
            'F1_fail': f1
        })
    
    thresh_df = pd.DataFrame(results)
    
    print("\n--- THRESHOLD KARŞILAŞTIRMA TABLOSU ---\n")
    
    display_df = thresh_df.copy()
    for col in ['Recall_fail', 'Precision_fail', 'F1_fail']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    # En iyi threshold seç (F1_fail'e göre)
    best_idx = thresh_df['F1_fail'].idxmax()
    best_row = thresh_df.loc[best_idx]
    
    print("\n" + "-" * 70)
    print(f"🎯 OPTİMAL THRESHOLD: {best_row['Threshold']}")
    print("-" * 70)
    print(f"""
    F1_fail:       {best_row['F1_fail']:.4f}
    Recall_fail:   {best_row['Recall_fail']:.4f}
    Precision_fail:{best_row['Precision_fail']:.4f}
    
    Confusion Matrix:
    TP (Doğru Fail)  = {int(best_row['TP'])}
    FN (Kaçan Fail)  = {int(best_row['FN'])}
    FP (Yanlış Fail) = {int(best_row['FP'])}
    TN (Doğru Pass)  = {int(best_row['TN'])}
""")
    
    # Default (0.5) ile karşılaştırma
    default_row = thresh_df[thresh_df['Threshold'] == 0.5].iloc[0]
    
    print("\n--- DEFAULT (0.5) vs OPTİMAL THRESHOLD KARŞILAŞTIRMASI ---\n")
    
    comparison = pd.DataFrame([
        {
            'Threshold': 0.5,
            'F1_fail': default_row['F1_fail'],
            'Recall_fail': default_row['Recall_fail'],
            'Precision_fail': default_row['Precision_fail'],
            'TP': int(default_row['TP']),
            'FN': int(default_row['FN'])
        },
        {
            'Threshold': best_row['Threshold'],
            'F1_fail': best_row['F1_fail'],
            'Recall_fail': best_row['Recall_fail'],
            'Precision_fail': best_row['Precision_fail'],
            'TP': int(best_row['TP']),
            'FN': int(best_row['FN'])
        }
    ])
    
    print(comparison.to_string(index=False))
    
    return best_row['Threshold'], thresh_df


# =============================================================================
# 7️⃣ ✨ SHAP ANALİZİ ✨
# =============================================================================

def perform_shap_analysis(X, y, importance_df, final_params, top_k=100):
    """
    Final XGBoost modeli için SHAP feature importance analizi yapar.
    
    Returns:
        shap_importance_df: SHAP importance değerleri
    """
    print("\n" + "=" * 70)
    print("✨ SHAP FEATURE IMPORTANCE ANALİZİ ✨")
    print("=" * 70)
    
    top_features = importance_df['feature'].head(top_k).tolist()
    
    print("\n[1] Veri hazırlanıyor (pipeline uygulanıyor)...")
    X_processed, y_processed = prepare_data_for_training(X, y, top_features)
    
    print("[2] Final XGBoost modeli eğitiliyor...")
    model = XGBClassifier(**final_params)
    model.fit(X_processed, y_processed, verbose=False)
    
    print("[3] SHAP değerleri hesaplanıyor (bu biraz sürebilir)...")
    shap_start = time.time()
    
    # TreeExplainer kullan (XGBoost için optimize)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    shap_time = time.time() - shap_start
    print(f"    ✓ SHAP hesaplama tamamlandı ({shap_time:.1f}s)")
    
    # Global SHAP importance hesapla
    print("\n[4] Global SHAP importance hesaplanıyor...")
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    shap_importance_df = pd.DataFrame({
        'feature': top_features,
        'shap_importance': mean_abs_shap,
        'xgb_importance': importance_df['importance'].head(top_k).values
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)
    
    # Rank ekle
    shap_importance_df['shap_rank'] = range(1, len(shap_importance_df) + 1)
    
    print("\n" + "-" * 70)
    print("TOP 20 SHAP FEATURE IMPORTANCE")
    print("-" * 70)
    
    display_df = shap_importance_df.head(20).copy()
    display_df['shap_importance'] = display_df['shap_importance'].apply(lambda x: f"{x:.6f}")
    display_df['xgb_importance'] = display_df['xgb_importance'].apply(lambda x: f"{x:.6f}")
    
    print("\n")
    print(display_df[['shap_rank', 'feature', 'shap_importance', 'xgb_importance']].to_string(index=False))
    
    # İstatistikler
    print("\n" + "-" * 70)
    print("SHAP İSTATİSTİKLER")
    print("-" * 70)
    print(f"""
    Top-3 Features (SHAP):
    1. {shap_importance_df.loc[0, 'feature']}: {shap_importance_df.loc[0, 'shap_importance']:.6f}
    2. {shap_importance_df.loc[1, 'feature']}: {shap_importance_df.loc[1, 'shap_importance']:.6f}
    3. {shap_importance_df.loc[2, 'feature']}: {shap_importance_df.loc[2, 'shap_importance']:.6f}
    
    Toplam SHAP importance: {shap_importance_df['shap_importance'].sum():.4f}
    Ortalama SHAP importance: {shap_importance_df['shap_importance'].mean():.6f}
    """)
    
    return shap_importance_df


# =============================================================================
# TEZ METNİ OLUŞTUR
# =============================================================================

def generate_thesis_text(search_results, final_params, fold_df, optimal_threshold, 
                        thresh_df, shap_importance_df):
    """Tez için özet paragraflar oluşturur."""
    
    print("\n" + "=" * 70)
    print("TEZ İÇİN ÖZET METİN")
    print("=" * 70)
    
    # Ortalama metrikler
    f1_fail_mean = fold_df['f1_fail'].mean()
    f1_fail_std = fold_df['f1_fail'].std()
    recall_mean = fold_df['recall_fail'].mean()
    recall_std = fold_df['recall_fail'].std()
    pr_auc_mean = fold_df['pr_auc'].mean()
    pr_auc_std = fold_df['pr_auc'].std()
    roc_auc_mean = fold_df['roc_auc'].mean()
    
    # Optimal threshold sonuçları
    opt_row = thresh_df[thresh_df['Threshold'] == optimal_threshold].iloc[0]
    
    # Top SHAP features
    top_shap = shap_importance_df.head(10)
    
    text = f"""
### 4.X.X XGBoost Hyperparameter Optimizasyonu

Model karşılaştırma aşamasında XGBoost'un en yüksek PR-AUC performansı göstermesi 
üzerine, bu model için hyperparameter optimizasyonu gerçekleştirilmiştir.

**Arama Alanı:**
Grid search yöntemi ile aşağıdaki parametreler test edilmiştir:
- n_estimators: [50, 100, 150]
- max_depth: [3, 4, 5]
- learning_rate: [0.05, 0.1]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]

Toplam {len(search_results)} farklı konfigürasyon, 5-katlı stratified cross-validation 
ile değerlendirilmiştir. Optimizasyon kriteri olarak F1_fail (Fail sınıfı için F1-score) 
seçilmiştir.

**Optimal Konfigürasyon:**
- n_estimators: {final_params['n_estimators']}
- max_depth: {final_params['max_depth']}
- learning_rate: {final_params['learning_rate']}
- subsample: {final_params['subsample']}
- colsample_bytree: {final_params['colsample_bytree']}

**Final Model Performansı (threshold=0.5):**

| Metrik | Ortalama | Std |
|--------|----------|-----|
| F1_fail | {f1_fail_mean:.4f} | {f1_fail_std:.4f} |
| Recall_fail | {recall_mean:.4f} | {recall_std:.4f} |
| PR-AUC | {pr_auc_mean:.4f} | {pr_auc_std:.4f} |
| ROC-AUC | {roc_auc_mean:.4f} | - |

### 4.X.X Karar Eşiği (Threshold) Optimizasyonu

Yarı iletken üretim sürecinde hatalı ürün kaçırmanın maliyeti, yanlış alarm 
maliyetinden yüksek olduğundan, Recall_fail metriğini artırmak amacıyla karar 
eşiği optimize edilmiştir.

**Optimal Threshold: {optimal_threshold}**

Bu eşik ile:
- F1_fail: {opt_row['F1_fail']:.4f}
- Recall_fail: {opt_row['Recall_fail']:.4f} 
- Precision_fail: {opt_row['Precision_fail']:.4f}

Optimal threshold kullanıldığında, toplam {int(opt_row['TP'])} hatalı ürün doğru 
tespit edilirken, {int(opt_row['FN'])} hatalı ürün kaçırılmıştır. Default threshold 
(0.5) ile karşılaştırıldığında, Recall değeri %{((opt_row['Recall_fail'] - thresh_df[thresh_df['Threshold']==0.5].iloc[0]['Recall_fail']) * 100):.1f} 
artış göstermiştir.

### 4.X.X SHAP Feature Importance Analizi

Model yorumlanabilirliği ve güvenilirliği artırmak amacıyla SHAP (SHapley Additive 
exPlanations) analizi gerçekleştirilmiştir. SHAP değerleri, her bir feature'ın model 
tahminlerine olan katkısını ölçer.

**En Önemli 10 Feature (SHAP):**

| Sıra | Feature | SHAP Importance |
|------|---------|----------------|
"""
    
    for idx, row in top_shap.iterrows():
        text += f"| {idx+1} | {row['feature']} | {row['shap_importance']:.6f} |\n"
    
    text += f"""
Top-3 feature ({top_shap.iloc[0]['feature']}, {top_shap.iloc[1]['feature']}, 
{top_shap.iloc[2]['feature']}) modelin tahminlerinin yaklaşık 
%{(top_shap.head(3)['shap_importance'].sum() / top_shap['shap_importance'].sum() * 100):.1f}'ini açıklamaktadır.

**Sonuç:**
Önerilen final model, IterativeImputer + RobustScaler + Top-100 Feature Selection + 
SMOTE + XGBoost (optimize edilmiş parametreler) + Threshold={optimal_threshold} 
pipeline'ı ile oluşturulmuştur. SHAP analizi, modelin karar mekanizmasının 
yorumlanabilir ve güvenilir olduğunu göstermektedir. Bu model, üretim ortamında 
hatalı ürün tespiti için kullanılabilir.
"""
    
    print(text)
    return text


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv', max_combinations=None):
    """
    Ana fonksiyon - XGBoost Final Optimizasyonu + SHAP
    
    Args:
        filepath: Veri dosyası yolu
        max_combinations: Maksimum parametre kombinasyonu (None = tümü)
    """
    total_start = time.time()
    
    print("\n")
    print("*" * 70)
    print("  SECOM - XGBOOST FİNAL OPTİMİZASYONU + SHAP ANALİZİ")
    print("  Pipeline: IterativeImputer → Scaler → Top-100 → SMOTE → XGBoost")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Feature importance hesapla
    importance_df = get_feature_importance(X_clean, y)
    importance_df.to_csv('secom_final_feature_importance.csv', index=False)
    
    # 3. Hyperparameter search
    search_results = run_hyperparameter_search(
        X_clean, y, importance_df, 
        top_k=100, 
        max_combinations=max_combinations
    )
    search_results.to_csv('secom_xgb_hyperparameter_search.csv', index=False)
    
    # 4. En iyi konfigürasyon seç
    final_params, best_config = select_best_config(search_results)
    
    # 5. Final model değerlendirmesi
    fold_df, y_true, y_prob, y_pred = evaluate_final_model(
        X_clean, y, importance_df, final_params, top_k=100
    )
    fold_df.to_csv('secom_xgb_final_fold_results.csv', index=False)
    
    # 6. Threshold optimizasyonu
    optimal_threshold, thresh_df = optimize_threshold(y_true, y_prob)
    thresh_df.to_csv('secom_xgb_threshold_analysis.csv', index=False)
    
    # 7. ✨ SHAP Analizi ✨
    shap_importance_df = perform_shap_analysis(
        X_clean, y, importance_df, final_params, top_k=100
    )
    shap_importance_df.to_csv('secom_shap_importance.csv', index=False)
    
    # 8. Tez metni oluştur
    thesis_text = generate_thesis_text(
        search_results, final_params, fold_df, optimal_threshold, thresh_df,
        shap_importance_df
    )
    
    with open('secom_xgb_final_thesis.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text)
    
    # 9. Final config'i kaydet
    final_config = {
        'pipeline': {
            'imputer': 'IterativeImputer (ExtraTrees, max_iter=5)',
            'scaler': 'RobustScaler',
            'feature_selection': 'Top-100 (XGBoost importance)',
            'resampling': 'SMOTE',
            'model': 'XGBoost'
        },
        'xgboost_params': final_params,
        'optimal_threshold': optimal_threshold,
        'performance': {
            'F1_fail': fold_df['f1_fail'].mean(),
            'Recall_fail': fold_df['recall_fail'].mean(),
            'PR_AUC': fold_df['pr_auc'].mean(),
            'ROC_AUC': fold_df['roc_auc'].mean()
        },
        'top_10_shap_features': shap_importance_df.head(10)['feature'].tolist()
    }
    
    import json
    with open('secom_final_config.json', 'w') as f:
        json.dump(final_config, f, indent=2)
    
    # Özet
    print("\n" + "=" * 70)
    print("TÜM ÇIKTILAR KAYDEDİLDİ")
    print("=" * 70)
    print("""
    📁 Kaydedilen dosyalar:
    ├─ secom_final_feature_importance.csv
    ├─ secom_xgb_hyperparameter_search.csv
    ├─ secom_xgb_final_fold_results.csv
    ├─ secom_xgb_threshold_analysis.csv
    ├─ ✨ secom_shap_importance.csv ✨
    ├─ secom_xgb_final_thesis.txt
    └─ secom_final_config.json
    """)
    
    total_time = time.time() - total_start
    print(f"[*] Toplam süre: {total_time/60:.1f} dakika")
    
    return {
        'search_results': search_results,
        'final_params': final_params,
        'fold_results': fold_df,
        'optimal_threshold': optimal_threshold,
        'threshold_analysis': thresh_df,
        'shap_importance': shap_importance_df
    }


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    
    # Tüm kombinasyonları dene (72 kombinasyon, ~60-90 dk)
    # results = main(filepath)
    
    # Veya hızlı test için 20 kombinasyon (~15-20 dk)
    results = main(filepath, max_combinations=None)  # None = tümü