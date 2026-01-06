"""
SECOM - MODEL BENCHMARK (GENİŞLETİLMİŞ VERSİYON)
=================================================
Temel modellere ek olarak:
    - Gradient Boosting (sklearn)
    - Naive Bayes (baseline)
    - SVM (opsiyonel - yavaş olabilir)
    - Threshold Optimization (Fail sınıfı için)

Bu dosya ana benchmark'a ek modeller eklemek istersen kullanılabilir.
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
from sklearn.ensemble import (
    ExtraTreesRegressor, 
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc
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
# YARDIMCI FONKSİYONLAR (önceki dosyadan)
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
# GENİŞLETİLMİŞ MODEL LİSTESİ
# =============================================================================

def get_extended_models(include_slow=False):
    """
    Genişletilmiş model listesi.
    
    Args:
        include_slow: True ise SVM de dahil edilir (çok yavaş olabilir)
    """
    
    models = {
        # ===== TEMEL MODELLER =====
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
        
        'NaiveBayes': GaussianNB(),
        
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=1
        ),
        
        # ===== GRADIENT BOOSTING MODELLER =====
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        
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
    }
    
    # SVM çok yavaş olabilir - opsiyonel
    if include_slow:
        models['SVM_RBF'] = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_SEED
        )
    
    return models


# =============================================================================
# THRESHOLD OPTİMİZASYONU
# =============================================================================

def optimize_threshold_for_recall(y_true, y_prob, target_recall=0.7):
    """
    Belirli bir recall hedefine ulaşmak için optimal threshold bulur.
    
    Yarı iletken üretimde hatalı ürün kaçırma maliyeti yüksek olduğundan,
    yüksek recall tercih edilir.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    
    # Target recall'a en yakın threshold'u bul
    for i, r in enumerate(recall):
        if r >= target_recall:
            if i < len(thresholds):
                return thresholds[i], precision[i], recall[i]
    
    # Bulunamazsa default 0.5
    return 0.5, None, None


def evaluate_with_threshold_optimization(y_test, y_prob, thresholds_to_try=[0.3, 0.4, 0.5]):
    """
    Farklı threshold değerlerini dener ve en iyi F1_fail veren threshold'u seçer.
    """
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds_to_try:
        y_pred = (y_prob >= thresh).astype(int)
        
        f1_fail = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        if f1_fail > best_f1:
            best_f1 = f1_fail
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'f1_fail': f1_fail,
                'recall_fail': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                'precision_fail': precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            }
    
    return best_threshold, best_metrics


# =============================================================================
# KARŞILAŞTIRMA TABLOSU OLUŞTURUCU
# =============================================================================

def create_latex_table(results_df):
    """LaTeX formatında tablo oluşturur (tez için)."""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Model Karşılaştırma Sonuçları}
\label{tab:model_benchmark}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{F1$_{fail}$} & \textbf{Recall} & \textbf{Precision} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{Süre(s)} \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        latex += f"{row['Model']} & "
        latex += f"{row['F1_fail_mean']:.4f} & "
        latex += f"{row['Recall_fail_mean']:.4f} & "
        latex += f"{row['Precision_fail_mean']:.4f} & "
        latex += f"{row['PR_AUC_mean']:.4f} & "
        latex += f"{row['ROC_AUC_mean']:.4f} & "
        latex += f"{row['Total_time']:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


# =============================================================================
# DETAYLI BENCHMARK (threshold optimization ile)
# =============================================================================

def run_detailed_benchmark(X, y, importance_df, top_k=100, include_slow=False):
    """
    Detaylı benchmark - threshold optimization dahil.
    """
    from sklearn.base import clone
    
    print("\n" + "=" * 70)
    print("DETAYLI MODEL BENCHMARK (Threshold Optimization)")
    print("=" * 70)
    
    # Top-K feature seç
    top_features = importance_df['feature'].head(top_k).tolist()
    X_selected = X[top_features]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = get_extended_models(include_slow=include_slow)
    
    X_array = X_selected.values
    y_array = y.values if hasattr(y, 'values') else y
    
    results = []
    threshold_results = []
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model_start = time.time()
        
        fold_scores = {
            'f1_macro': [], 'f1_fail': [], 'recall_fail': [],
            'precision_fail': [], 'pr_auc': [], 'roc_auc': [],
            'optimal_threshold': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            # Pipeline
            imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, 
                                              random_state=RANDOM_SEED, n_jobs=1),
                max_iter=5, random_state=RANDOM_SEED
            )
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            # Model
            model_clone = clone(model)
            model_clone.fit(X_train_resampled, y_train_resampled)
            
            # Default threshold (0.5)
            y_pred = model_clone.predict(X_test_scaled)
            
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model_clone.decision_function(X_test_scaled)
            
            # Optimal threshold bul
            opt_thresh, opt_metrics = evaluate_with_threshold_optimization(
                y_test, y_prob, 
                thresholds_to_try=[0.2, 0.3, 0.4, 0.5, 0.6]
            )
            
            # Optimal threshold ile tahmin
            y_pred_opt = (y_prob >= opt_thresh).astype(int)
            
            # Metrikler (optimal threshold ile)
            fold_scores['f1_macro'].append(f1_score(y_test, y_pred_opt, average='macro'))
            fold_scores['f1_fail'].append(f1_score(y_test, y_pred_opt, pos_label=1, zero_division=0))
            fold_scores['recall_fail'].append(recall_score(y_test, y_pred_opt, pos_label=1, zero_division=0))
            fold_scores['precision_fail'].append(precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0))
            fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
            fold_scores['optimal_threshold'].append(opt_thresh)
            
            print(f"  Fold {fold_idx+1}/5 ✓ (opt_thresh={opt_thresh:.2f})")
        
        model_time = time.time() - model_start
        
        result = {
            'Model': model_name,
            'F1_macro_mean': np.mean(fold_scores['f1_macro']),
            'F1_fail_mean': np.mean(fold_scores['f1_fail']),
            'F1_fail_std': np.std(fold_scores['f1_fail']),
            'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
            'Recall_fail_std': np.std(fold_scores['recall_fail']),
            'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
            'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
            'PR_AUC_std': np.std(fold_scores['pr_auc']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'Optimal_threshold': np.mean(fold_scores['optimal_threshold']),
            'Total_time': model_time
        }
        results.append(result)
        
        print(f"  → F1_fail: {result['F1_fail_mean']:.4f} | Recall: {result['Recall_fail_mean']:.4f}")
        print(f"  → PR-AUC: {result['PR_AUC_mean']:.4f} | Opt. Threshold: {result['Optimal_threshold']:.2f}")
    
    return pd.DataFrame(results)


# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    
    print("\n")
    print("*" * 70)
    print("  SECOM - GENİŞLETİLMİŞ MODEL BENCHMARK")
    print("*" * 70)
    
    # Veri hazırla
    X_clean, y = load_and_prepare_data(filepath)
    print(f"Veri boyutu: {X_clean.shape}")
    
    # Feature importance (basit versiyon)
    from sklearn.ensemble import ExtraTreesClassifier
    
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, random_state=42, n_jobs=1),
        max_iter=5, random_state=42
    )
    X_imp = imputer.fit_transform(X_clean)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imp, y)
    
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
    et.fit(X_res, y_res)
    
    importance_df = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': et.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Benchmark çalıştır
    # include_slow=True yaparsanız SVM de eklenir (çok yavaş!)
    results = run_detailed_benchmark(
        X_clean, y, importance_df, 
        top_k=100, 
        include_slow=False
    )
    
    # Sonuçları kaydet
    results.to_csv('secom_extended_benchmark.csv', index=False)
    
    # LaTeX tablosu oluştur
    latex_table = create_latex_table(results)
    with open('secom_benchmark_latex.tex', 'w') as f:
        f.write(latex_table)
    
    print("\n" + "=" * 70)
    print("ÖZET TABLO")
    print("=" * 70)
    print(results[['Model', 'F1_fail_mean', 'Recall_fail_mean', 
                   'PR_AUC_mean', 'Optimal_threshold', 'Total_time']].to_string(index=False))
