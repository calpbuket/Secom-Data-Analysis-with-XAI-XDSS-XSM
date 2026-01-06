"""
SECOM Veri Seti - MODEL KARŞILAŞTIRMA (BENCHMARK) - AŞAMA 4
============================================================
Sabit Pipeline: IterativeImputer → RobustScaler → Top-100 Features → SMOTE → Model

Karşılaştırılacak Modeller:
    1. Logistic Regression (baseline klasik model)
    2. Random Forest (ağaç tabanlı ensemble)
    3. XGBoost (gradient boosting - mevcut referans)
    4. LightGBM (hızlı gradient boosting)

Metrikler:
    - F1_macro, F1_fail, Recall_fail, Precision_fail
    - PR-AUC, ROC-AUC
    - Eğitim süresi

Gerekli paketler:
    pip install imbalanced-learn xgboost lightgbm --break-system-packages
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


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
    
    # Önce imputation
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
# MODEL TANIMLARI
# =============================================================================

def get_models():
    """Karşılaştırılacak modelleri döndürür."""
    
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_SEED,
            n_jobs=1
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
        
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1.0,  # SMOTE kullandığımız için 1.0
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
        )
    }
    
    return models


# =============================================================================
# BENCHMARK PIPELINE
# =============================================================================

def run_model_benchmark(X, y, importance_df, top_k=100):
    """
    Tüm modelleri sabit pipeline ile karşılaştırır.
    Pipeline: IterativeImputer → RobustScaler → Top-K Features → SMOTE → Model
    """
    print("\n" + "=" * 70)
    print("MODEL BENCHMARK")
    print(f"Pipeline: Imputer → Scaler → Top-{top_k} Features → SMOTE → Model")
    print("=" * 70)
    
    # Top-K feature seç
    top_features = importance_df['feature'].head(top_k).tolist()
    X_selected = X[top_features]
    print(f"\n[*] Top-{top_k} feature seçildi")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = get_models()
    
    X_array = X_selected.values if hasattr(X_selected, 'values') else X_selected
    y_array = y.values if hasattr(y, 'values') else y
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model_start = time.time()
        
        fold_scores = {
            'f1_macro': [], 'f1_fail': [], 'recall_fail': [],
            'precision_fail': [], 'pr_auc': [], 'roc_auc': [],
            'train_time': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y_array)):
            fold_start = time.time()
            
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            # 1. Imputation
            imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=5, max_depth=5, 
                                              random_state=RANDOM_SEED, n_jobs=1),
                max_iter=5, random_state=RANDOM_SEED
            )
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            # 2. Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            # 3. SMOTE (sadece train üzerinde)
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            # 4. Model eğitimi
            train_start = time.time()
            model_clone = clone_model(model)
            model_clone.fit(X_train_resampled, y_train_resampled)
            train_time = time.time() - train_start
            
            # 5. Tahminler
            y_pred = model_clone.predict(X_test_scaled)
            
            # Probability tahminleri
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model_clone.decision_function(X_test_scaled)
            
            # 6. Metrikler
            fold_scores['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
            fold_scores['f1_fail'].append(f1_score(y_test, y_pred, pos_label=1))
            fold_scores['recall_fail'].append(recall_score(y_test, y_pred, pos_label=1))
            fold_scores['precision_fail'].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
            fold_scores['pr_auc'].append(pr_auc_score(y_test, y_prob))
            fold_scores['roc_auc'].append(roc_auc_score(y_test, y_prob))
            fold_scores['train_time'].append(train_time)
            
            fold_time = time.time() - fold_start
            print(f"  Fold {fold_idx+1}/5 ✓ ({fold_time:.1f}s)")
        
        model_time = time.time() - model_start
        
        result = {
            'Model': model_name,
            'F1_macro_mean': np.mean(fold_scores['f1_macro']),
            'F1_macro_std': np.std(fold_scores['f1_macro']),
            'F1_fail_mean': np.mean(fold_scores['f1_fail']),
            'F1_fail_std': np.std(fold_scores['f1_fail']),
            'Recall_fail_mean': np.mean(fold_scores['recall_fail']),
            'Recall_fail_std': np.std(fold_scores['recall_fail']),
            'Precision_fail_mean': np.mean(fold_scores['precision_fail']),
            'Precision_fail_std': np.std(fold_scores['precision_fail']),
            'PR_AUC_mean': np.mean(fold_scores['pr_auc']),
            'PR_AUC_std': np.std(fold_scores['pr_auc']),
            'ROC_AUC_mean': np.mean(fold_scores['roc_auc']),
            'ROC_AUC_std': np.std(fold_scores['roc_auc']),
            'Train_time_mean': np.mean(fold_scores['train_time']),
            'Total_time': model_time
        }
        results.append(result)
        
        print(f"  → F1_fail: {result['F1_fail_mean']:.4f} (±{result['F1_fail_std']:.4f})")
        print(f"  → Recall: {result['Recall_fail_mean']:.4f} | PR-AUC: {result['PR_AUC_mean']:.4f}")
        print(f"  → Toplam süre: {model_time:.1f}s")
    
    return pd.DataFrame(results)


def clone_model(model):
    """Model klonlama - sklearn clone yerine basit versiyon."""
    from sklearn.base import clone
    return clone(model)


# =============================================================================
# SONUÇ RAPORLAMA
# =============================================================================

def generate_benchmark_report(results_df):
    """Benchmark sonuç raporu oluşturur."""
    
    print("\n" + "=" * 70)
    print("MODEL KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 70)
    
    # Ana tablo
    print("\n--- ÖZET TABLO ---\n")
    
    display_cols = ['Model', 'F1_fail_mean', 'Recall_fail_mean', 
                    'Precision_fail_mean', 'PR_AUC_mean', 'ROC_AUC_mean', 'Total_time']
    display_df = results_df[display_cols].copy()
    display_df.columns = ['Model', 'F1_fail', 'Recall', 'Precision', 'PR-AUC', 'ROC-AUC', 'Süre(s)']
    
    # Formatlama
    for col in ['F1_fail', 'Recall', 'Precision', 'PR-AUC', 'ROC-AUC']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    display_df['Süre(s)'] = display_df['Süre(s)'].apply(lambda x: f"{x:.1f}")
    
    print(display_df.to_string(index=False))
    
    # Metrik bazında en iyi modeller
    print("\n" + "-" * 70)
    print("METRİK BAZINDA EN İYİ MODELLER")
    print("-" * 70)
    
    metrics = {
        'F1_fail_mean': 'F1_fail (Hatalı ürün tespiti)',
        'Recall_fail_mean': 'Recall_fail (Kaçırma oranı)',
        'PR_AUC_mean': 'PR-AUC (Dengesiz veri performansı)',
        'ROC_AUC_mean': 'ROC-AUC (Genel ayırt edicilik)'
    }
    
    for metric, desc in metrics.items():
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_value = results_df.loc[best_idx, metric]
        print(f"  {desc}: {best_model} ({best_value:.4f})")
    
    # Hız karşılaştırması
    print("\n" + "-" * 70)
    print("HIZLI MODEL SIRALAMASI")
    print("-" * 70)
    
    speed_sorted = results_df.sort_values('Total_time')[['Model', 'Total_time']].values
    for i, (model, time_val) in enumerate(speed_sorted, 1):
        print(f"  {i}. {model}: {time_val:.1f}s")
    
    # Genel kazanan belirleme
    print("\n" + "=" * 70)
    print("GENEL DEĞERLENDİRME")
    print("=" * 70)
    
    # Skor hesapla (normalize edilmiş)
    score_df = results_df.copy()
    
    # Her metrik için sıralama puanı (4=en iyi, 1=en kötü)
    for metric in ['F1_fail_mean', 'Recall_fail_mean', 'PR_AUC_mean', 'ROC_AUC_mean']:
        score_df[f'{metric}_rank'] = score_df[metric].rank(ascending=False)
    
    # Toplam skor (düşük = iyi)
    score_df['total_rank'] = (
        score_df['F1_fail_mean_rank'] + 
        score_df['Recall_fail_mean_rank'] + 
        score_df['PR_AUC_mean_rank'] * 1.5 +  # PR-AUC'a daha fazla ağırlık
        score_df['ROC_AUC_mean_rank']
    )
    
    winner_idx = score_df['total_rank'].idxmin()
    winner = results_df.loc[winner_idx]
    
    print(f"""
🏆 ÖNERİLEN MODEL: {winner['Model']}

   Performans Özeti:
   ├─ F1_fail:    {winner['F1_fail_mean']:.4f} (±{winner['F1_fail_std']:.4f})
   ├─ Recall:     {winner['Recall_fail_mean']:.4f} (±{winner['Recall_fail_std']:.4f})
   ├─ Precision:  {winner['Precision_fail_mean']:.4f} (±{winner['Precision_fail_std']:.4f})
   ├─ PR-AUC:     {winner['PR_AUC_mean']:.4f} (±{winner['PR_AUC_std']:.4f})
   ├─ ROC-AUC:    {winner['ROC_AUC_mean']:.4f} (±{winner['ROC_AUC_std']:.4f})
   └─ Eğitim:     {winner['Total_time']:.1f}s

   Neden Bu Model?
   • Dengesiz veri setlerinde PR-AUC kritik metriktir
   • Fail sınıfını yakalamak (Recall) önceliklidir
   • Üretim ortamında hatalı ürün kaçırma maliyeti yüksektir
""")
    
    return winner['Model'], score_df


def generate_thesis_text(results_df, winner_model):
    """Tez için özet paragraf oluşturur."""
    
    print("\n" + "=" * 70)
    print("TEZ İÇİN ÖZET PARAGRAF")
    print("=" * 70)
    
    winner = results_df[results_df['Model'] == winner_model].iloc[0]
    
    # Model sıralaması (PR-AUC'a göre)
    pr_auc_sorted = results_df.sort_values('PR_AUC_mean', ascending=False)
    rankings = []
    for _, row in pr_auc_sorted.iterrows():
        rankings.append(f"{row['Model']} (PR-AUC: {row['PR_AUC_mean']:.4f})")
    
    text = f"""
### 4.X.X Model Karşılaştırma Analizi

Optimal ön işleme pipeline'ı (IterativeImputer → RobustScaler → Top-100 Feature 
Selection → SMOTE) belirlendikten sonra, dört farklı sınıflandırma algoritması 
karşılaştırılmıştır: Logistic Regression, Random Forest, XGBoost ve LightGBM.

**Deney Tasarımı:**
Tüm modeller aynı pipeline üzerinde 5-katlı stratified cross-validation ile 
değerlendirilmiştir. Dengesiz veri setinin doğası gereği, PR-AUC ve Fail sınıfı 
için F1-score öncelikli metrikler olarak belirlenmiştir.

**Sonuçlar:**
{chr(10).join([f'  • {r}' for r in rankings])}

**En İyi Model: {winner_model}**
Bu model, PR-AUC değeri {winner['PR_AUC_mean']:.4f} (±{winner['PR_AUC_std']:.4f}) 
ile en yüksek performansı göstermiştir. Fail sınıfı için:
- F1-score: {winner['F1_fail_mean']:.4f}
- Recall: {winner['Recall_fail_mean']:.4f}
- Precision: {winner['Precision_fail_mean']:.4f}

**Tartışma:**
{"Gradient boosting tabanlı modeller (XGBoost, LightGBM), ağaç yapılarının eksik " if winner_model in ['XGBoost', 'LightGBM'] else ""}
{"değerlere ve aykırı değerlere olan doğal dayanıklılığı sayesinde üstün performans " if winner_model in ['XGBoost', 'LightGBM'] else ""}
{"göstermiştir. " if winner_model in ['XGBoost', 'LightGBM'] else ""}
Yarı iletken üretim sürecinde hatalı ürün tespiti kritik öneme sahiptir; 
kaçırılan her hatalı ürün (False Negative) önemli maliyet ve kalite sorunlarına 
yol açabilir. Bu nedenle, yüksek Recall değeri elde eden {winner_model} modeli 
operasyonel uygulamalar için önerilmektedir.

**Tablo 4.X: Model Karşılaştırma Sonuçları**

| Model | F1_fail | Recall | Precision | PR-AUC | ROC-AUC |
|-------|---------|--------|-----------|--------|---------|
"""
    
    for _, row in results_df.iterrows():
        text += f"| {row['Model']} | {row['F1_fail_mean']:.4f} | {row['Recall_fail_mean']:.4f} | "
        text += f"{row['Precision_fail_mean']:.4f} | {row['PR_AUC_mean']:.4f} | {row['ROC_AUC_mean']:.4f} |\n"
    
    print(text)
    return text


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main(filepath='secom.csv'):
    """Ana fonksiyon."""
    total_start = time.time()
    
    print("\n")
    print("*" * 70)
    print("  SECOM - MODEL KARŞILAŞTIRMA (BENCHMARK) - AŞAMA 4")
    print("  Pipeline: IterativeImputer → Scaler → Top-100 → SMOTE → Model")
    print("*" * 70)
    
    # 1. Veri hazırlama
    X_clean, y = load_and_prepare_data(filepath)
    
    # 2. Feature importance hesapla
    importance_df = get_feature_importance(X_clean, y)
    
    # 3. Model benchmark
    results_df = run_model_benchmark(X_clean, y, importance_df, top_k=100)
    
    # 4. Sonuç raporu
    winner_model, score_df = generate_benchmark_report(results_df)
    
    # 5. Tez metni
    thesis_text = generate_thesis_text(results_df, winner_model)
    
    # 6. Sonuçları kaydet
    results_df.to_csv('secom_model_benchmark_results.csv', index=False)
    print("\n[✓] Sonuçlar 'secom_model_benchmark_results.csv' dosyasına kaydedildi")
    
    with open('secom_model_benchmark_thesis.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text)
    print("[✓] Tez metni 'secom_model_benchmark_thesis.txt' dosyasına kaydedildi")
    
    total_time = time.time() - total_start
    print(f"\n[*] Toplam süre: {total_time/60:.1f} dakika")
    
    return results_df, winner_model


if __name__ == "__main__":
    filepath = "Downloads/Buket/uci-secom.csv"
    results, winner = main(filepath)