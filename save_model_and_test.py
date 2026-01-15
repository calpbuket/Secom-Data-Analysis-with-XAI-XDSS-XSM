"""
Model ve Test Verisi Kaydetme - XAI Analizi İçin Hazırlık
===========================================================
Bu scripti final_pipeline.py'ı çalıştırdıktan sonra kullanın.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================================================
# VERİ HAZIRLIK FONKSİYONLARI
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

# =============================================================================
# MODEL VE TEST VERİSİ KAYDETME
# =============================================================================

def save_model_and_test_data(filepath, output_dir='./save_model_and_test_outputs/'):
    """
    Final modeli ve test verisini kaydeder.
    
    Kaydedilecek dosyalar:
    - xai_final_model.pkl: Eğitilmiş XGBoost modeli
    - xai_test_X.csv: Test feature'ları (impute + scale edilmiş)
    - xai_test_y.csv: Test hedef değişkeni
    - xai_test_X_original.csv: Test feature'ları (raw hali - sadece referans)
    - xai_feature_names.txt: Feature isimleri
    - xai_top100_features.txt: Top-100 feature listesi
    """
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "=" * 70)
    print("MODEL VE TEST VERİSİ KAYDETME")
    print("=" * 70)
    
    # 1. Veri yükleme ve hazırlama
    print("\n[1] Veri yükleniyor...")
    df = pd.read_csv(filepath)
    
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    y = df['Pass/Fail']
    X = df.drop(columns=['Pass/Fail'])
    
    X_clean, _ = drop_high_missing_columns(X, threshold=0.40)
    X_clean, _ = drop_constant_columns(X_clean)
    
    y_encoded = (y == 1).astype(int)
    
    print(f"    Temizlenmiş veri: {X_clean.shape}")
    
    # 2. Train-test split (stratified)
    print("\n[2] Train-Test split yapılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"    Train: {X_train.shape}")
    print(f"    Test: {X_test.shape}")
    
    # Test verisinin orijinal halini kaydet (referans için)
    X_test.to_csv(f'{output_dir}xai_test_X_original.csv', index=False)
    print(f"    ✓ Orijinal test verisi: {output_dir}xai_test_X_original.csv")
    
    # 3. Imputation
    print("\n[3] IterativeImputer uygulanıyor...")
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
        max_iter=10,
        random_state=42
    )
    
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    print("    ✓ Imputation tamamlandı")
    
    # 4. Scaling
    print("\n[4] RobustScaler uygulanıyor...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    print("    ✓ Scaling tamamlandı")
    
    # 5. Feature importance ile Top-100 seçimi
    print("\n[5] Feature importance hesaplanıyor...")
    
    # Önce SMOTE ile dengeleme (sadece importance hesabı için)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Temp model ile importance hesapla
    temp_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1
    )
    temp_model.fit(X_train_resampled, y_train_resampled)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': temp_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_100_features = importance_df['feature'].head(100).tolist()
    
    print(f"    ✓ Top-100 feature seçildi")
    print(f"    En önemli 5 feature:")
    for i, row in importance_df.head(5).iterrows():
        print(f"      {i+1}. {row['feature']}: {row['importance']:.6f}")
    
    # Top-100 feature'ları kaydet
    with open(f'{output_dir}xai_top100_features.txt', 'w') as f:
        for feat in top_100_features:
            f.write(f"{feat}\n")
    print(f"    ✓ Top-100 liste: {output_dir}xai_top100_features.txt")
    
    # 6. Top-100 ile final model eğitimi
    print("\n[6] Final model eğitiliyor (Top-100 features ile)...")
    
    # Train verisinde Top-100 seç
    X_train_top100 = pd.DataFrame(X_train_scaled, columns=X_clean.columns)[top_100_features]
    X_test_top100 = pd.DataFrame(X_test_scaled, columns=X_clean.columns)[top_100_features]
    
    # SMOTE (sadece train)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_top100, y_train)
    
    # Final model
    final_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist'
    )
    
    final_model.fit(X_train_resampled, y_train_resampled)
    print("    ✓ Final model eğitildi")
    
    # 7. Test performansı
    print("\n[7] Test performansı hesaplanıyor...")
    y_pred = final_model.predict(X_test_top100)
    y_prob = final_model.predict_proba(X_test_top100)[:, 1]
    
    from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_recall_curve, auc
    
    def pr_auc_score(y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        return auc(recall, precision)
    
    test_f1_fail = f1_score(y_test, y_pred, pos_label=1)
    test_recall = recall_score(y_test, y_pred, pos_label=1)
    test_roc_auc = roc_auc_score(y_test, y_prob)
    test_pr_auc = pr_auc_score(y_test, y_prob)
    
    print(f"    F1_fail: {test_f1_fail:.4f}")
    print(f"    Recall: {test_recall:.4f}")
    print(f"    ROC-AUC: {test_roc_auc:.4f}")
    print(f"    PR-AUC: {test_pr_auc:.4f}")
    
    # 8. Dosyaları kaydet
    print("\n[8] Dosyalar kaydediliyor...")
    
    # Model
    with open(f'{output_dir}xai_final_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print(f"    ✓ Model: {output_dir}xai_final_model.pkl")
    
    # Test verisi (processed - Top-100)
    X_test_df = pd.DataFrame(X_test_top100, columns=top_100_features)
    X_test_df.to_csv(f'{output_dir}xai_test_X.csv', index=False)
    print(f"    ✓ Test X: {output_dir}xai_test_X.csv")
    
    y_test_df = pd.DataFrame({'Pass/Fail': y_test.values})
    y_test_df.to_csv(f'{output_dir}xai_test_y.csv', index=False)
    print(f"    ✓ Test y: {output_dir}xai_test_y.csv")
    
    # Feature isimleri
    with open(f'{output_dir}xai_feature_names.txt', 'w') as f:
        for feat in top_100_features:
            f.write(f"{feat}\n")
    print(f"    ✓ Feature names: {output_dir}xai_feature_names.txt")
    
    # Tahminleri de kaydet (XAI için gerekli)
    predictions_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    predictions_df.to_csv(f'{output_dir}xai_test_predictions.csv', index=False)
    print(f"    ✓ Predictions: {output_dir}xai_test_predictions.csv")
    
    # Pipeline bileşenlerini de kaydet (opsiyonel)
    pipeline_components = {
        'imputer': imputer,
        'scaler': scaler,
        'top_100_features': top_100_features
    }
    with open(f'{output_dir}xai_pipeline_components.pkl', 'wb') as f:
        pickle.dump(pipeline_components, f)
    print(f"    ✓ Pipeline components: {output_dir}xai_pipeline_components.pkl")
    
    print("\n" + "=" * 70)
    print("✓ TÜM DOSYALAR BAŞARIYLA KAYDEDİLDİ!")
    print("=" * 70)
    print(f"\nKaydedilen dosyalar ({output_dir}):")
    print("  1. xai_final_model.pkl")
    print("  2. xai_test_X.csv")
    print("  3. xai_test_y.csv")
    print("  4. xai_test_X_original.csv")
    print("  5. xai_test_predictions.csv")
    print("  6. xai_top100_features.txt")
    print("  7. xai_feature_names.txt")
    print("  8. xai_pipeline_components.pkl")
    print("\nBu dosyaları 'xai_analysis.py' scripti ile kullanabilirsiniz.")
    
    return final_model, X_test_top100, y_test

# =============================================================================
# ÇALIŞTIR
# =============================================================================

if __name__ == "__main__":
    # Veri dosyası yolu - KENDİ YOLUNUZU YAZIN
    filepath = "./uci-secom.csv"
    
    # Model ve test verisini kaydet
    model, X_test, y_test = save_model_and_test_data(filepath)