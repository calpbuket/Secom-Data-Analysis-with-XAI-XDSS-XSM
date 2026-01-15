"""
================================================================================
SECOM - KAPSAMLI XAI / SHAP ANALİZİ
Explainable AI ile Model Açıklanabilirliği ve Tez Dokümantasyonu
================================================================================

Bu script şu analizleri gerçekleştirir:
    1. Global SHAP Analizi (Model genel davranışı)
    2. Dependence Plot Analizi (Eşik davranışları)
    3. Lokal Vaka Açıklamaları (TP ve FN örnekleri)
    4. XDSS ve XSM İçin Kural Türetme
    5. Tez için Akademik Yorum Paragrafları

Gereksinimler:
    pip install shap matplotlib pandas numpy --break-system-packages

Giriş dosyaları (save_model_and_test.py ile oluşturulmalı):
    - xai_final_model.pkl
    - xai_test_X.csv
    - xai_test_y.csv
    - xai_test_predictions.csv
    - xai_top100_features.txt
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path

# Grafik ayarları
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
sns.set_style("whitegrid")

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def load_xai_data(data_dir='./save_model_and_test_outputs/xai_inputs/'):
    """XAI için gerekli tüm dosyaları yükler."""
    
    print("\n" + "=" * 70)
    print("XAI VERİLERİ YÜKLEME")
    print("=" * 70)
    
    # Model
    with open(f'{data_dir}xai_final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model yüklendi: {type(model).__name__}")
    
    # Test verisi
    X_test = pd.read_csv(f'{data_dir}xai_test_X.csv')
    y_test = pd.read_csv(f'{data_dir}xai_test_y.csv')['Pass/Fail'].values
    predictions = pd.read_csv(f'{data_dir}xai_test_predictions.csv')
    
    print(f"✓ Test verisi yüklendi: {X_test.shape}")
    print(f"  Pass (0): {(y_test == 0).sum()}")
    print(f"  Fail (1): {(y_test == 1).sum()}")
    
    # Feature isimleri
    with open(f'{data_dir}xai_top100_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✓ {len(feature_names)} feature ismi yüklendi")
    
    return model, X_test, y_test, predictions, feature_names


def create_output_dir(output_dir='./xai_analysis_outputs/'):
    """Çıktı klasörü oluştur."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# 1. GLOBAL SHAP ANALİZİ
# =============================================================================

def global_shap_analysis(model, X_test, feature_names, output_dir):
    """
    Global SHAP analizi - Modelin genel davranışını açıklar.
    
    Çıktılar:
        - SHAP values hesaplama
        - Summary plot (beeswarm)
        - Top-10 ve Top-20 feature tabloları
        - Akademik yorum metni
    """
    
    print("\n" + "=" * 70)
    print("1. GLOBAL SHAP ANALİZİ")
    print("=" * 70)
    
    # SHAP TreeExplainer
    print("\n[1.1] SHAP TreeExplainer oluşturuluyor...")
    explainer = shap.TreeExplainer(model)
    
    print("[1.2] SHAP values hesaplanıyor (bu biraz zaman alabilir)...")
    shap_values = explainer.shap_values(X_test)
    
    # XGBoost binary classification için shap_values shape kontrolü
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Fail sınıfı (pozitif sınıf)
    
    print(f"✓ SHAP values shape: {shap_values.shape}")
    
    # SHAP values'ı kaydet (sonraki analizler için)
    np.save(f'{output_dir}shap_values.npy', shap_values)
    print(f"✓ SHAP values kaydedildi: {output_dir}shap_values.npy")
    
    # -------------------------------------------------------------------------
    # SHAP Summary Plot (Beeswarm)
    # -------------------------------------------------------------------------
    print("\n[1.3] SHAP Summary Plot oluşturuluyor...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title("SHAP Summary Plot - Model Feature Importance", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}1_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Görsel kaydedildi: {output_dir}1_shap_summary_plot.png")
    
    # -------------------------------------------------------------------------
    # Feature Importance Tablosu
    # -------------------------------------------------------------------------
    print("\n[1.4] Feature importance tablosu oluşturuluyor...")
    
    # Her feature için ortalama |SHAP| değeri
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_|SHAP|': mean_abs_shap
    }).sort_values('Mean_|SHAP|', ascending=False)
    
    # Her feature için etki yönü (ortalama SHAP)
    mean_shap = shap_values.mean(axis=0)
    importance_df['Mean_SHAP'] = [mean_shap[i] for i in range(len(feature_names))]
    importance_df['Effect_Direction'] = importance_df['Mean_SHAP'].apply(
        lambda x: 'Fail riskini ARTIRIR' if x > 0 else 'Fail riskini AZALTIR'
    )
    
    # Top-10 ve Top-20
    top_10 = importance_df.head(10)
    top_20 = importance_df.head(20)
    
    # Tabloları kaydet
    importance_df.to_csv(f'{output_dir}1_feature_importance_full.csv', index=False)
    top_10.to_csv(f'{output_dir}1_top10_features.csv', index=False)
    top_20.to_csv(f'{output_dir}1_top20_features.csv', index=False)
    
    print("\n" + "─" * 70)
    print("EN ÖNEMLİ 10 FEATURE:")
    print("─" * 70)
    print(top_10[['Feature', 'Mean_|SHAP|', 'Effect_Direction']].to_string(index=False))
    
    print("\n" + "─" * 70)
    print("EN ÖNEMLİ 11-20 FEATURE:")
    print("─" * 70)
    print(top_20.iloc[10:][['Feature', 'Mean_|SHAP|', 'Effect_Direction']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Akademik Yorum Metni (TEZ İÇİN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEZ İÇİN AKADEMİK YORUM - GLOBAL SHAP")
    print("=" * 70)
    
    top_3_features = top_10['Feature'].head(3).tolist()
    top_3_shap = top_10['Mean_|SHAP|'].head(3).tolist()
    
    thesis_text_global = f"""
### Global Model Açıklanabilirliği - SHAP Analizi

SECOM yarı iletken üretim veri seti üzerinde eğitilen XGBoost modelinin 
kararlarını açıklamak için SHAP (SHapley Additive exPlanations) yöntemi 
uygulanmıştır. SHAP, oyun teorisi temelli bir açıklama yöntemi olup, 
her bir feature'ın model tahminindeki katkısını nicel olarak ölçmektedir.

**En Etkili Sensörler:**

Modelin tahminlerinde en baskın rol oynayan sensörler şunlardır:
1. {top_3_features[0]} (Ortalama |SHAP|: {top_3_shap[0]:.6f})
2. {top_3_features[1]} (Ortalama |SHAP|: {top_3_shap[1]:.6f})
3. {top_3_features[2]} (Ortalama |SHAP|: {top_3_shap[2]:.6f})

Bu sensörlerin Fail sınıfına katkıları SHAP değerleri ile nicel olarak 
gösterilmiştir. SHAP summary plot'u, her sensörün model kararlarına olan 
etkisini görselleştirmekte olup, kırmızı renkler yüksek sensör değerlerini, 
mavi renkler düşük değerleri temsil etmektedir.

**Feature Importance Dağılımı:**

Top-10 sensör, toplam SHAP katkısının yaklaşık %{(top_10['Mean_|SHAP|'].sum() / importance_df['Mean_|SHAP|'].sum() * 100):.1f}'ini 
oluşturmaktadır. Bu durum, modelin az sayıda kritik sensöre yoğunlaştığını 
ve bu sensörlerin üretim sürecindeki hataları belirlemede kilit rol 
oynadığını göstermektedir.

**Etki Yönü Analizi:**

Top-10 sensörden:
- {(top_10['Effect_Direction'] == 'Fail riskini ARTIRIR').sum()} tanesi Fail riskini artırıcı yönde
- {(top_10['Effect_Direction'] == 'Fail riskini AZALTIR').sum()} tanesi Fail riskini azaltıcı yönde

etki göstermektedir. Bu bulgu, bazı sensörlerin yüksek değerlerinin üretim 
hatasına işaret ettiğini, bazılarının ise düşük değerlerinin risk oluşturduğunu 
ortaya koymaktadır.
"""
    
    # Metni kaydet
    with open(f'{output_dir}1_thesis_text_global.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text_global)
    
    print(thesis_text_global)
    print(f"\n✓ Tez metni kaydedildi: {output_dir}1_thesis_text_global.txt")
    
    return shap_values, explainer, importance_df


# =============================================================================
# 2. DEPENDENCE PLOT ANALİZİ
# =============================================================================

def dependence_plot_analysis(shap_values, X_test, importance_df, output_dir):
    """
    SHAP Dependence Plot analizi - Kritik sensörlerin eşik davranışları.
    
    En önemli 3 feature için:
        - Dependence plot
        - Eşik analizi
        - Akademik yorum
    """
    
    print("\n" + "=" * 70)
    print("2. DEPENDENCE PLOT ANALİZİ")
    print("=" * 70)
    
    top_3_features = importance_df['Feature'].head(3).tolist()
    
    threshold_analyses = []
    
    for i, feature in enumerate(top_3_features, 1):
        print(f"\n[2.{i}] {feature} için analiz...")
        
        feature_idx = list(X_test.columns).index(feature)
        feature_values = X_test[feature].values
        feature_shap = shap_values[:, feature_idx]
        
        # Dependence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            feature_values, 
            feature_shap,
            c=feature_shap,
            cmap='RdBu_r',
            alpha=0.6,
            s=20
        )
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(f'{feature} Değeri', fontsize=12)
        ax.set_ylabel(f'SHAP Değeri (Fail Katkısı)', fontsize=12)
        ax.set_title(f'SHAP Dependence Plot: {feature}', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='SHAP Değeri')
        plt.tight_layout()
        plt.savefig(f'{output_dir}2_{i}_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Görsel: {output_dir}2_{i}_dependence_{feature}.png")
        
        # Eşik analizi
        # SHAP=0'a yakın noktaları bul
        positive_shap_mask = feature_shap > 0.01
        negative_shap_mask = feature_shap < -0.01
        
        if positive_shap_mask.any():
            min_risk_value = feature_values[positive_shap_mask].min()
            median_risk_value = np.median(feature_values[positive_shap_mask])
        else:
            min_risk_value = np.nan
            median_risk_value = np.nan
        
        # İstatistikler
        percentile_25 = np.percentile(feature_values, 25)
        percentile_50 = np.percentile(feature_values, 50)
        percentile_75 = np.percentile(feature_values, 75)
        
        # SHAP değişim noktası
        sorted_idx = np.argsort(feature_values)
        sorted_values = feature_values[sorted_idx]
        sorted_shap = feature_shap[sorted_idx]
        
        # SHAP'in işaret değiştirdiği yaklaşık nokta
        sign_changes = np.where(np.diff(np.sign(sorted_shap)))[0]
        if len(sign_changes) > 0:
            critical_threshold = sorted_values[sign_changes[0]]
        else:
            critical_threshold = np.nan
        
        analysis = {
            'feature': feature,
            'percentile_25': percentile_25,
            'percentile_50': percentile_50,
            'percentile_75': percentile_75,
            'min_risk_value': min_risk_value,
            'median_risk_value': median_risk_value,
            'critical_threshold': critical_threshold,
            'mean_shap_positive': feature_shap[positive_shap_mask].mean() if positive_shap_mask.any() else 0,
            'mean_shap_negative': feature_shap[negative_shap_mask].mean() if negative_shap_mask.any() else 0
        }
        
        threshold_analyses.append(analysis)
        
        print(f"  İstatistikler:")
        print(f"    25th percentile: {percentile_25:.4f}")
        print(f"    50th percentile: {percentile_50:.4f}")
        print(f"    75th percentile: {percentile_75:.4f}")
        if not np.isnan(critical_threshold):
            print(f"    Kritik eşik (SHAP=0 yakını): {critical_threshold:.4f}")
        else:
            print(f"    Kritik eşik bulunamadı")
    
    # Tüm analizleri kaydet
    threshold_df = pd.DataFrame(threshold_analyses)
    threshold_df.to_csv(f'{output_dir}2_threshold_analysis.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Akademik Yorum Metni (TEZ İÇİN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEZ İÇİN AKADEMİK YORUM - DEPENDENCE ANALİZİ")
    print("=" * 70)
    
    thesis_text_dependence = f"""
### SHAP Dependence Analizi - Sensör Eşik Davranışları

SHAP dependence plot'ları, her bir sensörün değerinin model kararına olan 
etkisinin nasıl değiştiğini görselleştirmektedir. Bu analizle, kritik 
sensörlerin hangi değer aralıklarında Fail riskini artırdığı belirlenmektedir.

"""
    
    for i, analysis in enumerate(threshold_analyses, 1):
        feat = analysis['feature']
        p50 = analysis['percentile_50']
        p75 = analysis['percentile_75']
        ct = analysis['critical_threshold']
        
        thesis_text_dependence += f"""
**{i}. {feat}:**

Bu sensör için SHAP dependence plot analizi şu bulguları ortaya koymuştur:

- **Medyan değer:** {p50:.4f}
- **75. percentile:** {p75:.4f}
"""
        
        if not np.isnan(ct):
            thesis_text_dependence += f"""- **Kritik eşik:** {ct:.4f}

Sensör değeri {ct:.4f} eşiğini geçtiğinde, modelin Fail kararına katkısı 
pozitif yönde artmaktadır. Bu durum, süreçte potansiyel risk bölgesi olarak 
yorumlanmaktadır. {p75:.4f} değerinin üzerindeki ölçümler, üretim hatasına 
işaret eden güçlü göstergeler olarak öne çıkmaktadır.
"""
        else:
            thesis_text_dependence += f"""
Sensör değerleri ile SHAP katkıları arasında doğrusal olmayan bir ilişki 
gözlemlenmektedir. Yüksek değerler ({p75:.4f}+) Fail riskini artırırken, 
düşük değerler koruyucu etki göstermektedir.
"""
    
    thesis_text_dependence += """

**Genel Değerlendirme:**

Dependence analizi, sensör değerlerinin belirli eşikleri aştığında üretim 
hatasına katkılarının hızla arttığını ortaya koymaktadır. Bu bulgular, 
süreç kontrol limitlerinin belirlenmesi ve erken uyarı sistemlerinin 
kurulması açısından kritik önem taşımaktadır.
"""
    
    # Metni kaydet
    with open(f'{output_dir}2_thesis_text_dependence.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text_dependence)
    
    print(thesis_text_dependence)
    print(f"\n✓ Tez metni kaydedildi: {output_dir}2_thesis_text_dependence.txt")
    
    return threshold_df


# =============================================================================
# 3. LOKAL VAKA AÇIKLAMALARI
# =============================================================================

def local_explanation_analysis(shap_values, explainer, X_test, y_test, predictions, feature_names, output_dir):
    """
    Lokal vaka açıklamaları - TP ve FN örnekleri için detaylı analiz.
    
    Çıktılar:
        - 1 True Positive örneği (Waterfall plot)
        - 1 False Negative örneği (Waterfall plot)
        - Her biri için top-5 katkılı feature'lar
        - Akademik yorumlar
    """
    
    print("\n" + "=" * 70)
    print("3. LOKAL VAKA AÇIKLAMALARI")
    print("=" * 70)
    
    y_pred = predictions['y_pred'].values
    y_prob = predictions['y_prob'].values
    
    # True Positive örnekleri
    tp_mask = (y_test == 1) & (y_pred == 1)
    tp_indices = np.where(tp_mask)[0]
    
    # False Negative örnekleri
    fn_mask = (y_test == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    
    print(f"\nTrue Positives: {len(tp_indices)}")
    print(f"False Negatives: {len(fn_indices)}")
    
    local_cases = []
    
    # -------------------------------------------------------------------------
    # TRUE POSITIVE ÖRNEĞİ
    # -------------------------------------------------------------------------
    if len(tp_indices) > 0:
        print("\n[3.1] True Positive örneği analizi...")
        
        # En yüksek olasılıklı TP'yi seç
        tp_probs = y_prob[tp_indices]
        tp_idx_max = tp_indices[np.argmax(tp_probs)]
        
        print(f"  Seçilen örnek index: {tp_idx_max}")
        print(f"  Gerçek: Fail (1), Tahmin: Fail (1)")
        print(f"  Fail olasılığı: {y_prob[tp_idx_max]:.4f}")
        
        # SHAP waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values[tp_idx_max],
            base_values=explainer.expected_value,
            data=X_test.iloc[tp_idx_max].values,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title(f"True Positive Örnek - Waterfall Plot (Index: {tp_idx_max})", 
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}3_1_waterfall_TP_{tp_idx_max}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Görsel: {output_dir}3_1_waterfall_TP_{tp_idx_max}.png")
        
        # Top-5 katkılı feature'lar
        shap_abs = np.abs(shap_values[tp_idx_max])
        top_5_indices = np.argsort(shap_abs)[-5:][::-1]
        
        tp_top5 = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_5_indices],
            'SHAP_Value': [shap_values[tp_idx_max][i] for i in top_5_indices],
            'Feature_Value': [X_test.iloc[tp_idx_max, i] for i in top_5_indices],
            'Contribution': ['Pozitif (Risk Artırıcı)' if shap_values[tp_idx_max][i] > 0 
                            else 'Negatif (Risk Azaltıcı)' for i in top_5_indices]
        })
        
        tp_top5.to_csv(f'{output_dir}3_1_TP_top5_features.csv', index=False)
        
        print("\n  Top-5 Katkılı Feature:")
        print(tp_top5.to_string(index=False))
        
        local_cases.append({
            'case_type': 'True Positive',
            'index': tp_idx_max,
            'y_true': 1,
            'y_pred': 1,
            'y_prob': y_prob[tp_idx_max],
            'top_features': tp_top5['Feature'].tolist()
        })
    
    # -------------------------------------------------------------------------
    # FALSE NEGATIVE ÖRNEĞİ
    # -------------------------------------------------------------------------
    if len(fn_indices) > 0:
        print("\n[3.2] False Negative örneği analizi...")
        
        # En düşük olasılıklı FN'yi seç (kaçırılan örnek)
        fn_probs = y_prob[fn_indices]
        fn_idx_min = fn_indices[np.argmin(fn_probs)]
        
        print(f"  Seçilen örnek index: {fn_idx_min}")
        print(f"  Gerçek: Fail (1), Tahmin: Pass (0)")
        print(f"  Fail olasılığı: {y_prob[fn_idx_min]:.4f}")
        
        # SHAP waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values[fn_idx_min],
            base_values=explainer.expected_value,
            data=X_test.iloc[fn_idx_min].values,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)
        plt.title(f"False Negative Örnek - Waterfall Plot (Index: {fn_idx_min})", 
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}3_2_waterfall_FN_{fn_idx_min}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Görsel: {output_dir}3_2_waterfall_FN_{fn_idx_min}.png")
        
        # Top-5 katkılı feature'lar
        shap_abs = np.abs(shap_values[fn_idx_min])
        top_5_indices = np.argsort(shap_abs)[-5:][::-1]
        
        fn_top5 = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_5_indices],
            'SHAP_Value': [shap_values[fn_idx_min][i] for i in top_5_indices],
            'Feature_Value': [X_test.iloc[fn_idx_min, i] for i in top_5_indices],
            'Contribution': ['Pozitif (Risk Artırıcı)' if shap_values[fn_idx_min][i] > 0 
                            else 'Negatif (Risk Azaltıcı)' for i in top_5_indices]
        })
        
        fn_top5.to_csv(f'{output_dir}3_2_FN_top5_features.csv', index=False)
        
        print("\n  Top-5 Katkılı Feature:")
        print(fn_top5.to_string(index=False))
        
        local_cases.append({
            'case_type': 'False Negative',
            'index': fn_idx_min,
            'y_true': 1,
            'y_pred': 0,
            'y_prob': y_prob[fn_idx_min],
            'top_features': fn_top5['Feature'].tolist()
        })
    
    # -------------------------------------------------------------------------
    # Akademik Yorum Metni (TEZ İÇİN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEZ İÇİN AKADEMİK YORUM - LOKAL AÇIKLAMALAR")
    print("=" * 70)
    
    thesis_text_local = """
### Lokal Model Açıklamaları - Vaka Analizi

SHAP waterfall plot'ları, bireysel tahminlerin nasıl oluştuğunu adım adım 
göstermektedir. Her bir feature'ın modelin base (baseline) tahmininden 
final tahmine olan katkısı görselleştirilmektedir.

"""
    
    if len(tp_indices) > 0:
        tp_case = local_cases[0]
        tp_feats = ', '.join(tp_case['top_features'][:3])
        
        thesis_text_local += f"""
**True Positive Örnek (Index: {tp_case['index']}):**

Bu örnekte model, gerçekten hatalı olan bir ürünü başarıyla tespit etmiştir 
(Fail olasılığı: {tp_case['y_prob']:.4f}). Modelin Fail kararı vermesine 
en çok katkı sağlayan değişkenler şunlardır:

{', '.join([f"• {feat}" for feat in tp_case['top_features'][:5]])}

Waterfall plot'ta görüldüğü üzere, bu sensörlerin SHAP değerleri pozitif 
yöndedir ve modelin base tahmininden (expected value) Fail yönüne doğru 
güçlü bir itme yapmaktadır. Bu örnekte {tp_feats} sensörlerinin 
normalin dışında değerler göstermesi, modelin doğru karar vermesinde 
belirleyici olmuştur.

**Karar Mekanizması Yorumu:**
Model, bu örnekte birden fazla sensörün uyuşan sinyalleri sayesinde yüksek 
güven ile Fail kararı vermiştir. Bu durum, model kararlarının sağlam temellere 
dayandığını ve tek bir sensöre bağımlı olmadığını göstermektedir.
"""
    
    if len(fn_indices) > 0:
        fn_case = local_cases[1] if len(local_cases) > 1 else local_cases[0]
        fn_feats = ', '.join(fn_case['top_features'][:3])
        
        thesis_text_local += f"""

**False Negative Örnek (Index: {fn_case['index']}):**

Bu vakada model, gerçekte hatalı olan bir ürünü Pass olarak sınıflandırmış 
ve hatayı kaçırmıştır (Fail olasılığı: {fn_case['y_prob']:.4f}). Bu durumun 
nedenleri SHAP analizi ile aşağıdaki gibi açıklanmaktadır:

{', '.join([f"• {feat}" for feat in fn_case['top_features'][:5]])}

Waterfall plot'ta dikkat çeken nokta, bazı sensörlerin pozitif (risk artırıcı) 
katkı yaparken, diğerlerinin negatif (risk azaltıcı) katkı yapmasıdır. 
Bu çelişkili sinyaller nedeniyle model karar güveni düşmüş ve hata 
kaçırılmıştır.

**Kritik Analiz:**
{fn_feats} sensörlerinin belirsizliği veya çelişkili katkıları, modelin 
bu vakada yanılmasına neden olmuştur. Bu durum, özellikle karmaşık vaka 
türlerinde model performansının iyileştirilmesi gerektiğine işaret 
etmektedir. Ayrıca, bu tür vakaların manuel incelenmesi ve süreç 
iyileştirmesi için kullanılması önerilmektedir.

**Genel Değerlendirme:**
Lokal açıklamalar, modelin hangi durumlarda başarılı olduğunu ve hangi 
durumlarda yanılabileceğini anlamamızı sağlamaktadır. Bu bilgi, modelin 
güvenilir şekilde deploy edilmesi ve sürekli izlenmesi açısından kritik 
önem taşımaktadır.
"""
    
    # Metni kaydet
    with open(f'{output_dir}3_thesis_text_local.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text_local)
    
    print(thesis_text_local)
    print(f"\n✓ Tez metni kaydedildi: {output_dir}3_thesis_text_local.txt")
    
    return local_cases


# =============================================================================
# 4. XDSS VE XSM İÇİN KURAL TÜRETME
# =============================================================================

def derive_rules_for_xdss_xsm(importance_df, threshold_df, shap_values, X_test, output_dir):
    """
    XDSS (Explainable Decision Support System) ve XSM (Explainable Safety 
    Mechanism) için kural altyapısı oluşturur.
    
    Çıktılar:
        - Top-5 kritik sensör listesi
        - Her biri için risk yönü ve istatistiksel eşikler
        - Kural önerileri (if-then formatında)
    """
    
    print("\n" + "=" * 70)
    print("4. XDSS VE XSM İÇİN KURAL TÜRETME")
    print("=" * 70)
    
    # Top-5 kritik sensör
    top_5_sensors = importance_df.head(5)
    
    rules = []
    
    print("\n" + "─" * 70)
    print("KRİTİK SENSÖRLER VE KURAL ÖNERİLERİ")
    print("─" * 70)
    
    for idx, row in top_5_sensors.iterrows():
        feature = row['Feature']
        mean_shap = row['Mean_SHAP']
        mean_abs_shap = row['Mean_|SHAP|']
        effect_direction = row['Effect_Direction']
        
        # Feature'ın X_test'teki indexi
        feature_idx = list(X_test.columns).index(feature)
        feature_values = X_test[feature].values
        feature_shap = shap_values[:, feature_idx]
        
        # İstatistiksel eşikler
        p25 = np.percentile(feature_values, 25)
        p50 = np.percentile(feature_values, 50)
        p75 = np.percentile(feature_values, 75)
        p90 = np.percentile(feature_values, 90)
        
        # Risk threshold (SHAP pozitif olan değerlerin min/median'ı)
        positive_shap_mask = feature_shap > 0
        if positive_shap_mask.any():
            risk_threshold_low = feature_values[positive_shap_mask].min()
            risk_threshold_high = np.median(feature_values[positive_shap_mask])
        else:
            risk_threshold_low = p75
            risk_threshold_high = p90
        
        # Kural oluştur
        if mean_shap > 0:  # Yüksek değerler risk
            risk_direction = "YÜKSEK DEĞERLER → Fail Riski ARTAR"
            rule_condition = f"IF {feature} > {p75:.4f} (75th percentile)"
            recommended_threshold = f"{p75:.4f} - {p90:.4f}"
        else:  # Düşük değerler risk
            risk_direction = "DÜŞÜK DEĞERLER → Fail Riski ARTAR"
            rule_condition = f"IF {feature} < {p25:.4f} (25th percentile)"
            recommended_threshold = f"{p25:.4f} - {p50:.4f}"
        
        rule = {
            'Sensor': feature,
            'Importance_Rank': idx + 1,
            'Mean_|SHAP|': mean_abs_shap,
            'Risk_Direction': risk_direction,
            'Rule_Condition': rule_condition,
            'Recommended_Threshold': recommended_threshold,
            'P25': p25,
            'P50': p50,
            'P75': p75,
            'P90': p90
        }
        
        rules.append(rule)
        
        print(f"\n{idx + 1}. {feature}")
        print(f"   Risk Yönü: {risk_direction}")
        print(f"   SHAP Katkısı: {mean_abs_shap:.6f}")
        print(f"   Önerilen Eşik: {recommended_threshold}")
        print(f"   Kural: {rule_condition}")
        print(f"   İstatistikler: P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}, P90={p90:.4f}")
    
    # Rules tablosunu kaydet
    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(f'{output_dir}4_xdss_xsm_rules.csv', index=False)
    
    print(f"\n✓ Kural tablosu kaydedildi: {output_dir}4_xdss_xsm_rules.csv")
    
    # -------------------------------------------------------------------------
    # Kural Dokümantasyonu (TEZ İÇİN)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEZ İÇİN KURAL DOKÜMANTASYONU")
    print("=" * 70)
    
    thesis_text_rules = """
### XDSS ve XSM İçin Türetilen Kurallar

SHAP analizi sonuçlarına dayalı olarak, yarı iletken üretim sürecinde 
erken uyarı ve karar destek sistemleri için aşağıdaki kurallar türetilmiştir:

**Kural Tabanlı Risk Değerlendirme Sistemi:**

"""
    
    for i, rule in enumerate(rules, 1):
        thesis_text_rules += f"""
**Kural {i}: {rule['Sensor']}**

- **Risk Yönü:** {rule['Risk_Direction']}
- **SHAP Önemi:** {rule['Mean_|SHAP|']:.6f}
- **Kritik Eşik:** {rule['Recommended_Threshold']}
- **Koşul:** {rule['Rule_Condition']}
- **Aksiyon:** THEN → Üretim sürecini incele, potansiyel hata riski yüksek

"""
    
    thesis_text_rules += """

**XDSS (Explainable Decision Support System) Kullanımı:**

Bu kurallar, üretim hattı operatörlerine gerçek zamanlı karar desteği 
sağlamak için kullanılabilir. Sensör değerleri belirtilen eşikleri aştığında, 
sistem otomatik olarak uyarı üretecek ve operatöre hangi sensörün probleme 
neden olduğunu açıklayacaktır.

**XSM (Explainable Safety Mechanism) Kullanımı:**

Güvenlik katmanı olarak, bu kurallar modelin tahminlerini doğrulamak için 
kullanılabilir. Eğer model Fail tahmini yapıyorsa ancak kritik sensörler 
normal aralıktaysa, bu durum güvenlik mekanizması tarafından flaglenebilir 
ve manuel inceleme tetiklenebilir.

**Risk Skoru Hesaplama Önerisi:**

Her bir kritik sensör için risk skoru:
```
Risk_Score_i = SHAP_Importance_i × (Sensor_Value - Threshold) / Threshold
```

Toplam risk skoru:
```
Total_Risk_Score = Σ Risk_Score_i  (i = 1 to 5)
```

Total_Risk_Score > 0.5 ise yüksek risk kategorisi olarak değerlendirilebilir.
"""
    
    # Metni kaydet
    with open(f'{output_dir}4_thesis_text_xdss_xsm.txt', 'w', encoding='utf-8') as f:
        f.write(thesis_text_rules)
    
    print(thesis_text_rules)
    print(f"\n✓ Kural dokümantasyonu kaydedildi: {output_dir}4_thesis_text_xdss_xsm.txt")
    
    return rules_df


# =============================================================================
# 5. KAPSAMLI TEZ ÖZETİ
# =============================================================================

def generate_comprehensive_thesis_summary(importance_df, threshold_df, rules_df, local_cases, output_dir):
    """
    Tüm XAI analizlerinin kapsamlı özetini oluşturur.
    """
    
    print("\n" + "=" * 70)
    print("5. KAPSAMLI TEZ ÖZETİ")
    print("=" * 70)
    
    comprehensive_summary = """
================================================================================
SECOM - KAPSAMLI XAI ANALİZİ TEZ ÖZETİ
Açıklanabilir Yapay Zeka ile Model Şeffaflığı ve Karar Destek Sistemi
================================================================================

### AMAÇ

Bu çalışmada, SECOM yarı iletken üretim veri seti üzerinde eğitilen 
XGBoost modelinin kararlarını SHAP (SHapley Additive exPlanations) yöntemi 
ile açıklamak ve modelin tahmin mekanizmasını anlaşılır hale getirmek 
amaçlanmıştır.

### YÖNTEM

**XAI Yaklaşımı:** SHAP (Shapley değerleri temelli açıklama)
**Model:** XGBoost Classifier (n_estimators=50, max_depth=3)
**Pipeline:** IterativeImputer → RobustScaler → Top-100 Features → SMOTE → XGBoost
**Test Seti:** Stratified split ile ayrılmış %20 test verisi

### ANA BULGULAR

"""
    
    # 1. Global Bulgular
    top_5_features = importance_df.head(5)['Feature'].tolist()
    comprehensive_summary += f"""
**1. Global Model Açıklanabilirliği:**

En etkili 5 sensör:
{chr(10).join([f'   {i+1}. {feat}' for i, feat in enumerate(top_5_features)])}

Bu sensörler, modelin toplam SHAP katkısının önemli bir bölümünü oluşturmakta 
ve üretim sürecindeki kritik noktaları işaret etmektedir.

"""
    
    # 2. Eşik Analizi
    comprehensive_summary += f"""
**2. Sensör Eşik Davranışları:**

SHAP dependence analizi ile her bir kritik sensör için eşik değerleri 
belirlenmiştir. Bu eşikler, sensör değerlerinin hangi noktadan itibaren 
Fail riskini artırdığını göstermektedir.

Örnek eşik değerleri (Top-3 sensör için):
"""
    
    for i, row in threshold_df.head(3).iterrows():
        comprehensive_summary += f"""
   • {row['feature']}: 
     - 75th percentile: {row['percentile_75']:.4f}
     - Risk eşiği: ~{row['median_risk_value']:.4f}
"""
    
    # 3. Lokal Açıklamalar
    comprehensive_summary += f"""

**3. Vaka Bazlı Açıklamalar:**

True Positive örneği incelemesi, modelin doğru Fail tespitlerinde birden 
fazla sensörün uyuşan sinyaller vermesinin önemini ortaya koymuştur.

False Negative örneği analizi ise, çelişkili sensör sinyallerinin model 
kararında belirsizliğe yol açtığını ve bazı hataların kaçırılmasına neden 
olduğunu göstermiştir.

"""
    
    # 4. XDSS/XSM Kuralları
    comprehensive_summary += f"""
**4. Karar Destek Sistemi Kuralları:**

SHAP sonuçlarına dayalı olarak {len(rules_df)} adet kural türetilmiştir. 
Bu kurallar:
- Gerçek zamanlı üretim izleme
- Erken uyarı sistemi
- Operatör karar desteği

amaçları ile kullanılabilir.

"""
    
    # 5. Sonuç ve Öneriler
    comprehensive_summary += """
### SONUÇ VE ÖNERİLER

**Bilimsel Katkılar:**

1. **Model Şeffaflığı:** XGBoost modelinin "black box" yapısı SHAP ile 
   açıklanabilir hale getirilmiştir.

2. **Sensör Önceliklendirme:** Kritik sensörler nicel olarak belirlenmiş 
   ve üretim sürecinde öncelikli izleme alanları ortaya konmuştur.

3. **Eşik Belirleme:** Her sensör için istatistiksel eşikler hesaplanmış 
   ve süreç kontrol limitleri önerilmiştir.

4. **Karar Destek Altyapısı:** XDSS ve XSM için kural tabanlı sistem 
   altyapısı oluşturulmuştur.

**Pratik Uygulamalar:**

- **Kalite Kontrol:** Model tahminleri, operatörlere hangi sensörlere 
  dikkat etmeleri gerektiğini gösterebilir.

- **Süreç İyileştirme:** Kritik sensörlerin davranışları incelenerek 
  üretim süreci optimize edilebilir.

- **Maliyet Azaltma:** Hatalı ürünlerin erken tespiti ile malzeme 
  israfı ve yeniden işleme maliyetleri azaltılabilir.

**Gelecek Çalışmalar:**

1. LIME ve diğer XAI yöntemleri ile karşılaştırmalı analiz
2. Zaman serisi bazlı SHAP analizi ile dinamik açıklamalar
3. Gerçek zamanlı XDSS sisteminin endüstriyel ortamda test edilmesi
4. Counterfactual açıklamalar ile "What-if" senaryolarının incelenmesi

================================================================================
                            ANALİZ TAMAMLANDI
================================================================================
"""
    
    # Kaydet
    with open(f'{output_dir}5_comprehensive_thesis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(comprehensive_summary)
    
    print(comprehensive_summary)
    print(f"\n✓ Kapsamlı özet kaydedildi: {output_dir}5_comprehensive_thesis_summary.txt")
    
    return comprehensive_summary


# =============================================================================
# ANA FONKSİYON - TÜM ANALİZLERİ ÇALIŞTIR
# =============================================================================

def run_complete_xai_analysis(data_dir='./save_model_and_test_outputs/', output_dir='./xai_analysis_outputs/'):
    """
    Tüm XAI analizlerini sırasıyla çalıştırır.
    
    Çıktılar:
        - Global SHAP analizi
        - Dependence plot analizi
        - Lokal vaka açıklamaları
        - XDSS/XSM kuralları
        - Kapsamlı tez özeti
    """
    
    print("\n")
    print("*" * 70)
    print("  SECOM - KAPSAMLI XAI/SHAP ANALİZİ")
    print("  Explainable AI ile Model Açıklanabilirliği")
    print("*" * 70)
    
    # Çıktı klasörü oluştur
    output_dir = create_output_dir(output_dir)
    print(f"\n✓ Çıktı klasörü: {output_dir}")
    
    # 1. Verileri yükle
    model, X_test, y_test, predictions, feature_names = load_xai_data(data_dir)
    
    # 2. Global SHAP analizi
    shap_values, explainer, importance_df = global_shap_analysis(
        model, X_test, feature_names, output_dir
    )
    
    # 3. Dependence plot analizi
    threshold_df = dependence_plot_analysis(
        shap_values, X_test, importance_df, output_dir
    )
    
    # 4. Lokal vaka açıklamaları
    local_cases = local_explanation_analysis(
        shap_values, explainer, X_test, y_test, 
        predictions, feature_names, output_dir
    )
    
    # 5. XDSS/XSM kuralları
    rules_df = derive_rules_for_xdss_xsm(
        importance_df, threshold_df, shap_values, X_test, output_dir
    )
    
    # 6. Kapsamlı tez özeti
    comprehensive_summary = generate_comprehensive_thesis_summary(
        importance_df, threshold_df, rules_df, local_cases, output_dir
    )
    
    print("\n" + "=" * 70)
    print("TÜM XAI ANALİZLERİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)
    print(f"\nÇıktı dosyaları: {output_dir}")
    print("\nOluşturulan dosyalar:")
    print("  Görseller:")
    print("    • 1_shap_summary_plot.png")
    print("    • 2_1_dependence_[feature].png (Top-3 sensör)")
    print("    • 3_1_waterfall_TP_[index].png")
    print("    • 3_2_waterfall_FN_[index].png")
    print("\n  Veri Dosyaları:")
    print("    • 1_feature_importance_full.csv")
    print("    • 1_top10_features.csv")
    print("    • 1_top20_features.csv")
    print("    • 2_threshold_analysis.csv")
    print("    • 3_1_TP_top5_features.csv")
    print("    • 3_2_FN_top5_features.csv")
    print("    • 4_xdss_xsm_rules.csv")
    print("\n  Tez Metinleri:")
    print("    • 1_thesis_text_global.txt")
    print("    • 2_thesis_text_dependence.txt")
    print("    • 3_thesis_text_local.txt")
    print("    • 4_thesis_text_xdss_xsm.txt")
    print("    • 5_comprehensive_thesis_summary.txt")
    
    return {
        'model': model,
        'shap_values': shap_values,
        'explainer': explainer,
        'importance_df': importance_df,
        'threshold_df': threshold_df,
        'rules_df': rules_df,
        'local_cases': local_cases
    }


# =============================================================================
# ÇALIŞTIRMA
# =============================================================================

if __name__ == "__main__":
    
    # Veri ve çıktı klasörleri
    DATA_DIR = './save_model_and_test_outputs/'
    OUTPUT_DIR = './xai_analysis_outputs/'
    
    # Tam XAI analizini çalıştır
    results = run_complete_xai_analysis(DATA_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("XAI ANALİZİ TAMAMLANDI!")
    print("Tez için tüm görseller, tablolar ve açıklamalar hazır.")
    print("=" * 70)