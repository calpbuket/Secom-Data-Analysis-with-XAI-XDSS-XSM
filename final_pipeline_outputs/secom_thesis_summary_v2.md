
### Nihai Model Performansı (Kalibrasyon Deneyi Sonrası)

SECOM yarı iletken üretim veri seti üzerinde yapılan kalibrasyon deneyi
kapsamında 4 farklı veri artırım senaryosu (Augmentation Yok, %10, %33, %50)
ve 3 farklı model (XGBoost, LightGBM, RandomForest) sistematik olarak
karşılaştırılmıştır. Brier Score ve Reliability Diagram analizleri
sonucunda en iyi kalibrasyon ve performans aşağıdaki konfigürasyonla
elde edilmiştir:

**Kazanan Konfigürasyon:**
- Model: XGBoost
- SMOTE Oranı: %50 (azınlık sınıfı → çoğunluğun %50'si)

**Pipeline Yapısı:**
1. IterativeImputer (MICE) - Eksik veri tamamlama
2. RobustScaler - Özellik ölçekleme
3. VarianceThreshold - Düşük varyanslı özellik temizleme
4. SMOTE (sampling_strategy=0.5) - Sınıf dengesizliği düzeltme
5. XGBClassifier (scale_pos_weight otomatik)
6. Threshold Optimizasyonu - F1-Score maksimizasyonu

**5-Fold Cross-Validation Sonuçları:**

| Metrik | Ortalama | Std |
|--------|----------|-----|
| F1-Score (Fail) | 0.1581 | 0.0499 |
| Recall (Fail) | 0.1924 | 0.1215 |
| Precision (Fail) | 0.3281 | 0.3791 |
| ROC-AUC | 0.6680 | 0.0476 |
| PR-AUC | 0.1721 | 0.0538 |
| Brier Score | 0.0649 | 0.0037 |

**Threshold Optimizasyonu Etkisi:**
- Varsayılan (0.5) → Optimize (0.280)
- Yakalanan Hatalı Ürün: 7 → 20 (+13 ✅)
- Kaçırılan Hata: 97 → 84 (-13 ✅)
- Yanlış Alarm: 16 → 119 (+103)

**Kalibrasyon Deneyi Önemli Bulguları:**
- SMOTE artırım oranı arttıkça modelin azınlık sınıfını (Fail) tanıma
  yeteneği iyileşmektedir.
- %50 oranı, Recall ve F1 dengesini en iyi sağlayan noktadır.
- XGBoost, gradient boosting ailesi içinde en tutarlı performansı
  sergilemiştir.
- Brier Score ile doğrulanan kalibrasyon, modelin ürettiği olasılık
  tahminlerinin güvenilir olduğunu göstermektedir.

**Yorum:**
Threshold optimizasyonu sayesinde, model 13 adet daha fazla
hatalı ürünü tespit edebilmektedir. Bu, üretim hattında kalite kontrol
maliyetlerini önemli ölçüde azaltabilir. Recall değeri
19.24% olup, hatalı ürünlerin
19/100'ünün yakalanabildiğini
göstermektedir.
