# SECOM Veri Seti - Sınıf Dengesizliği Stratejileri Karşılaştırması

## AŞAMA 1 Özet Raporu

### Deney Konfigürasyonu
- **Sabit Imputer:** IterativeImputer (MICE)
- **Sabit Scaler:** RobustScaler  
- **Sabit Model:** XGBClassifier
- **Cross-Validation:** 5-Fold Stratified
- **Veri:** 1567 örnek, 442 feature (temizleme sonrası)
- **Sınıf Dağılımı:** Pass %93.36, Fail %6.64 (14:1 oran)

---

## Sonuç Tablosu

| Strateji | F1_macro | F1_fail | Recall_fail | Precision_fail | ROC-AUC | PR-AUC | Süre (s) |
|----------|----------|---------|-------------|----------------|---------|--------|----------|
| Baseline (spw=14.07) | 0.4907 | 0.0182 | 0.0095 | 0.2000 | 0.7102 | 0.1695 | 4447 |
| **SMOTE** | **0.5463** | 0.1285 | 0.0771 | **0.4705** | 0.7054 | **0.1954** | 4910 |
| ADASYN | 0.5204 | 0.0783 | 0.0486 | 0.2238 | **0.7154** | 0.1664 | 4403 |
| Random Undersampling | 0.4811 | **0.1812** | **0.5767** | 0.1076 | 0.6631 | 0.1510 | 4030 |
| SMOTE + Tomek | 0.5292 | 0.0956 | 0.0581 | 0.3133 | 0.6959 | 0.1701 | 3927 |

---

## Baseline'a Göre Değişimler

| Strateji | ΔF1_fail | ΔRecall_fail | ΔPR-AUC | ΔROC-AUC |
|----------|----------|--------------|---------|----------|
| SMOTE | +0.1103 | +0.0676 | +0.0259 | -0.0048 |
| ADASYN | +0.0601 | +0.0390 | -0.0030 | +0.0052 |
| Random Undersampling | +0.1630 | +0.5671 | -0.0184 | -0.0471 |
| SMOTE + Tomek | +0.0774 | +0.0486 | +0.0006 | -0.0142 |

---

## Confusion Matrix Özeti (5 Fold Toplamı)

| Strateji | TN | FP | FN | TP | TPR | FPR |
|----------|----|----|----|----|-----|-----|
| Baseline | 1455 | 8 | 103 | 1 | 0.010 | 0.005 |
| SMOTE | 1451 | 12 | 96 | 8 | 0.077 | 0.008 |
| ADASYN | 1449 | 14 | 99 | 5 | 0.048 | 0.010 |
| **Undersampling** | 966 | 497 | 44 | **60** | **0.577** | 0.340 |
| SMOTE + Tomek | 1449 | 14 | 98 | 6 | 0.058 | 0.010 |

---

## Tez için Yorum Taslağı

### 4.X.X Sınıf Dengesizliği Stratejilerinin Değerlendirilmesi

SECOM veri setindeki %6.64'lük Fail sınıfı oranı, ciddi bir sınıf dengesizliği problemi oluşturmaktadır. Bu dengesizliğin sınıflandırma performansına etkisini değerlendirmek amacıyla beş farklı strateji karşılaştırılmıştır.

#### Baseline (scale_pos_weight)
XGBoost'un yerleşik `scale_pos_weight` parametresi (14.07) kullanıldığında, model neredeyse tüm örnekleri Pass olarak sınıflandırmış ve yalnızca **1 adet** Fail örneği doğru tespit edebilmiştir (Recall: %0.95). Bu sonuç, salt maliyet ağırlıklandırmasının bu denli dengesiz veri setlerinde yetersiz kaldığını göstermektedir.

#### SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE, sentetik Fail örnekleri üreterek eğitim setini dengelemiştir. Sonuçlar incelendiğinde:
- F1_fail: 0.0182 → **0.1285** (+607% artış)
- Recall_fail: 0.0095 → **0.0771** (+712% artış)
- PR-AUC: 0.1695 → **0.1954** (+15% artış)
- Precision: 0.2000 → **0.4705** (+135% artış)

SMOTE, hem Recall hem de Precision'da dengeli iyileşme sağlamış, ROC-AUC kaybı minimal düzeyde kalmıştır (-0.0048).

#### ADASYN
ADASYN, SMOTE'a benzer performans göstermiş ancak tüm metriklerde SMOTE'un gerisinde kalmıştır. ROC-AUC'da marjinal iyileşme (+0.0052) sağlamasına rağmen, F1_fail ve PR-AUC değerleri SMOTE'tan düşüktür.

#### Random Undersampling
En yüksek Recall değerini (%57.67) elde etmiş, 104 Fail örneğinden **60'ını** doğru tespit etmiştir. Ancak bu başarı, **497 False Positive** (yanlış alarm) pahasına gelmiştir. Üretim ortamında bu denli yüksek FP oranı (%34) kabul edilemez düzeyde operasyonel maliyet yaratacaktır.

#### SMOTE + Tomek Links
Hibrit yaklaşım, SMOTE'tan düşük performans göstermiştir. Tomek Links'in sınır örneklerini temizlemesi, zaten küçük olan Fail sınıfından kritik bilgi kaybına yol açmış olabilir.

---

### Karar Matrisi

| Kriter | En İyi Strateji | Değer | Yorum |
|--------|-----------------|-------|-------|
| F1_macro | SMOTE | 0.5463 | Genel denge |
| F1_fail | Undersampling | 0.1812 | Ama çok FP |
| Recall_fail | Undersampling | 0.5767 | Hatalı ürün tespiti |
| Precision_fail | SMOTE | 0.4705 | Yanlış alarm minimizasyonu |
| ROC-AUC | ADASYN | 0.7154 | Genel ayırt edicilik |
| PR-AUC | SMOTE | 0.1954 | Dengesiz veri performansı |

### Final Karar

**Akademik/Tez için önerilen strateji: SMOTE**

**Gerekçe:**
1. PR-AUC'da en yüksek skor (dengesiz veri için kritik metrik)
2. Precision ve Recall arasında dengeli iyileşme
3. ROC-AUC kaybı minimal
4. False Positive oranı kabul edilebilir düzeyde (%0.8)

**Alternatif Senaryo:**
Eğer üretim hattında "hiçbir hatalı ürün kaçırılmamalı" prensibi geçerliyse ve yanlış alarmların maliyeti düşükse, **Random Undersampling** tercih edilebilir. Ancak bu durumda her 3 alarmdan 1'inin gerçek hata olacağı göz önünde bulundurulmalıdır.

---

## Notlar

- Tüm deneyler aynı random_state (42) ile tekrarlanabilir şekilde yapılmıştır
- IterativeImputer hesaplama süresi nedeniyle toplam deney ~6 saat sürmüştür
- Resampling yalnızca eğitim seti üzerinde uygulanmıştır (data leakage önlemi)
