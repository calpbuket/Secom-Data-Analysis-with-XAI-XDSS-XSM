4.X.X Eksik Veri Stratejilerinin Değerlendirilmesi
SECOM veri setinde eksik değerler, toplam veri matrisinin %4.54'ünü oluşturmakta olup, 590 sensör sütununun 538'inde en az bir eksik gözlem bulunmaktadır. Eksik veri mekanizması incelendiğinde, sensör arızaları ve ölçüm kesintilerinin rastgele dağılım göstermesi nedeniyle Missing at Random (MAR) veya karma bir mekanizma olduğu değerlendirilmiştir.
Çalışmada üç farklı eksik veri tamamlama (imputation) stratejisi karşılaştırılmıştır:

Basit Medyan İmputasyonu: Her sütunun medyan değeri ile eksik gözlemlerin doldurulması
K-En Yakın Komşu (KNN) İmputasyonu: Öklid uzaklığına dayalı k=5 komşu kullanılarak çok değişkenli tamamlama
Çoklu İmputasyon (MICE/IterativeImputer): ExtraTrees regresör tabanlı iteratif tamamlama

Sonuçlar incelendiğinde, IterativeImputer yönteminin hem genel sınıflandırma performansı (F1_macro: 0.580) hem de azınlık sınıfı tespiti (F1_fail: 0.204, PR-AUC: 0.189) açısından en yüksek skorları elde ettiği görülmektedir. Bu yöntem, değişkenler arası korelasyonları modelleyerek eksik değerleri tahmin etmekte ve böylece veri yapısını daha iyi korumaktadır.
KNN İmputasyonu, IterativeImputer ile karşılaştırılabilir F1 skorları (0.580) sunmasına rağmen, ROC-AUC (0.687 vs 0.711) ve PR-AUC (0.162 vs 0.189) metriklerinde geride kalmıştır. Ancak hesaplama süresi açısından önemli bir avantaj sunmaktadır (6.3s vs 479.5s).
Medyan İmputasyonu en hızlı yöntem olmasına karşın, değişkenler arası ilişkileri göz ardı etmesi nedeniyle en düşük performansı sergilemiştir.
Karar
Kullanım SenaryosuÖnerilen YöntemGerekçeAkademik çalışma / Ana modelIterativeImputer (MICE)En yüksek PR-AUC ve ROC-AUC; Fail sınıfı tespitinde üstün performansOperasyonel / Gerçek zamanlıKNNImputer76x daha hızlı; kabul edilebilir performans kaybıHızlı prototiplemeSimpleImputer (median)Baseline olarak kullanılabilir
Bu çalışmanın devamında, IterativeImputer ana imputation yöntemi olarak sabitlenmiş ve sınıf dengesizliği stratejileri bu temel üzerinde değerlendirilmiştir.

Notlar

Veri hazırlama sonrası: 442 feature (590 → 558 → 442)
%40+ eksik olan 32 sütun silindi
Sabit değerli 116 sütun silindi
Sınıf dağılımı: Pass %93.36, Fail %6.64 (oran: 14:1)