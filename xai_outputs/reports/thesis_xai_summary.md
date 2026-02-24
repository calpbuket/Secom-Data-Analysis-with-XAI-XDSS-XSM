## XAI Analizi Bulguları
### SHAP Global Analizi
SECOM veri setinde 2 farklı model (XGBoost, LightGBM, RandomForest) üzerinde SHAP (SHapley Additive exPlanations) analizi uygulanmıştır. Analiz sonucunda her model için en kritik sensörler aşağıdaki gibi tespit edilmiştir:

**XGBoost** — Top 5 Kritik Sensör:
1. `59` (Mean |SHAP| = 0.840663)
2. `419` (Mean |SHAP| = 0.365080)
3. `130` (Mean |SHAP| = 0.335090)
4. `460` (Mean |SHAP| = 0.305468)
5. `486` (Mean |SHAP| = 0.275342)

**LightGBM** — Top 5 Kritik Sensör:
1. `59` (Mean |SHAP| = 1.261042)
2. `486` (Mean |SHAP| = 0.758375)
3. `419` (Mean |SHAP| = 0.705156)
4. `130` (Mean |SHAP| = 0.593130)
5. `377` (Mean |SHAP| = 0.513707)

### Tüm Modellerde Ortak Kritik Sensörler
Aşağıdaki sensörler birden fazla modelin Top-20 listesinde yer almaktadır. Bu sensörler üretim sürecinde öncelikli izleme listesine alınmalıdır:

- `59` → 2/2 modelde kritik
- `419` → 2/2 modelde kritik
- `130` → 2/2 modelde kritik
- `460` → 2/2 modelde kritik
- `486` → 2/2 modelde kritik
- `377` → 2/2 modelde kritik
- `16` → 2/2 modelde kritik
- `433` → 2/2 modelde kritik
- `90` → 2/2 modelde kritik
- `33` → 2/2 modelde kritik

### SHAP Local (Tekil Örnek) Analizi
Gerçek fail örnekleri üzerinde yerel SHAP analizi yapılarak, her hatalı wafer için spesifik sensör katkıları belirlenmiştir. Bu analiz, üretim mühendisinin 'Bu wafer neden başarısız oldu?' sorusunu veri odaklı yanıtlamasına olanak tanır.

### LIME Analizi
LIME (Local Interpretable Model-agnostic Explanations) yöntemi, model tipinden bağımsız olarak tekil örnekleri doğrusal bir yaklaşımla açıklar. SHAP analizi ile tutarlı sensörlerin LIME'da da öne çıkması, bulguların güvenilirliğini doğrulamaktadır.
