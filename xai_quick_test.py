"""
XAI ANALİZİ - HIZLI TEST VE DEMO
=================================

Bu script, XAI analizinin doğru çalıştığını test etmek için 
basitleştirilmiş bir demo sunar.

Küçük bir veri alt kümesi ile hızlıca test yapabilirsiniz.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def quick_test():
    """
    XAI analiz sistemininin çalışıp çalışmadığını hızlıca test eder.
    """
    
    print("\n" + "=" * 70)
    print("XAI ANALİZİ - HIZLI TEST")
    print("=" * 70)
    
    # 1. Gerekli klasörlerin varlığını kontrol et
    print("\n[1] Klasör kontrolü...")
    
    xai_inputs_exists = Path('./xai_inputs/').exists()
    xai_outputs_exists = Path('./xai_outputs/').exists()
    
    print(f"  xai_inputs/: {'✓ Mevcut' if xai_inputs_exists else '✗ Yok (oluşturulacak)'}")
    print(f"  xai_outputs/: {'✓ Mevcut' if xai_outputs_exists else '✗ Yok (oluşturulacak)'}")
    
    if not xai_inputs_exists:
        print("\n  ⚠️  UYARI: xai_inputs/ klasörü bulunamadı!")
        print("  Önce 'save_model_and_test.py' scriptini çalıştırmalısınız.")
        return False
    
    # 2. Gerekli dosyaların varlığını kontrol et
    print("\n[2] Dosya kontrolü...")
    
    required_files = [
        'xai_final_model.pkl',
        'xai_test_X.csv',
        'xai_test_y.csv',
        'xai_test_predictions.csv',
        'xai_top100_features.txt'
    ]
    
    all_files_exist = True
    for file in required_files:
        file_path = Path(f'./xai_inputs/{file}')
        exists = file_path.exists()
        print(f"  {file}: {'✓' if exists else '✗'}")
        if not exists:
            all_files_exist = False
    
    if not all_files_exist:
        print("\n  ✗ HATA: Bazı dosyalar eksik!")
        print("  Önce 'save_model_and_test.py' scriptini çalıştırın.")
        return False
    
    # 3. Python paketlerini kontrol et
    print("\n[3] Paket kontrolü...")
    
    packages = {
        'shap': None,
        'sklearn': None,
        'xgboost': None,
        'pandas': None,
        'numpy': None,
        'matplotlib': None
    }
    
    for package_name in packages.keys():
        try:
            if package_name == 'sklearn':
                import sklearn
                packages[package_name] = sklearn.__version__
            elif package_name == 'shap':
                import shap
                packages[package_name] = shap.__version__
            elif package_name == 'xgboost':
                import xgboost
                packages[package_name] = xgboost.__version__
            elif package_name == 'pandas':
                packages[package_name] = pd.__version__
            elif package_name == 'numpy':
                packages[package_name] = np.__version__
            elif package_name == 'matplotlib':
                packages[package_name] = plt.matplotlib.__version__
        except ImportError:
            packages[package_name] = None
    
    all_packages_ok = True
    for package_name, version in packages.items():
        if version:
            print(f"  {package_name}: ✓ v{version}")
        else:
            print(f"  {package_name}: ✗ YÜKLENMEMİŞ")
            all_packages_ok = False
    
    if not all_packages_ok:
        print("\n  ✗ HATA: Bazı paketler eksik!")
        print("  pip install shap scikit-learn xgboost pandas numpy matplotlib --break-system-packages")
        return False
    
    # 4. Basit veri yükleme testi
    print("\n[4] Veri yükleme testi...")
    
    try:
        import pickle
        
        # Model yükle
        with open('./xai_inputs/xai_final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"  Model: ✓ {type(model).__name__}")
        
        # Test verisi yükle
        X_test = pd.read_csv('./xai_inputs/xai_test_X.csv')
        y_test = pd.read_csv('./xai_inputs/xai_test_y.csv')
        
        print(f"  X_test: ✓ {X_test.shape}")
        print(f"  y_test: ✓ {len(y_test)}")
        
        # Feature isimleri
        with open('./xai_inputs/xai_top100_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"  Features: ✓ {len(feature_names)} adet")
        
    except Exception as e:
        print(f"  ✗ HATA: {str(e)}")
        return False
    
    # 5. SHAP basit test
    print("\n[5] SHAP hesaplama testi (küçük örnek)...")
    
    try:
        import shap
        
        # Sadece 10 örnek ile test
        X_sample = X_test.head(10)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        print(f"  SHAP values: ✓ {shap_values.shape}")
        print(f"  Mean |SHAP|: {np.abs(shap_values).mean():.6f}")
        
    except Exception as e:
        print(f"  ✗ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Başarılı
    print("\n" + "=" * 70)
    print("✓ TÜM TESTLER BAŞARILI!")
    print("=" * 70)
    print("\nXAI analiz sisteminiz çalışmaya hazır.")
    print("Şimdi 'python xai_analysis.py' komutunu çalıştırabilirsiniz.")
    
    return True


def print_quick_start():
    """Hızlı başlangıç kılavuzu."""
    
    print("""

================================================================================
HIZLI BAŞLANGIÇ
================================================================================

XAI analizini çalıştırmak için:

1. Terminal açın
2. Aşağıdaki komutu çalıştırın:

   python xai_analysis.py

3. İşlem bittiğinde, xai_outputs/ klasöründe tüm çıktılarınız hazır olacak.

================================================================================
ÇIKTILAR
================================================================================

• Görseller: 1_shap_summary_plot.png, 2_*_dependence_*.png, 3_*_waterfall_*.png
• Tablolar: *_features.csv, *_analysis.csv, *_rules.csv
• Tez Metinleri: *_thesis_text_*.txt

Tüm dosyalar tezinizde doğrudan kullanıma hazır olacaktır.

================================================================================
""")


if __name__ == "__main__":
    
    # Test çalıştır
    success = quick_test()
    
    if success:
        # Hızlı başlangıç kılavuzu göster
        print_quick_start()
    else:
        print("\n" + "=" * 70)
        print("✗ TEST BAŞARISIZ")
        print("=" * 70)
        print("\nLütfen yukarıdaki hataları düzeltin ve tekrar deneyin.")
        print("Yardıma ihtiyacınız varsa, XAI_KULLANIM_KILAVUZU.txt dosyasını okuyun.")