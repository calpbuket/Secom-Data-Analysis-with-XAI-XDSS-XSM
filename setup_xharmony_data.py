"""
================================================================================
X-HARMONY - VERİ HAZIRLIK VE DOĞRULAMA SCRİPTİ
================================================================================

Bu script, X-HARMONY simülasyonu için gerekli tüm dosyaların varlığını kontrol eder
ve eksik dosyalar için rehberlik sağlar.

Kullanım:
    python setup_xharmony_data.py --source_dir <kaynak> --target_dir <hedef>
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse
import shutil


class XHARMONYDataSetup:
    """X-HARMONY için veri hazırlık ve doğrulama sınıfı."""
    
    REQUIRED_FILES = {
        'model': 'xai_final_model.pkl',
        'test_X': 'xai_test_X.csv',
        'test_y': 'xai_test_y.csv',
        'shap': 'global_shap_values.npy',
        'rules': 'xdss_xsm_rules.csv',
        'features': 'xai_feature_names.txt'
    }
    
    def __init__(self, source_dir: str, target_dir: str, verbose: bool = True):
        """
        Setup sınıfını başlat.
        
        Args:
            source_dir: Kaynak dosyaların bulunduğu klasör
            target_dir: Hedef klasör (organize edilmiş dosyalar)
            verbose: Detaylı çıktı
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.verbose = verbose
        
        # Hedef klasörü oluştur
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("X-HARMONY VERİ HAZIRLIK")
            print("=" * 70)
            print(f"Kaynak klasör: {self.source_dir}")
            print(f"Hedef klasör: {self.target_dir}")
            print("=" * 70)
    
    def check_file_exists(self, filename: str) -> bool:
        """Dosya varlığını kontrol et."""
        return (self.source_dir / filename).exists()
    
    def validate_model_file(self, filepath: Path) -> dict:
        """Model dosyasını doğrula."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            # XGBoost mu kontrol et
            model_type = type(model).__name__
            
            validation = {
                'status': 'OK',
                'model_type': model_type,
                'n_features': getattr(model, 'n_features_in_', 'Unknown')
            }
            
            return validation
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def validate_csv_file(self, filepath: Path, expected_cols: list = None) -> dict:
        """CSV dosyasını doğrula."""
        try:
            df = pd.read_csv(filepath)
            
            validation = {
                'status': 'OK',
                'shape': df.shape,
                'columns': list(df.columns)
            }
            
            if expected_cols:
                missing_cols = set(expected_cols) - set(df.columns)
                if missing_cols:
                    validation['status'] = 'WARNING'
                    validation['missing_columns'] = list(missing_cols)
            
            return validation
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def validate_npy_file(self, filepath: Path) -> dict:
        """NumPy array dosyasını doğrula."""
        try:
            arr = np.load(filepath)
            
            validation = {
                'status': 'OK',
                'shape': arr.shape,
                'dtype': str(arr.dtype)
            }
            
            return validation
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def validate_txt_file(self, filepath: Path) -> dict:
        """Text dosyasını doğrula."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            validation = {
                'status': 'OK',
                'n_lines': len(lines),
                'sample': lines[:3] if len(lines) >= 3 else lines
            }
            
            return validation
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def validate_all_files(self) -> dict:
        """Tüm gerekli dosyaları kontrol et ve doğrula."""
        
        if self.verbose:
            print("\n[1/3] DOSYA VARLIĞI KONTROLÜ")
            print("-" * 70)
        
        results = {}
        
        for file_key, filename in self.REQUIRED_FILES.items():
            filepath = self.source_dir / filename
            exists = filepath.exists()
            
            if self.verbose:
                status = "✓" if exists else "✗"
                print(f"  {status} {filename}: {'BULUNDU' if exists else 'BULUNAMADI'}")
            
            results[file_key] = {
                'filename': filename,
                'exists': exists,
                'filepath': str(filepath)
            }
        
        # Eksik dosyalar varsa uyarı ver
        missing_files = [v['filename'] for v in results.values() if not v['exists']]
        
        if missing_files:
            if self.verbose:
                print(f"\n⚠️  EKSİK DOSYALAR: {len(missing_files)}")
                for f in missing_files:
                    print(f"    - {f}")
                print("\nLütfen eksik dosyaları kaynak klasöre ekleyin.")
            return results
        
        # Dosya doğrulama
        if self.verbose:
            print("\n[2/3] DOSYA DOĞRULAMA")
            print("-" * 70)
        
        # Model
        if results['model']['exists']:
            val = self.validate_model_file(Path(results['model']['filepath']))
            results['model']['validation'] = val
            if self.verbose:
                print(f"  Model: {val.get('model_type', 'Unknown')}, "
                      f"Features: {val.get('n_features', 'Unknown')}")
        
        # Test X
        if results['test_X']['exists']:
            val = self.validate_csv_file(Path(results['test_X']['filepath']))
            results['test_X']['validation'] = val
            if self.verbose:
                print(f"  Test X: {val['shape']}")
        
        # Test y
        if results['test_y']['exists']:
            val = self.validate_csv_file(
                Path(results['test_y']['filepath']),
                expected_cols=['Pass/Fail']
            )
            results['test_y']['validation'] = val
            if self.verbose:
                print(f"  Test y: {val['shape']}")
        
        # SHAP
        if results['shap']['exists']:
            val = self.validate_npy_file(Path(results['shap']['filepath']))
            results['shap']['validation'] = val
            if self.verbose:
                print(f"  SHAP values: {val['shape']}")
        
        # Rules
        if results['rules']['exists']:
            val = self.validate_csv_file(
                Path(results['rules']['filepath']),
                expected_cols=['Sensor', 'Risk_Direction', 'Recommended_Threshold']
            )
            results['rules']['validation'] = val
            if self.verbose:
                print(f"  Rules: {val['shape']}")
        
        # Features
        if results['features']['exists']:
            val = self.validate_txt_file(Path(results['features']['filepath']))
            results['features']['validation'] = val
            if self.verbose:
                print(f"  Features: {val['n_lines']} features")
        
        return results
    
    def copy_files_to_target(self, validation_results: dict):
        """Doğrulanmış dosyaları hedef klasöre kopyala."""
        
        if self.verbose:
            print("\n[3/3] DOSYA KOPYALAMA")
            print("-" * 70)
        
        for file_key, info in validation_results.items():
            if info['exists']:
                source = Path(info['filepath'])
                target = self.target_dir / info['filename']
                
                try:
                    shutil.copy2(source, target)
                    if self.verbose:
                        print(f"  ✓ {info['filename']} kopyalandı")
                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ {info['filename']} kopyalanamadı: {e}")
    
    def generate_compatibility_report(self, validation_results: dict):
        """Uyumluluk raporu oluştur."""
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("UYUMLULUK RAPORU")
            print("=" * 70)
        
        # Shape uyumu kontrolü
        test_X_shape = None
        test_y_shape = None
        shap_shape = None
        n_features = None
        
        if 'test_X' in validation_results and validation_results['test_X']['exists']:
            test_X_shape = validation_results['test_X']['validation']['shape']
        
        if 'test_y' in validation_results and validation_results['test_y']['exists']:
            test_y_shape = validation_results['test_y']['validation']['shape']
        
        if 'shap' in validation_results and validation_results['shap']['exists']:
            shap_shape = validation_results['shap']['validation']['shape']
        
        if 'model' in validation_results and validation_results['model']['exists']:
            n_features = validation_results['model']['validation'].get('n_features', None)
        
        if 'features' in validation_results and validation_results['features']['exists']:
            n_feature_names = validation_results['features']['validation']['n_lines']
        else:
            n_feature_names = None
        
        # Uyumluluk kontrolleri
        compatible = True
        issues = []
        
        if test_X_shape and test_y_shape:
            if test_X_shape[0] != test_y_shape[0]:
                compatible = False
                issues.append(f"Test X ve Test y örnek sayıları uyumsuz: "
                            f"{test_X_shape[0]} vs {test_y_shape[0]}")
        
        if test_X_shape and shap_shape:
            if test_X_shape != shap_shape:
                compatible = False
                issues.append(f"Test X ve SHAP shape'leri uyumsuz: "
                            f"{test_X_shape} vs {shap_shape}")
        
        if test_X_shape and n_features:
            if test_X_shape[1] != n_features:
                compatible = False
                issues.append(f"Test X feature sayısı model ile uyumsuz: "
                            f"{test_X_shape[1]} vs {n_features}")
        
        if test_X_shape and n_feature_names:
            if test_X_shape[1] != n_feature_names:
                compatible = False
                issues.append(f"Test X feature sayısı feature names ile uyumsuz: "
                            f"{test_X_shape[1]} vs {n_feature_names}")
        
        if self.verbose:
            if compatible:
                print("\n✓ TÜM DOSYALAR UYUMLU!")
                print(f"  Örnek sayısı: {test_X_shape[0] if test_X_shape else 'Unknown'}")
                print(f"  Feature sayısı: {test_X_shape[1] if test_X_shape else 'Unknown'}")
            else:
                print("\n⚠️  UYUMSUZLUK TESPİT EDİLDİ!")
                for issue in issues:
                    print(f"  • {issue}")
        
        return {
            'compatible': compatible,
            'issues': issues
        }
    
    def setup(self):
        """Ana setup işlemini yap."""
        
        # 1. Dosyaları doğrula
        validation_results = self.validate_all_files()
        
        # Eksik dosya varsa dur
        missing = [v for v in validation_results.values() if not v['exists']]
        if missing:
            print("\n❌ Setup tamamlanamadı. Eksik dosyaları ekleyin ve tekrar deneyin.")
            return False
        
        # 2. Uyumluluk kontrolü
        compatibility = self.generate_compatibility_report(validation_results)
        
        if not compatibility['compatible']:
            print("\n⚠️  Uyumsuzluklar var ancak devam edebilirsiniz.")
            response = input("Devam etmek istiyor musunuz? (e/h): ")
            if response.lower() != 'e':
                print("\nSetup iptal edildi.")
                return False
        
        # 3. Dosyaları kopyala
        self.copy_files_to_target(validation_results)
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("✅ SETUP TAMAMLANDI!")
            print("=" * 70)
            print(f"\nHedef klasör: {self.target_dir}")
            print("\nŞimdi simülasyonu çalıştırabilirsiniz:")
            print(f"python xharmony_simulation.py --data_dir {self.target_dir} --n_samples 100")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='X-HARMONY Veri Hazırlık ve Doğrulama'
    )
    
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Kaynak dosyaların bulunduğu klasör'
    )
    
    parser.add_argument(
        '--target_dir',
        type=str,
        default='./setup_xharmony_data_outputs/',
        help='Hedef klasör (varsayılan: ./setup_xharmony_data_outputs/)'
    )
    
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help='Doğrulama adımını atla (hızlı kopya)'
    )
    
    args = parser.parse_args()
    
    # Setup'ı çalıştır
    setup = XHARMONYDataSetup(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        verbose=True
    )
    
    success = setup.setup()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()