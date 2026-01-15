"""
================================================================================
X-HARMONY SİMÜLASYON ORTAMI
XDSS + XSM Entegre Sistem Simülasyonu
================================================================================

Bu script, X-HARMONY mimarisinin tam entegrasyonunu gösterir:
    1. XGBoost model tahminleri
    2. SHAP açıklamalar
    3. XDSS karar destek
    4. XSM güvenlik kontrolleri
    5. Entegre raporlama

Kullanım:
    python xharmony_simulation.py --data_dir <path> --output_dir <path> [--n_samples N]

Gereksinimler:
    - save_model_and_test_outputs/ klasöründe model ve test verisi
    - xai_analysis_outputs/ klasöründe SHAP değerleri ve kurallar
    - xdss_module.py
    - xsm_module.py

================================================================================
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# X-HARMONY modüllerini import et
from xdss_module import XDSSModule
from xsm_module import XSMModule


class XHARMONYSimulator:
    """
    X-HARMONY Simülasyon Sınıfı.
    
    XDSS ve XSM modüllerini entegre eder ve üretim simülasyonu yapar.
    """
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        test_labels_path: str,
        shap_values_path: str,
        rules_path: str,
        feature_names_path: str,
        verbose: bool = True
    ):
        """
        Simülatörü başlat.
        
        Args:
            model_path: XGBoost model pickle dosyası
            test_data_path: Test X CSV
            test_labels_path: Test y CSV
            shap_values_path: SHAP values .npy dosyası
            rules_path: XDSS kurallar CSV
            feature_names_path: Feature isimleri TXT
            verbose: Detaylı çıktı
        """
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("X-HARMONY SİMÜLATÖR BAŞLATILIYOR")
            print("=" * 70)
        
        # 1. Model ve veriyi yükle
        self._load_model_and_data(
            model_path, test_data_path, test_labels_path,
            shap_values_path, feature_names_path
        )
        
        # 2. XDSS ve XSM modüllerini başlat
        self.xdss = XDSSModule(rules_path=rules_path, verbose=verbose)
        self.xsm = XSMModule(verbose=verbose)
        
        if self.verbose:
            print("\n✓ X-HARMONY Simülatör hazır!")
            print("=" * 70)
    
    def _load_model_and_data(
        self,
        model_path: str,
        test_data_path: str,
        test_labels_path: str,
        shap_values_path: str,
        feature_names_path: str
    ):
        """Model, test verisi ve SHAP değerlerini yükle."""
        
        if self.verbose:
            print("\n📂 Dosyalar yükleniyor...")
        
        # Model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if self.verbose:
            print(f"  ✓ Model: {type(self.model).__name__}")
        
        # Test verisi
        self.X_test = pd.read_csv(test_data_path)
        self.y_test = pd.read_csv(test_labels_path)['Pass/Fail'].values
        if self.verbose:
            print(f"  ✓ Test verisi: {self.X_test.shape}")
            print(f"    Pass (0): {(self.y_test == 0).sum()}")
            print(f"    Fail (1): {(self.y_test == 1).sum()}")
        
        # SHAP değerleri
        self.shap_values = np.load(shap_values_path)
        if self.verbose:
            print(f"  ✓ SHAP values: {self.shap_values.shape}")
        
        # Feature isimleri
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        if self.verbose:
            print(f"  ✓ Feature names: {len(self.feature_names)}")
        
        # Model tahminlerini hesapla
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        if self.verbose:
            print(f"  ✓ Tahminler hesaplandı")
    
    def run_single_simulation(self, sample_idx: int) -> dict:
        """
        Tek bir örnek için tam simülasyon.
        
        Args:
            sample_idx: Test setinden örnek indexi
            
        Returns:
            simulation_result: Simülasyon sonuçları
        """
        # Örnek verisini al
        x_sample = self.X_test.iloc[sample_idx].values
        y_true = self.y_test[sample_idx]
        y_pred = self.y_pred[sample_idx]
        y_prob = self.y_prob[sample_idx]
        shap_sample = self.shap_values[sample_idx]
        
        # 1. XDSS Kararı
        xdss_decision = self.xdss.xdss_decision(
            pred_prob=y_prob,
            shap_values=shap_sample,
            feature_values=x_sample,
            feature_names=self.feature_names
        )
        
        # 2. XSM Güvenlik Kontrolü
        xsm_report = self.xsm.xsm_security_check(
            pred_prob=y_prob,
            shap_values=shap_sample,
            feature_values=x_sample,
            feature_names=self.feature_names
        )
        
        # 3. Sonuçları birleştir
        simulation_result = {
            'sample_idx': sample_idx,
            'y_true': int(y_true),
            'y_pred': int(y_pred),
            'y_prob': float(y_prob),
            'xdss_decision': xdss_decision,
            'xsm_report': xsm_report,
            'prediction_correct': (y_true == y_pred)
        }
        
        return simulation_result
    
    def run_batch_simulation(self, n_samples: int = None) -> list:
        """
        Toplu simülasyon çalıştır.
        
        Args:
            n_samples: Simüle edilecek örnek sayısı (None = tümü)
            
        Returns:
            results: Simülasyon sonuçları listesi
        """
        if n_samples is None:
            n_samples = len(self.X_test)
        else:
            n_samples = min(n_samples, len(self.X_test))
        
        if self.verbose:
            print(f"\n🚀 Batch simülasyon başlatılıyor: {n_samples} örnek")
            print("=" * 70)
        
        results = []
        
        for i in range(n_samples):
            result = self.run_single_simulation(i)
            results.append(result)
            
            if self.verbose and (i + 1) % 50 == 0:
                print(f"  İşlenen: {i + 1}/{n_samples}")
        
        if self.verbose:
            print(f"\n✓ Simülasyon tamamlandı: {n_samples} örnek")
        
        return results
    
    def generate_comprehensive_report(
        self, 
        results: list,
        output_dir: str
    ):
        """
        Kapsamlı simülasyon raporu üret.
        
        Args:
            results: run_batch_simulation() çıktısı
            output_dir: Raporun kaydedileceği klasör
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("KAPSAMLI SİMÜLASYON RAPORU OLUŞTURULUYOR")
        print("=" * 70)
        
        # 1. Özet istatistikler
        self._generate_summary_stats(results, output_dir)
        
        # 2. XDSS analizi
        self._analyze_xdss_decisions(results, output_dir)
        
        # 3. XSM analizi
        self._analyze_xsm_security(results, output_dir)
        
        # 4. Confusion matrix + XDSS/XSM overlay
        self._generate_confusion_analysis(results, output_dir)
        
        # 5. En kritik vakaları çıkar
        self._extract_critical_cases(results, output_dir)
        
        # 6. Görselleştirmeler
        self._create_visualizations(results, output_dir)
        
        print("\n✓ Rapor oluşturma tamamlandı!")
        print(f"📁 Çıktı klasörü: {output_dir}")
        print("=" * 70)
    
    def _generate_summary_stats(self, results: list, output_dir: str):
        """Özet istatistikler."""
        print("\n[1/6] Özet istatistikler hesaplanıyor...")
        
        n_total = len(results)
        n_correct = sum(1 for r in results if r['prediction_correct'])
        accuracy = n_correct / n_total
        
        # XDSS dağılımı
        xdss_counts = {
            'STOP': sum(1 for r in results if r['xdss_decision']['decision'] == 'STOP'),
            'CHECK': sum(1 for r in results if r['xdss_decision']['decision'] == 'CHECK'),
            'CONTINUE': sum(1 for r in results if r['xdss_decision']['decision'] == 'CONTINUE')
        }
        
        # XSM dağılımı
        xsm_counts = {
            'CRITICAL': sum(1 for r in results if r['xsm_report']['status'] == 'CRITICAL'),
            'WARNING': sum(1 for r in results if r['xsm_report']['status'] == 'WARNING'),
            'SAFE': sum(1 for r in results if r['xsm_report']['status'] == 'SAFE')
        }
        
        # Rapor oluştur
        summary = {
            'total_samples': n_total,
            'model_accuracy': accuracy,
            'xdss_stop': xdss_counts['STOP'],
            'xdss_check': xdss_counts['CHECK'],
            'xdss_continue': xdss_counts['CONTINUE'],
            'xsm_critical': xsm_counts['CRITICAL'],
            'xsm_warning': xsm_counts['WARNING'],
            'xsm_safe': xsm_counts['SAFE']
        }
        
        # CSV'ye kaydet
        pd.DataFrame([summary]).to_csv(
            f'{output_dir}/summary_statistics.csv', index=False
        )
        
        # Konsol çıktısı
        print(f"  ✓ Model Accuracy: {accuracy:.4f}")
        print(f"  ✓ XDSS: STOP={xdss_counts['STOP']}, "
              f"CHECK={xdss_counts['CHECK']}, "
              f"CONTINUE={xdss_counts['CONTINUE']}")
        print(f"  ✓ XSM: CRITICAL={xsm_counts['CRITICAL']}, "
              f"WARNING={xsm_counts['WARNING']}, "
              f"SAFE={xsm_counts['SAFE']}")
    
    def _analyze_xdss_decisions(self, results: list, output_dir: str):
        """XDSS kararlarını analiz et."""
        print("\n[2/6] XDSS kararları analiz ediliyor...")
        
        xdss_records = []
        
        for r in results:
            xdss = r['xdss_decision']
            record = {
                'sample_idx': r['sample_idx'],
                'y_true': r['y_true'],
                'y_pred': r['y_pred'],
                'y_prob': r['y_prob'],
                'decision': xdss['decision'],
                'confidence': xdss['confidence'],
                'n_critical': xdss['n_critical'],
                'n_warning': xdss['n_warning'],
                'action': xdss['action']
            }
            xdss_records.append(record)
        
        df = pd.DataFrame(xdss_records)
        df.to_csv(f'{output_dir}/xdss_decisions.csv', index=False)
        print(f"  ✓ XDSS kararları kaydedildi")
    
    def _analyze_xsm_security(self, results: list, output_dir: str):
        """XSM güvenlik kontrollerini analiz et."""
        print("\n[3/6] XSM güvenlik kontrolleri analiz ediliyor...")
        
        xsm_records = []
        
        for r in results:
            xsm = r['xsm_report']
            record = {
                'sample_idx': r['sample_idx'],
                'status': xsm['status'],
                'n_critical': xsm['n_critical'],
                'n_warning': xsm['n_warning'],
                'n_info': xsm['n_info'],
                'recommendation': xsm['recommendation']
            }
            xsm_records.append(record)
        
        df = pd.DataFrame(xsm_records)
        df.to_csv(f'{output_dir}/xsm_security_reports.csv', index=False)
        print(f"  ✓ XSM raporları kaydedildi")
    
    def _generate_confusion_analysis(self, results: list, output_dir: str):
        """Confusion matrix + XDSS/XSM overlay."""
        print("\n[4/6] Confusion matrix analizi yapılıyor...")
        
        records = []
        
        for r in results:
            record = {
                'y_true': r['y_true'],
                'y_pred': r['y_pred'],
                'xdss_decision': r['xdss_decision']['decision'],
                'xsm_status': r['xsm_report']['status']
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Confusion matrix ile XDSS kararlarını çaprazla
        confusion_xdss = pd.crosstab(
            index=[df['y_true'], df['y_pred']],
            columns=df['xdss_decision'],
            rownames=['True', 'Pred'],
            colnames=['XDSS']
        )
        
        confusion_xdss.to_csv(f'{output_dir}/confusion_xdss_matrix.csv')
        print(f"  ✓ Confusion + XDSS matrisi kaydedildi")
    
    def _extract_critical_cases(self, results: list, output_dir: str):
        """En kritik vakaları çıkar."""
        print("\n[5/6] Kritik vakalar çıkarılıyor...")
        
        # XDSS STOP kararları
        stop_cases = [r for r in results if r['xdss_decision']['decision'] == 'STOP']
        
        # XSM CRITICAL alert'leri
        critical_cases = [r for r in results if r['xsm_report']['status'] == 'CRITICAL']
        
        # Her iki koşulu da sağlayanlar
        double_critical = [r for r in results 
                          if r['xdss_decision']['decision'] == 'STOP' 
                          and r['xsm_report']['status'] == 'CRITICAL']
        
        # False Negatives (model Pass dedi ama gerçek Fail)
        fn_cases = [r for r in results if r['y_true'] == 1 and r['y_pred'] == 0]
        
        # Raporlama
        critical_summary = {
            'xdss_stop_count': len(stop_cases),
            'xsm_critical_count': len(critical_cases),
            'double_critical_count': len(double_critical),
            'false_negative_count': len(fn_cases)
        }
        
        pd.DataFrame([critical_summary]).to_csv(
            f'{output_dir}/critical_cases_summary.csv', index=False
        )
        
        # Detaylı listeler
        if double_critical:
            double_critical_df = pd.DataFrame([
                {
                    'sample_idx': r['sample_idx'],
                    'y_true': r['y_true'],
                    'y_pred': r['y_pred'],
                    'y_prob': r['y_prob']
                }
                for r in double_critical
            ])
            double_critical_df.to_csv(
                f'{output_dir}/double_critical_cases.csv', index=False
            )
        
        print(f"  ✓ XDSS STOP: {len(stop_cases)}")
        print(f"  ✓ XSM CRITICAL: {len(critical_cases)}")
        print(f"  ✓ Double critical: {len(double_critical)}")
        print(f"  ✓ False Negatives: {len(fn_cases)}")
    
    def _create_visualizations(self, results: list, output_dir: str):
        """Görselleştirmeler oluştur."""
        print("\n[6/6] Görselleştirmeler oluşturuluyor...")
        
        # 1. XDSS Decision Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        xdss_decisions = [r['xdss_decision']['decision'] for r in results]
        xsm_statuses = [r['xsm_report']['status'] for r in results]
        
        # XDSS
        xdss_counts = pd.Series(xdss_decisions).value_counts()
        axes[0].bar(xdss_counts.index, xdss_counts.values, 
                   color=['#d32f2f', '#ff9800', '#4caf50'])
        axes[0].set_title('XDSS Decision Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].set_xlabel('Decision')
        
        # XSM
        xsm_counts = pd.Series(xsm_statuses).value_counts()
        axes[1].bar(xsm_counts.index, xsm_counts.values,
                   color=['#d32f2f', '#ff9800', '#4caf50'])
        axes[1].set_title('XSM Security Status Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count')
        axes[1].set_xlabel('Status')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/xdss_xsm_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Probability vs Confidence scatter
        fig, ax = plt.subplots(figsize=(10, 7))
        
        probs = [r['y_prob'] for r in results]
        confidences = [r['xdss_decision']['confidence'] for r in results]
        colors = [r['xdss_decision']['decision'] for r in results]
        
        color_map = {'STOP': '#d32f2f', 'CHECK': '#ff9800', 'CONTINUE': '#4caf50'}
        
        for decision in ['STOP', 'CHECK', 'CONTINUE']:
            mask = [c == decision for c in colors]
            ax.scatter(
                [p for p, m in zip(probs, mask) if m],
                [c for c, m in zip(confidences, mask) if m],
                label=decision,
                color=color_map[decision],
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('Fail Probability (p_fail)', fontsize=12)
        ax.set_ylabel('XDSS Confidence', fontsize=12)
        ax.set_title('XDSS Decision Space: Probability vs Confidence', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/xdss_decision_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Görselleştirmeler kaydedildi")
    
    def print_sample_report(self, sample_idx: int):
        """
        Tek bir örnek için detaylı rapor yazdır.
        
        Args:
            sample_idx: Örnek indexi
        """
        result = self.run_single_simulation(sample_idx)
        
        print("\n" + "=" * 70)
        print(f"X-HARMONY DETAYLI RAPOR - ÖRNEK #{sample_idx}")
        print("=" * 70)
        
        print(f"\n📊 GERÇEK DURUM ve MODEL TAHMİNİ:")
        print(f"  Gerçek: {'FAIL' if result['y_true'] == 1 else 'PASS'}")
        print(f"  Tahmin: {'FAIL' if result['y_pred'] == 1 else 'PASS'}")
        print(f"  Olasılık: {result['y_prob']:.4f}")
        print(f"  Doğru mu: {'✓' if result['prediction_correct'] else '✗'}")
        
        print(self.xdss.format_decision_report(result['xdss_decision']))
        print(self.xsm.format_security_report(result['xsm_report']))


# =============================================================================
# MAIN - COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='X-HARMONY Simülasyon Ortamı'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./setup_xharmony_data_outputs',
        help='Veri klasörü yolu'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./xharmony_simulation_outputs',
        help='Çıktı klasörü yolu'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=100,
        help='Simüle edilecek örnek sayısı'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo mode (örnek veri ile)'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        print("\n🎮 DEMO MODE: Örnek senaryolar çalıştırılıyor...\n")
        
        # XDSS ve XSM'nin kendi demo'larını çalıştır
        from xdss_module import demo_xdss
        from xsm_module import demo_xsm
        
        demo_xdss()
        demo_xsm()
        
        print("\n✓ Demo tamamlandı!")
        return
    
    # Normal simülasyon modu
    data_dir = args.data_dir
    output_dir = args.output_dir
    n_samples = args.n_samples
    
    # Dosya yolları
    model_path = f'{data_dir}/xai_final_model.pkl'
    test_data_path = f'{data_dir}/xai_test_X.csv'
    test_labels_path = f'{data_dir}/xai_test_y.csv'
    shap_values_path = f'{data_dir}/global_shap_values.npy'
    rules_path = f'{data_dir}/xdss_xsm_rules.csv'
    feature_names_path = f'{data_dir}/xai_feature_names.txt'
    
    # Simülatörü başlat
    simulator = XHARMONYSimulator(
        model_path=model_path,
        test_data_path=test_data_path,
        test_labels_path=test_labels_path,
        shap_values_path=shap_values_path,
        rules_path=rules_path,
        feature_names_path=feature_names_path,
        verbose=True
    )
    
    # Simülasyonu çalıştır
    results = simulator.run_batch_simulation(n_samples=n_samples)
    
    # Kapsamlı rapor üret
    simulator.generate_comprehensive_report(results, output_dir)
    
    # Birkaç örnek için detaylı rapor yazdır
    print("\n\n" + "=" * 70)
    print("ÖRNEK DETAYLI RAPORLAR")
    print("=" * 70)
    
    for i in range(min(3, len(results))):
        simulator.print_sample_report(i)
    
    print("\n\n✅ X-HARMONY Simülasyonu başarıyla tamamlandı!")
    print(f"📁 Tüm sonuçlar: {output_dir}")


if __name__ == "__main__":
    main()
