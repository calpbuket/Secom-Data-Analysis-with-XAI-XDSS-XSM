"""
================================================================================
X-HARMONY - TEZ ŞEKİL VE TABLO ÜRETİCİ
================================================================================

Bu script, X-HARMONY simülasyon sonuçlarından tez için kullanılabilecek
profesyonel şekiller ve tablolar üretir.

Kullanım:
    python generate_thesis_figures.py --results_dir <path> --output_dir <path>

Çıktılar:
    1. Confusion Matrix (XDSS overlay ile)
    2. ROC Curve + PR Curve
    3. XDSS Karar Dağılımı
    4. XSM Güvenlik Analizi
    5. Karşılaştırmalı Tablolar (LaTeX formatında)

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakterler için
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ThesisFigureGenerator:
    """Tez şekilleri ve tabloları üretici sınıf."""
    
    def __init__(self, results_dir: str, output_dir: str, language: str = 'tr'):
        """
        Figure generator'ı başlat.
        
        Args:
            results_dir: Simülasyon sonuçları klasörü
            output_dir: Şekillerin kaydedileceği klasör
            language: Dil ('tr' veya 'en')
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        
        # Çıktı klasörünü oluştur
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dil ayarları
        self.labels = self._get_labels()
        
        # Grafik stili
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        print("\n" + "=" * 70)
        print("TEZ ŞEKİL ÜRETİCİ")
        print("=" * 70)
        print(f"Sonuç klasörü: {self.results_dir}")
        print(f"Çıktı klasörü: {self.output_dir}")
        print(f"Dil: {self.language.upper()}")
        print("=" * 70)
    
    def _get_labels(self) -> Dict:
        """Dil etiketlerini getir."""
        if self.language == 'tr':
            return {
                'pass': 'PASS (Başarılı)',
                'fail': 'FAIL (Hatalı)',
                'true': 'Gerçek',
                'predicted': 'Tahmin',
                'stop': 'DURDUR',
                'check': 'KONTROL',
                'continue': 'DEVAM',
                'safe': 'GÜVENLİ',
                'warning': 'UYARI',
                'critical': 'KRİTİK',
                'probability': 'Olasılık',
                'confidence': 'Güven Skoru',
                'count': 'Sayı',
                'rate': 'Oran',
                'accuracy': 'Doğruluk',
                'precision': 'Kesinlik',
                'recall': 'Duyarlılık',
                'f1_score': 'F1 Skoru'
            }
        else:  # English
            return {
                'pass': 'PASS',
                'fail': 'FAIL',
                'true': 'True',
                'predicted': 'Predicted',
                'stop': 'STOP',
                'check': 'CHECK',
                'continue': 'CONTINUE',
                'safe': 'SAFE',
                'warning': 'WARNING',
                'critical': 'CRITICAL',
                'probability': 'Probability',
                'confidence': 'Confidence',
                'count': 'Count',
                'rate': 'Rate',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1 Score'
            }
    
    def load_results(self):
        """Simülasyon sonuçlarını yükle."""
        print("\n[1] Sonuçlar yükleniyor...")
        
        self.summary = pd.read_csv(self.results_dir / 'summary_statistics.csv')
        self.xdss_decisions = pd.read_csv(self.results_dir / 'xdss_decisions.csv')
        self.xsm_reports = pd.read_csv(self.results_dir / 'xsm_security_reports.csv')
        
        print(f"  ✓ {len(self.xdss_decisions)} XDSS kararı")
        print(f"  ✓ {len(self.xsm_reports)} XSM raporu")
    
    def figure1_confusion_matrix_with_xdss(self):
        """
        Şekil 1: Confusion Matrix + XDSS Overlay
        
        4x4 subplot:
        - Ana confusion matrix
        - XDSS kararlarıyla overlay
        """
        print("\n[2] Şekil 1: Confusion Matrix + XDSS...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Ana Confusion Matrix
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(
            self.xdss_decisions['y_true'],
            self.xdss_decisions['y_pred']
        )
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[self.labels['pass'], self.labels['fail']],
            yticklabels=[self.labels['pass'], self.labels['fail']],
            ax=axes[0, 0],
            cbar_kws={'label': self.labels['count']}
        )
        axes[0, 0].set_xlabel(f"{self.labels['predicted']}", fontsize=12)
        axes[0, 0].set_ylabel(f"{self.labels['true']}", fontsize=12)
        axes[0, 0].set_title('(a) Confusion Matrix', fontsize=14, fontweight='bold')
        
        # 2. XDSS Kararları ile Overlay
        # Her confusion cell için XDSS dağılımı
        for true_label in [0, 1]:
            for pred_label in [0, 1]:
                subset = self.xdss_decisions[
                    (self.xdss_decisions['y_true'] == true_label) &
                    (self.xdss_decisions['y_pred'] == pred_label)
                ]
                
                if len(subset) > 0:
                    xdss_dist = subset['decision'].value_counts()
                    
                    row = 1 if true_label == 1 else 0
                    col = 1 if pred_label == 1 else 0
                    
                    # Mini bar plot
                    ax_idx = (row, col) if row + col > 0 else (0, 1)
                    
                    if ax_idx != (0, 0):
                        xdss_dist.plot(
                            kind='bar',
                            ax=axes[ax_idx[0], ax_idx[1]],
                            color=['#d32f2f', '#ff9800', '#4caf50']
                        )
                        axes[ax_idx[0], ax_idx[1]].set_title(
                            f'({chr(97 + ax_idx[0]*2 + ax_idx[1])}) '
                            f'True={self.labels["pass"] if true_label==0 else self.labels["fail"]}, '
                            f'Pred={self.labels["pass"] if pred_label==0 else self.labels["fail"]}',
                            fontsize=12, fontweight='bold'
                        )
                        axes[ax_idx[0], ax_idx[1]].set_ylabel(self.labels['count'])
                        axes[ax_idx[0], ax_idx[1]].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_confusion_xdss.png')
        plt.close()
        print("  ✓ Kaydedildi: figure1_confusion_xdss.png")
    
    def figure2_xdss_decision_space(self):
        """
        Şekil 2: XDSS Karar Uzayı
        
        Scatter: p_fail vs confidence, XDSS kararlarına göre renklendirilmiş
        """
        print("\n[3] Şekil 2: XDSS Karar Uzayı...")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Renk haritası
        color_map = {
            'STOP': '#d32f2f',
            'CHECK': '#ff9800',
            'CONTINUE': '#4caf50'
        }
        
        for decision in ['CONTINUE', 'CHECK', 'STOP']:
            subset = self.xdss_decisions[self.xdss_decisions['decision'] == decision]
            
            ax.scatter(
                subset['y_prob'],
                subset['confidence'],
                label=self.labels[decision.lower()],
                color=color_map[decision],
                alpha=0.6,
                s=80,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Eşik çizgileri
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel(f'{self.labels["probability"]} (p_fail)', fontsize=13)
        ax.set_ylabel(self.labels['confidence'], fontsize=13)
        ax.set_title('XDSS Karar Uzayı: Olasılık vs Güven', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_xdss_space.png')
        plt.close()
        print("  ✓ Kaydedildi: figure2_xdss_space.png")
    
    def figure3_xsm_security_distribution(self):
        """
        Şekil 3: XSM Güvenlik Dağılımı
        
        2 subplot:
        - Güvenlik durumu dağılımı
        - Alert sayıları (kritik vs uyarı)
        """
        print("\n[4] Şekil 3: XSM Güvenlik Dağılımı...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Güvenlik durumu dağılımı
        status_counts = self.xsm_reports['status'].value_counts()
        colors = ['#4caf50', '#ff9800', '#d32f2f']
        
        axes[0].bar(
            [self.labels[s.lower()] for s in status_counts.index],
            status_counts.values,
            color=[colors[['SAFE', 'WARNING', 'CRITICAL'].index(s)] 
                  for s in status_counts.index]
        )
        axes[0].set_ylabel(self.labels['count'], fontsize=12)
        axes[0].set_title('(a) Güvenlik Durumu Dağılımı', 
                         fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=0)
        
        # 2. Alert sayıları
        alert_data = pd.DataFrame({
            self.labels['critical']: self.xsm_reports['n_critical'],
            self.labels['warning']: self.xsm_reports['n_warning'],
            self.labels['safe']: self.xsm_reports['n_info']
        })
        
        alert_data.boxplot(ax=axes[1], patch_artist=True, grid=False)
        axes[1].set_ylabel(self.labels['count'], fontsize=12)
        axes[1].set_title('(b) Alert Sayıları Dağılımı', 
                         fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_xsm_distribution.png')
        plt.close()
        print("  ✓ Kaydedildi: figure3_xsm_distribution.png")
    
    def figure4_integrated_analysis(self):
        """
        Şekil 4: XDSS + XSM Entegre Analiz
        
        Heatmap: XDSS kararları vs XSM durumları
        """
        print("\n[5] Şekil 4: Entegre Analiz...")
        
        # XDSS ve XSM'yi merge et
        merged = pd.merge(
            self.xdss_decisions[['sample_idx', 'decision']],
            self.xsm_reports[['sample_idx', 'status']],
            on='sample_idx'
        )
        
        # Crosstab
        ct = pd.crosstab(
            merged['decision'],
            merged['status'],
            normalize='index'
        ) * 100  # Yüzde olarak
        
        # Sıralama
        ct = ct.reindex(['STOP', 'CHECK', 'CONTINUE'], fill_value=0)
        ct = ct.reindex(columns=['CRITICAL', 'WARNING', 'SAFE'], fill_value=0)
        
        # Türkçe etiketler
        ct.index = [self.labels[d.lower()] for d in ct.index]
        ct.columns = [self.labels[c.lower()] for c in ct.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            ct, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': f'{self.labels["rate"]} (%)'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_xlabel('XSM Güvenlik Durumu', fontsize=13)
        ax.set_ylabel('XDSS Kararı', fontsize=13)
        ax.set_title('XDSS-XSM Entegre Analiz: Karar vs Güvenlik Durumu (%)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_integrated_heatmap.png')
        plt.close()
        print("  ✓ Kaydedildi: figure4_integrated_heatmap.png")
    
    def table1_performance_summary(self):
        """
        Tablo 1: Performans Özeti (LaTeX formatında)
        """
        print("\n[6] Tablo 1: Performans Özeti...")
        
        # İstatistikleri hesapla
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = self.xdss_decisions['y_true']
        y_pred = self.xdss_decisions['y_pred']
        
        metrics = {
            self.labels['accuracy']: accuracy_score(y_true, y_pred),
            self.labels['precision']: precision_score(y_true, y_pred, pos_label=1),
            self.labels['recall']: recall_score(y_true, y_pred, pos_label=1),
            self.labels['f1_score']: f1_score(y_true, y_pred, pos_label=1)
        }
        
        # LaTeX tablosu
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Model Performans Metrikleri}\n"
        latex_table += "\\label{tab:performance}\n"
        latex_table += "\\begin{tabular}{|l|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "Metrik & Değer \\\\\n"
        latex_table += "\\hline\n"
        
        for metric, value in metrics.items():
            latex_table += f"{metric} & {value:.4f} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        # Kaydet
        with open(self.output_dir / 'table1_performance.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("  ✓ Kaydedildi: table1_performance.tex")
        
        # CSV versiyonu
        pd.DataFrame([metrics]).to_csv(
            self.output_dir / 'table1_performance.csv',
            index=False
        )
        print("  ✓ Kaydedildi: table1_performance.csv")
    
    def table2_xdss_distribution(self):
        """
        Tablo 2: XDSS Karar Dağılımı
        """
        print("\n[7] Tablo 2: XDSS Karar Dağılımı...")
        
        decision_counts = self.xdss_decisions['decision'].value_counts()
        decision_pcts = (decision_counts / len(self.xdss_decisions)) * 100
        
        # DataFrame
        df = pd.DataFrame({
            'Karar': [self.labels[d.lower()] for d in decision_counts.index],
            'Sayı': decision_counts.values,
            'Yüzde (%)': decision_pcts.values.round(2)
        })
        
        # CSV
        df.to_csv(self.output_dir / 'table2_xdss_distribution.csv', index=False)
        
        # LaTeX
        latex = df.to_latex(index=False, caption='XDSS Karar Dağılımı', 
                           label='tab:xdss_dist')
        
        with open(self.output_dir / 'table2_xdss_distribution.tex', 'w', encoding='utf-8') as f:
            f.write(latex)
        
        print("  ✓ Kaydedildi: table2_xdss_distribution.csv & .tex")
    
    def table3_xsm_summary(self):
        """
        Tablo 3: XSM Güvenlik Özeti
        """
        print("\n[8] Tablo 3: XSM Güvenlik Özeti...")
        
        status_counts = self.xsm_reports['status'].value_counts()
        status_pcts = (status_counts / len(self.xsm_reports)) * 100
        
        # DataFrame
        df = pd.DataFrame({
            'Durum': [self.labels[s.lower()] for s in status_counts.index],
            'Sayı': status_counts.values,
            'Yüzde (%)': status_pcts.values.round(2),
            'Ort. Kritik Alert': [
                self.xsm_reports[self.xsm_reports['status'] == s]['n_critical'].mean()
                for s in status_counts.index
            ],
            'Ort. Uyarı Alert': [
                self.xsm_reports[self.xsm_reports['status'] == s]['n_warning'].mean()
                for s in status_counts.index
            ]
        })
        
        # CSV
        df.to_csv(self.output_dir / 'table3_xsm_summary.csv', index=False)
        
        # LaTeX
        latex = df.to_latex(index=False, caption='XSM Güvenlik Özeti', 
                           label='tab:xsm_summary', float_format='%.2f')
        
        with open(self.output_dir / 'table3_xsm_summary.tex', 'w', encoding='utf-8') as f:
            f.write(latex)
        
        print("  ✓ Kaydedildi: table3_xsm_summary.csv & .tex")
    
    def generate_all(self):
        """Tüm şekil ve tabloları üret."""
        
        # Sonuçları yükle
        self.load_results()
        
        # Şekiller
        self.figure1_confusion_matrix_with_xdss()
        self.figure2_xdss_decision_space()
        self.figure3_xsm_security_distribution()
        self.figure4_integrated_analysis()
        
        # Tablolar
        self.table1_performance_summary()
        self.table2_xdss_distribution()
        self.table3_xsm_summary()
        
        print("\n" + "=" * 70)
        print("✅ TÜM ŞEKÄ°L VE TABLOLAR OLUŞTURULDU!")
        print("=" * 70)
        print(f"\nÇıktı klasörü: {self.output_dir}")
        print("\nOluşturulan dosyalar:")
        print("  Şekiller:")
        print("    - figure1_confusion_xdss.png")
        print("    - figure2_xdss_space.png")
        print("    - figure3_xsm_distribution.png")
        print("    - figure4_integrated_heatmap.png")
        print("  Tablolar:")
        print("    - table1_performance.csv / .tex")
        print("    - table2_xdss_distribution.csv / .tex")
        print("    - table3_xsm_summary.csv / .tex")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='X-HARMONY Tez Şekil ve Tablo Üreticisi'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./xharmony_simulation_outputs', 
        help='Simülasyon sonuçları klasörü'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./generate_thesis_figures_outputs',
        help='Şekillerin kaydedileceği klasör'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='tr',
        choices=['tr', 'en'],
        help='Dil seçimi (tr veya en)'
    )
    
    args = parser.parse_args()
    
    # Generator'ı oluştur ve çalıştır
    generator = ThesisFigureGenerator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        language=args.language
    )
    
    generator.generate_all()


if __name__ == "__main__":
    main()
