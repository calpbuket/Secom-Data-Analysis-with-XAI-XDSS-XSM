"""
================================================================================
X-HARMONY V2 - HUMAN-IN-THE-LOOP SİMÜLASYON
Operatör-Sistem Çatışması ve Güvenlik Kilidi Entegrasyonu
================================================================================

TEZ BÖLÜMÜ: 4.5 - Human-in-the-Loop ve Explainable Safety Gate Mimarisi

Bu simülasyon, X-HARMONY mimarisinin tam akışını gösterir:
    1. Model Tahmini (XGBoost)
    2. XAI Açıklamalar (SHAP)
    3. XDSS Karar Destek → Öneri
    4. Operatör Karar Katmanı → Nihai karar (TEZ 4.5.1)
    5. Çatışma Tespiti → XDSS vs Operatör (TEZ 4.5.2)
    6. XSM Interlock → Güvenlik kilidi (TEZ 4.5.3)
    7. Final Aksiyon Yürütme

Kullanım:
    python xharmony_simulation_v2.py --data_dir <path> --n_samples 100 --operator_profile CAUTIOUS_MID

Yazar: X-HARMONY Implementation - Thesis Chapter 4.5
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
from typing import Dict, List
warnings.filterwarnings('ignore')

# X-HARMONY modülleri
from xdss_module import XDSSModule
from xsm_module import XSMModule

# Yeni modüller (TEZ 4.5)
from operator_decision_module import (
    OperatorDecisionModule, 
    OperatorProfile,
    OperatorBehaviorType,
    create_operator_profiles
)
from conflict_detector import ConflictDetector
from xsm_interlock import XSMInterlock


class XHARMONYSimulatorV2:
    """
    X-HARMONY Simülatör V2 - Human-in-the-Loop ile genişletilmiş.
    
    TEZ 4.5: Bu sınıf operatör-sistem etkileşimini ve
    güvenlik kilidi mekanizmasını simüle eder.
    """
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        test_labels_path: str,
        shap_values_path: str,
        rules_path: str,
        feature_names_path: str,
        operator_profile_name: str = "EXPERT_COMPLIANT",
        verbose: bool = True
    ):
        """
        Simülatörü başlat.
        
        Args:
            model_path: XGBoost model dosyası
            test_data_path: Test X CSV
            test_labels_path: Test y CSV
            shap_values_path: SHAP values NPY
            rules_path: XDSS kurallar CSV
            feature_names_path: Feature isimleri TXT
            operator_profile_name: Operatör profil adı
            verbose: Detaylı çıktı
        """
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("X-HARMONY V2 SİMÜLATÖR BAŞLATILIYOR")
            print("TEZ 4.5: Human-in-the-Loop + Safety Interlock")
            print("=" * 70)
        
        # 1. Model ve veriyi yükle
        self._load_model_and_data(
            model_path, test_data_path, test_labels_path,
            shap_values_path, feature_names_path
        )
        
        # 2. Orijinal modülleri başlat
        self.xdss = XDSSModule(rules_path=rules_path, verbose=verbose)
        self.xsm = XSMModule(verbose=verbose)
        
        # 3. YENİ MODÜLLER (TEZ 4.5)
        
        # 3a. Operatör Karar Modülü (TEZ 4.5.1)
        profiles = create_operator_profiles()
        operator_profile = profiles.get(operator_profile_name, profiles["EXPERT_COMPLIANT"])
        self.operator_module = OperatorDecisionModule(
            operator_profile=operator_profile,
            verbose=verbose
        )
        
        # 3b. Çatışma Tespit Mekanizması (TEZ 4.5.2)
        self.conflict_detector = ConflictDetector(verbose=verbose)
        
        # 3c. XSM Interlock (TEZ 4.5.3)
        self.xsm_interlock = XSMInterlock(verbose=verbose)
        
        if self.verbose:
            print("\n✓ X-HARMONY V2 Simülatör hazır!")
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
        
        # SHAP değerleri
        self.shap_values = np.load(shap_values_path)
        if self.verbose:
            print(f"  ✓ SHAP values: {self.shap_values.shape}")
        
        # Feature isimleri
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        if self.verbose:
            print(f"  ✓ Feature names: {len(self.feature_names)}")
        
        # Model tahminleri
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        if self.verbose:
            print(f"  ✓ Tahminler hesaplandı")
    
    def run_single_simulation(self, sample_idx: int) -> Dict:
        """
        Tek bir örnek için tam simülasyon.
        
        TEZ 4.5: Bu fonksiyon tüm akışı simüle eder:
        Model → XDSS → Operatör → Çatışma → XSM Interlock → Final Aksiyon
        
        Args:
            sample_idx: Test setinden örnek indexi
            
        Returns:
            simulation_result: Kapsamlı simülasyon sonuçları
        """
        
        # Örnek verisini al
        x_sample = self.X_test.iloc[sample_idx].values
        y_true = self.y_test[sample_idx]
        y_pred = self.y_pred[sample_idx]
        y_prob = self.y_prob[sample_idx]
        shap_sample = self.shap_values[sample_idx]
        
        # =====================================================================
        # ADIM 1: XDSS KARARI (Sistem Önerisi)
        # =====================================================================
        xdss_decision = self.xdss.xdss_decision(
            pred_prob=y_prob,
            shap_values=shap_sample,
            feature_values=x_sample,
            feature_names=self.feature_names
        )
        
        # =====================================================================
        # ADIM 2: OPERATÖR KARARI (TEZ 4.5.1)
        # =====================================================================
        operator_decision = self.operator_module.make_decision(
            xdss_recommendation=xdss_decision['decision'],
            xdss_confidence=xdss_decision['confidence'],
            model_prob=y_prob,
            context={'sample_idx': sample_idx}
        )
        
        # =====================================================================
        # ADIM 3: ÇATIŞMA TESPİTİ (TEZ 4.5.2)
        # =====================================================================
        conflict_report = self.conflict_detector.detect_conflict(
            xdss_action=xdss_decision['decision'],
            operator_action=operator_decision['operator_action'],
            xdss_confidence=xdss_decision['confidence'],
            operator_confidence=operator_decision['operator_confidence'],
            model_prob=y_prob
        )
        
        # =====================================================================
        # ADIM 4: XSM GÜVENLİK KONTROLÜ (TEZ 4.5.2)
        # =====================================================================
        xsm_report = self.xsm.xsm_security_check(
            pred_prob=y_prob,
            shap_values=shap_sample,
            feature_values=x_sample,
            feature_names=self.feature_names
        )
        
        # =====================================================================
        # ADIM 5: XSM INTERLOCK (GÜVENLİK KİLİDİ) (TEZ 4.5.3)
        # =====================================================================
        interlock_report = self.xsm_interlock.interlock_decision(
            conflict_report=conflict_report,
            model_prob=y_prob,
            xdss_confidence=xdss_decision['confidence'],
            operator_profile={
                'experience_years': self.operator_module.profile.experience_years,
                'historical_accuracy': self.operator_module.profile.historical_accuracy
            },
            shap_values=shap_sample,
            feature_values=x_sample,
            xsm_anomaly_report=xsm_report
        )
        
        # =====================================================================
        # SONUÇLARI BİRLEŞTİR
        # =====================================================================
        simulation_result = {
            # Temel bilgiler
            'sample_idx': sample_idx,
            'y_true': int(y_true),
            'y_pred': int(y_pred),
            'y_prob': float(y_prob),
            'prediction_correct': (y_true == y_pred),
            
            # XDSS
            'xdss_action': xdss_decision['decision'],
            'xdss_confidence': xdss_decision['confidence'],
            'xdss_full_report': xdss_decision,
            
            # Operatör (TEZ 4.5.1)
            'operator_action': operator_decision['operator_action'],
            'operator_confidence': operator_decision['operator_confidence'],
            'operator_reasoning': operator_decision['reasoning'],
            'operator_override': operator_decision['override_flag'],
            
            # Çatışma (TEZ 4.5.2)
            'conflict_flag': conflict_report['conflict_flag'],
            'conflict_type': conflict_report['conflict_type'].value if hasattr(conflict_report['conflict_type'], 'value') else str(conflict_report['conflict_type']),
            'conflict_severity': conflict_report['severity'].value if hasattr(conflict_report['severity'], 'value') else str(conflict_report['severity']),
            'conflict_risk_score': conflict_report['risk_score'],
            
            # XSM Interlock (TEZ 4.5.3)
            'interlock_decision': interlock_report['decision'].value if hasattr(interlock_report['decision'], 'value') else str(interlock_report['decision']),
            'interlock_risk_score': interlock_report['risk_score'],
            'final_action': interlock_report['final_action'],
            'interlock_triggered': interlock_report['interlock_triggered'],
            
            # XSM güvenlik
            'xsm_status': xsm_report['status'],
            'xsm_report': xsm_report
        }
        
        return simulation_result
    
    def run_batch_simulation(self, n_samples: int = None) -> List[Dict]:
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
        results: List[Dict],
        output_dir: str
    ):
        """
        Kapsamlı simülasyon raporu üret (TEZ için).
        
        Args:
            results: run_batch_simulation() çıktısı
            output_dir: Raporun kaydedileceği klasör
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("KAPSAMLI SİMÜLASYON RAPORU OLUŞTURULUYOR (TEZ 4.5)")
        print("=" * 70)
        
        # 1. Özet istatistikler
        self._generate_summary_stats_v2(results, output_dir)
        
        # 2. Operatör analizi (TEZ 4.5.1)
        self._analyze_operator_decisions(results, output_dir)
        
        # 3. Çatışma analizi (TEZ 4.5.2)
        self._analyze_conflicts(results, output_dir)
        
        # 4. XSM Interlock analizi (TEZ 4.5.3)
        self._analyze_interlock_decisions(results, output_dir)
        
        # 5. XDSS ve XSM orijinal analizler
        self._analyze_xdss_xsm(results, output_dir)
        
        # 6. Kritik vakalar
        self._extract_critical_cases_v2(results, output_dir)
        
        # 7. Görselleştirmeler
        self._create_visualizations_v2(results, output_dir)
        
        print(f"\n✓ Tüm raporlar kaydedildi: {output_dir}")
    
    def _generate_summary_stats_v2(self, results: List[Dict], output_dir: str):
        """Özet istatistikler (genişletilmiş)."""
        print("\n[1/7] Özet istatistikler hesaplanıyor...")
        
        df = pd.DataFrame(results)
        
        summary = {
            # Model
            'total_samples': len(results),
            'model_accuracy': (df['y_true'] == df['y_pred']).mean(),
            
            # XDSS
            'xdss_stop': (df['xdss_action'] == 'STOP').sum(),
            'xdss_check': (df['xdss_action'] == 'CHECK').sum(),
            'xdss_continue': (df['xdss_action'] == 'CONTINUE').sum(),
            
            # Operatör (TEZ 4.5.1)
            'operator_override_rate': df['operator_override'].mean(),
            'operator_avg_confidence': df['operator_confidence'].mean(),
            
            # Çatışma (TEZ 4.5.2)
            'conflict_rate': df['conflict_flag'].mean(),
            'critical_conflicts': (df['conflict_severity'] == 'critical').sum(),
            'avg_conflict_risk': df[df['conflict_flag']]['conflict_risk_score'].mean() if df['conflict_flag'].any() else 0,
            
            # XSM Interlock (TEZ 4.5.3)
            'interlock_allow_rate': (df['interlock_decision'] == 'allow').mean(),
            'interlock_deny_rate': (df['interlock_decision'] == 'deny').mean(),
            'interlock_escalate_rate': (df['interlock_decision'] == 'escalate').mean(),
            'avg_interlock_risk': df[df['interlock_triggered']]['interlock_risk_score'].mean() if df['interlock_triggered'].any() else 0,
            
            # XSM
            'xsm_critical': (df['xsm_status'] == 'CRITICAL').sum()
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'{output_dir}/simulation_summary_v2.csv', index=False)
        
        print(f"  ✓ Model Accuracy: {summary['model_accuracy']:.2%}")
        print(f"  ✓ Operatör Override Oranı: {summary['operator_override_rate']:.2%}")
        print(f"  ✓ Çatışma Oranı: {summary['conflict_rate']:.2%}")
        print(f"  ✓ Interlock DENY Oranı: {summary['interlock_deny_rate']:.2%}")
    
    def _analyze_operator_decisions(self, results: List[Dict], output_dir: str):
        """Operatör karar analizi (TEZ 4.5.1)."""
        print("\n[2/7] Operatör kararları analiz ediliyor (TEZ 4.5.1)...")
        
        df = pd.DataFrame(results)
        
        # Operatör aksiyonlarının dağılımı
        operator_actions = df['operator_action'].value_counts()
        operator_actions.to_csv(f'{output_dir}/operator_actions_distribution.csv')
        
        # Override durumları
        override_analysis = df.groupby('operator_override').agg({
            'operator_confidence': 'mean',
            'y_prob': 'mean',
            'sample_idx': 'count'
        }).rename(columns={'sample_idx': 'count'})
        override_analysis.to_csv(f'{output_dir}/operator_override_analysis.csv')
        
        # Operatör istatistikleri
        operator_stats = self.operator_module.get_operator_statistics()
        pd.DataFrame([operator_stats]).to_csv(
            f'{output_dir}/operator_statistics.csv', index=False
        )
        
        print(f"  ✓ Override oranı: {operator_stats['override_rate']:.2%}")
        print(f"  ✓ Toplam karar: {operator_stats['total_decisions']}")
    
    def _analyze_conflicts(self, results: List[Dict], output_dir: str):
        """Çatışma analizi (TEZ 4.5.2)."""
        print("\n[3/7] Çatışmalar analiz ediliyor (TEZ 4.5.2)...")
        
        df = pd.DataFrame(results)
        conflicts = df[df['conflict_flag'] == True]
        
        if len(conflicts) > 0:
            # Çatışma tipleri
            conflict_types = conflicts['conflict_type'].value_counts()
            conflict_types.to_csv(f'{output_dir}/conflict_types_distribution.csv')
            
            # Çatışma şiddeti
            conflict_severity = conflicts['conflict_severity'].value_counts()
            conflict_severity.to_csv(f'{output_dir}/conflict_severity_distribution.csv')
            
            # Çatışma risk skorları
            conflict_risk_stats = conflicts['conflict_risk_score'].describe()
            conflict_risk_stats.to_csv(f'{output_dir}/conflict_risk_statistics.csv')
            
            print(f"  ✓ Toplam çatışma: {len(conflicts)}")
            print(f"  ✓ Kritik çatışma: {(conflicts['conflict_severity'] == 'critical').sum()}")
        else:
            print(f"  ℹ️  Çatışma tespit edilmedi")
        
        # Çatışma istatistikleri
        conflict_stats = self.conflict_detector.get_conflict_statistics()
        pd.DataFrame([conflict_stats]).to_csv(
            f'{output_dir}/conflict_detector_statistics.csv', index=False
        )
    
    def _analyze_interlock_decisions(self, results: List[Dict], output_dir: str):
        """XSM Interlock analizi (TEZ 4.5.3)."""
        print("\n[4/7] XSM Interlock kararları analiz ediliyor (TEZ 4.5.3)...")
        
        df = pd.DataFrame(results)
        interlocked = df[df['interlock_triggered'] == True]
        
        if len(interlocked) > 0:
            # Interlock kararları
            interlock_decisions = interlocked['interlock_decision'].value_counts()
            interlock_decisions.to_csv(f'{output_dir}/interlock_decisions_distribution.csv')
            
            # Risk skorları
            interlock_risk_stats = interlocked['interlock_risk_score'].describe()
            interlock_risk_stats.to_csv(f'{output_dir}/interlock_risk_statistics.csv')
            
            # Final aksiyon dağılımı
            final_actions = interlocked['final_action'].value_counts()
            final_actions.to_csv(f'{output_dir}/final_actions_distribution.csv')
            
            print(f"  ✓ Toplam interlock: {len(interlocked)}")
            print(f"  ✓ ALLOW: {(interlocked['interlock_decision'] == 'allow').sum()}")
            print(f"  ✓ DENY: {(interlocked['interlock_decision'] == 'deny').sum()}")
            print(f"  ✓ ESCALATE: {(interlocked['interlock_decision'] == 'escalate').sum()}")
        else:
            print(f"  ℹ️  Interlock tetiklenmedi")
        
        # Interlock istatistikleri
        interlock_stats = self.xsm_interlock.get_interlock_statistics()
        pd.DataFrame([interlock_stats]).to_csv(
            f'{output_dir}/xsm_interlock_statistics.csv', index=False
        )
    
    def _analyze_xdss_xsm(self, results: List[Dict], output_dir: str):
        """XDSS ve XSM analizleri."""
        print("\n[5/7] XDSS ve XSM analizleri...")
        
        df = pd.DataFrame(results)
        
        # XDSS dağılımı
        xdss_dist = df['xdss_action'].value_counts()
        xdss_dist.to_csv(f'{output_dir}/xdss_distribution.csv')
        
        # XSM status dağılımı
        xsm_dist = df['xsm_status'].value_counts()
        xsm_dist.to_csv(f'{output_dir}/xsm_status_distribution.csv')
        
        print(f"  ✓ XDSS ve XSM raporları kaydedildi")
    
    def _extract_critical_cases_v2(self, results: List[Dict], output_dir: str):
        """Kritik vakaları çıkar (genişletilmiş)."""
        print("\n[6/7] Kritik vakalar çıkarılıyor...")
        
        # Çatışmalı ve DENY edilen vakalar
        critical_cases = [
            r for r in results 
            if r['conflict_flag'] and r['interlock_decision'] == 'deny'
        ]
        
        # ESCALATE vakaları
        escalate_cases = [
            r for r in results 
            if r['interlock_decision'] == 'escalate'
        ]
        
        # Yüksek riskli override vakaları
        high_risk_override = [
            r for r in results 
            if r['operator_override'] and r['y_prob'] > 0.8
        ]
        
        summary = {
            'critical_deny_count': len(critical_cases),
            'escalate_count': len(escalate_cases),
            'high_risk_override_count': len(high_risk_override)
        }
        
        pd.DataFrame([summary]).to_csv(
            f'{output_dir}/critical_cases_summary_v2.csv', index=False
        )
        
        # Detaylı listeler
        if critical_cases:
            pd.DataFrame(critical_cases).to_csv(
                f'{output_dir}/critical_deny_cases.csv', index=False
            )
        
        if escalate_cases:
            pd.DataFrame(escalate_cases).to_csv(
                f'{output_dir}/escalate_cases.csv', index=False
            )
        
        print(f"  ✓ Critical DENY: {len(critical_cases)}")
        print(f"  ✓ ESCALATE: {len(escalate_cases)}")
        print(f"  ✓ High-risk Override: {len(high_risk_override)}")
    
    def _create_visualizations_v2(self, results: List[Dict], output_dir: str):
        """Görselleştirmeler (genişletilmiş)."""
        print("\n[7/7] Görselleştirmeler oluşturuluyor...")
        
        df = pd.DataFrame(results)
        
        # 1. Ana kontrol akışı
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # XDSS
        xdss_counts = df['xdss_action'].value_counts()
        axes[0, 0].bar(xdss_counts.index, xdss_counts.values, color=['#d32f2f', '#ff9800', '#4caf50'])
        axes[0, 0].set_title('XDSS Önerileri', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Sayı')
        
        # Operatör
        operator_counts = df['operator_action'].value_counts()
        axes[0, 1].bar(operator_counts.index, operator_counts.values, color='steelblue')
        axes[0, 1].set_title('Operatör Kararları (TEZ 4.5.1)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Sayı')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Çatışma   
        n_conflicts = df['conflict_flag'].sum()         # True sayısı (Çatışma)
        n_compliant = len(df) - n_conflicts             # False sayısı (Uyum)
        
        # 2. Dinamik veri listeleri oluştur
        pie_values = []
        pie_labels = []
        pie_colors = []
        
        # Uyum varsa listeye ekle
        if n_compliant > 0:
            pie_values.append(n_compliant)
            pie_labels.append('Uyum')
            pie_colors.append('#4caf50') # Yeşil
            
        # Çatışma varsa listeye ekle
        if n_conflicts > 0:
            pie_values.append(n_conflicts)
            pie_labels.append('Çatışma')
            pie_colors.append('#ff9800') # Turuncu
            
        # 3. Grafiği çiz (Eğer veri varsa)
        if pie_values:
            axes[1, 0].pie(pie_values, labels=pie_labels, 
                          autopct='%1.1f%%', colors=pie_colors)
        else:
            axes[1, 0].text(0.5, 0.5, "Veri Yok", ha='center')
            
        axes[1, 0].set_title('Çatışma Oranı (TEZ 4.5.2)', fontsize=14, fontweight='bold')
        
        # Interlock
        interlock_counts = df['interlock_decision'].value_counts()
        axes[1, 1].bar(interlock_counts.index, interlock_counts.values, 
                      color=['#4caf50', '#d32f2f', '#ff9800'])
        axes[1, 1].set_title('XSM Interlock Kararları (TEZ 4.5.3)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Sayı')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/control_flow_overview_v2.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Risk skorları
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Çatışma risk skorları
        conflicts = df[df['conflict_flag'] == True]
        if len(conflicts) > 0:
            axes[0].hist(conflicts['conflict_risk_score'], bins=20, color='coral', edgecolor='black')
            axes[0].set_title('Çatışma Risk Skorları', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Risk Skoru')
            axes[0].set_ylabel('Frekans')
        
        # Interlock risk skorları
        interlocked = df[df['interlock_triggered'] == True]
        if len(interlocked) > 0:
            axes[1].hist(interlocked['interlock_risk_score'], bins=20, color='skyblue', edgecolor='black')
            axes[1].set_title('XSM Interlock Risk Skorları', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Risk Skoru')
            axes[1].set_ylabel('Frekans')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_scores_distribution.png', dpi=300, bbox_inches='tight')
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
        print(f"X-HARMONY V2 DETAYLI RAPOR - ÖRNEK #{sample_idx}")
        print("TEZ 4.5: Human-in-the-Loop + Safety Interlock")
        print("=" * 70)
        
        print(f"\n📊 GERÇEK DURUM ve MODEL TAHMİNİ:")
        print(f"  Gerçek: {'FAIL' if result['y_true'] == 1 else 'PASS'}")
        print(f"  Tahmin: {'FAIL' if result['y_pred'] == 1 else 'PASS'}")
        print(f"  Olasılık: {result['y_prob']:.4f}")
        
        print(f"\n🤖 XDSS ÖNERİSİ:")
        print(f"  Aksiyon: {result['xdss_action']}")
        print(f"  Güven: {result['xdss_confidence']:.2f}")
        
        print(f"\n👨‍🔧 OPERATÖR KARARI (TEZ 4.5.1):")
        print(f"  Aksiyon: {result['operator_action']}")
        print(f"  Güven: {result['operator_confidence']:.2f}")
        print(f"  Override: {'EVET ⚠️' if result['operator_override'] else 'HAYIR ✓'}")
        print(f"  Gerekçe: {result['operator_reasoning']}")
        
        print(f"\n⚠️  ÇATIŞMA DURUMU (TEZ 4.5.2):")
        if result['conflict_flag']:
            print(f"  Çatışma: EVET")
            print(f"  Tip: {result['conflict_type']}")
            print(f"  Şiddet: {result['conflict_severity']}")
            print(f"  Risk Skoru: {result['conflict_risk_score']:.3f}")
        else:
            print(f"  Çatışma: HAYIR (Uyumlu)")
        
        print(f"\n🔒 XSM INTERLOCK KARARI (TEZ 4.5.3):")
        if result['interlock_triggered']:
            print(f"  Karar: {result['interlock_decision'].upper()}")
            print(f"  Risk Skoru: {result['interlock_risk_score']:.3f}")
            print(f"  Final Aksiyon: {result['final_action']}")
        else:
            print(f"  Interlock: Tetiklenmedi")
        
        print(f"\n🛡️  XSM GÜVENLİK:")
        print(f"  Status: {result['xsm_status']}")
        
        print("\n" + "=" * 70)


# =============================================================================
# MAIN - COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='X-HARMONY V2 Simülasyonu (Human-in-the-Loop)'
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
        default='./xharmony_v2_outputs',
        help='Çıktı klasörü yolu'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=100,
        help='Simüle edilecek örnek sayısı'
    )
    
    parser.add_argument(
        '--operator_profile',
        type=str,
        default='CAUTIOUS_MID',
        choices=['EXPERT_COMPLIANT', 'CAUTIOUS_MID', 'OPTIMISTIC_SENIOR', 'VETERAN', 'NOVICE', 'RANDOM_TEST'],
        help='Operatör profil tipi'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo mode'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        print("\n🎮 DEMO MODE: Modül demoları çalıştırılıyor...\n")
        
        from operator_decision_module import demo_operator_module
        from conflict_detector import demo_conflict_detector
        from xsm_interlock import demo_xsm_interlock
        
        demo_operator_module()
        print("\n" + "="*70 + "\n")
        demo_conflict_detector()
        print("\n" + "="*70 + "\n")
        demo_xsm_interlock()
        
        print("\n✓ Demo tamamlandı!")
        return
    
    # Normal simülasyon modu
    data_dir = args.data_dir
    output_dir = args.output_dir
    n_samples = args.n_samples
    operator_profile = args.operator_profile
    
    # Dosya yolları
    model_path = f'{data_dir}/xai_final_model.pkl'
    test_data_path = f'{data_dir}/xai_test_X.csv'
    test_labels_path = f'{data_dir}/xai_test_y.csv'
    shap_values_path = f'{data_dir}/global_shap_values.npy'
    rules_path = f'{data_dir}/xdss_xsm_rules.csv'
    feature_names_path = f'{data_dir}/xai_feature_names.txt'
    
    # Simülatörü başlat
    simulator = XHARMONYSimulatorV2(
        model_path=model_path,
        test_data_path=test_data_path,
        test_labels_path=test_labels_path,
        shap_values_path=shap_values_path,
        rules_path=rules_path,
        feature_names_path=feature_names_path,
        operator_profile_name=operator_profile,
        verbose=True
    )
    
    # Simülasyonu çalıştır
    results = simulator.run_batch_simulation(n_samples=n_samples)
    
    # Kapsamlı rapor üret
    simulator.generate_comprehensive_report(results, output_dir)
    
    # Birkaç örnek için detaylı rapor
    print("\n\n" + "=" * 70)
    print("ÖRNEK DETAYLI RAPORLAR")
    print("=" * 70)
    
    for i in range(min(3, len(results))):
        simulator.print_sample_report(i)
    
    print("\n\n✅ X-HARMONY V2 Simülasyonu başarıyla tamamlandı!")
    print(f"📁 Tüm sonuçlar: {output_dir}")
    print(f"\nOperatör Profil: {operator_profile}")


if __name__ == "__main__":
    main()