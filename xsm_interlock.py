"""
================================================================================
XSM INTERLOCK (OVERRIDE GATE)
X-HARMONY Mimarisi - GÜVENLİK KİLİDİ MEKANİZMASI
================================================================================

TEZ BAŞLIĞI: 4.5.3 - XSM Safety Interlock (Güvenlik Kilidi Mekanizması)

Bu modül, Operatör-XDSS çatışması tespit edildiğinde devreye girer ve
çok boyutlu risk analizi yaparak üç seviyeli güvenlik kararı verir.

GÜVENLİK Kararları (Interlock Decisions):
    - ALLOW: Operatör kararına izin ver (risk kabul edilebilir)
    - DENY: Operatör engellenir, XDSS önerisi zorlanır (yüksek risk)
    - ESCALATE: İkinci kontrol gerekli (supervisor/manual review)

Risk Analiz Bileşenleri:
    1. Model Güvenilirlik Skoru (Model Confidence)
       - Tahmin olasılığı
       - Model kalibrasyon durumu
    
    2. XAI Belirsizlik Skoru (XAI Uncertainty)
       - SHAP değer varyansı
       - Açıklama tutarlılığı
       - Feature importance entropi
    
    3. Operatör Performans Skoru (Operator Track Record)
       - Geçmiş başarı oranı
       - Deneyim seviyesi
       - Benzer senaryolarda performans
    
    4. Sistem Drift/Anomali Skoru
       - Data distribution shift
       - Sensör anomalileri
       - Model drift göstergeleri

Final Risk Score: Ağırlıklı kombinasyon (0-1)
Decision Logic: Risk skoruna göre threshold-based karar

Yazar: X-HARMONY Implementation - Thesis Chapter 4.5.3
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
from scipy import stats as scipy_stats


class InterlockDecision(Enum):
    """XSM Interlock kararları."""
    ALLOW = "allow"           # Operatör kararına izin
    DENY = "deny"             # Operatör engellendi
    ESCALATE = "escalate"     # Yükselt (supervisor review)


class XSMInterlock:
    """
    XSM Safety Interlock - Güvenlik Kilidi Mekanizması.
    
    TEZ 4.5.3: Bu modül çatışma durumunda çok boyutlu risk analizi
    yaparak operatör kararının yürütülmesine izin verir veya engeller.
    """
    
    def __init__(
        self,
        risk_thresholds: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        XSM Interlock'u başlat.
        
        Args:
            risk_thresholds: Risk eşikleri (opsiyonel, varsayılan kullanılır)
            verbose: Detaylı çıktı
        """
        self.verbose = verbose
        
        # Risk eşikleri
        self.thresholds = risk_thresholds or self._default_thresholds()
        
        # Karar geçmişi
        self.interlock_history = []
        
        # Ağırlıklar (risk bileşenleri)
        self.risk_weights = {
            'model_confidence': 0.30,
            'xai_uncertainty': 0.25,
            'operator_performance': 0.25,
            'system_drift': 0.20
        }
        
        if self.verbose:
            print("=" * 70)
            print("XSM INTERLOCK (GÜVENLİK KİLİDİ) BAÅžLATILDI (TEZ 4.5.3)")
            print("=" * 70)
            print("✓ Çok boyutlu risk analiz motoru aktif")
            print(f"✓ Risk ağırlıkları: {self.risk_weights}")
            print(f"✓ Karar eşikleri: ALLOW<{self.thresholds['allow_max']:.2f}, "
                  f"DENY>{self.thresholds['deny_min']:.2f}")
            print("=" * 70)
    
    def _default_thresholds(self) -> Dict[str, float]:
        """
        Varsayılan risk eşiklerini ayarla.
        
        TEZ: Bu eşikler güvenlik-verimlilik dengesini belirler.
        Daha düşük allow_max = daha muhafazakar sistem
        """
        return {
            'allow_max': 0.35,      # Risk < 0.35 → ALLOW
            'deny_min': 0.65,       # Risk > 0.65 → DENY
            'escalate_range': (0.35, 0.65),  # 0.35-0.65 arası → ESCALATE
            
            # Alt bileşen eşikleri
            'model_conf_critical': 0.3,    # Model güveni < 0.3 → kritik
            'xai_uncertainty_high': 0.7,    # XAI belirsizliği > 0.7 → yüksek
            'operator_acc_low': 0.7,        # Operatör başarı < 0.7 → düşük
            'drift_critical': 0.8           # Drift > 0.8 → kritik
        }
    
    def interlock_decision(
        self,
        conflict_report: Dict,
        model_prob: float,
        xdss_confidence: float,
        operator_profile: Dict,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        xsm_anomaly_report: Optional[Dict] = None
    ) -> Dict:
        """
        XSM Interlock kararını ver.
        
        TEZ 4.5.3: Bu fonksiyon çatışma durumunda risk analizi yaparak
        operatör kararının yürütülmesine izin verir veya engeller.
        
        Args:
            conflict_report: Conflict Detector'dan gelen rapor
            model_prob: Model fail olasılığı (0-1)
            xdss_confidence: XDSS güven skoru
            operator_profile: Operatör profil bilgileri
            shap_values: SHAP açıklama değerleri
            feature_values: Sensör değerleri
            xsm_anomaly_report: XSM anomali raporu (opsiyonel)
            
        Returns:
            interlock_report: {
                'decision': InterlockDecision,
                'risk_score': float,
                'risk_breakdown': Dict,
                'reasoning': str,
                'final_action': str,
                'confidence': float
            }
        """
        
        # Çatışma yoksa interlock gerekmez
        if not conflict_report['conflict_flag']:
            return {
                'decision': InterlockDecision.ALLOW,
                'risk_score': 0.0,
                'risk_breakdown': {},
                'reasoning': "Çatışma tespit edilmedi. Interlock gerekli değil.",
                'final_action': conflict_report['operator_action'],
                'confidence': 1.0,
                'interlock_triggered': False
            }
        
        # 1. Çok boyutlu risk analizi
        risk_breakdown = self._comprehensive_risk_analysis(
            model_prob=model_prob,
            xdss_confidence=xdss_confidence,
            operator_profile=operator_profile,
            shap_values=shap_values,
            feature_values=feature_values,
            xsm_anomaly_report=xsm_anomaly_report,
            conflict_severity=conflict_report['severity']
        )
        
        # 2. Final risk skorunu hesapla
        final_risk_score = self._calculate_final_risk_score(risk_breakdown)
        
        # 3. Karar ver (ALLOW / DENY / ESCALATE)
        interlock_decision, decision_confidence = self._make_interlock_decision(
            final_risk_score, risk_breakdown
        )
        
        # 4. Final aksiyonu belirle
        final_action = self._determine_final_action(
            interlock_decision,
            conflict_report['operator_action'],
            conflict_report['xdss_action']
        )
        
        # 5. Gerekçe üret
        reasoning = self._generate_reasoning(
            interlock_decision, final_risk_score,
            risk_breakdown, conflict_report
        )
        
        # Rapor oluştur
        interlock_report = {
            'decision': interlock_decision,
            'risk_score': final_risk_score,
            'risk_breakdown': risk_breakdown,
            'reasoning': reasoning,
            'final_action': final_action,
            'confidence': decision_confidence,
            'interlock_triggered': True,
            'conflict_summary': {
                'xdss_action': conflict_report['xdss_action'],
                'operator_action': conflict_report['operator_action'],
                'conflict_severity': conflict_report['severity'].value
            }
        }
        
        # GeçmiÅŸe kaydet
        self.interlock_history.append(interlock_report)
        
        return interlock_report
    
    def _comprehensive_risk_analysis(
        self,
        model_prob: float,
        xdss_confidence: float,
        operator_profile: Dict,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        xsm_anomaly_report: Optional[Dict],
        conflict_severity: Enum
    ) -> Dict:
        """
        Çok boyutlu risk analizi.
        
        TEZ: Bu fonksiyon 4 ana risk bileşenini hesaplar:
        1. Model Güvenilirliği
        2. XAI Belirsizliği
        3. Operatör Performansı
        4. Sistem Drift/Anomali
        """
        
        risk_breakdown = {}
        
        # 1. MODEL GÜVENİLİRLİK SKORU
        risk_breakdown['model_confidence'] = self._assess_model_reliability(
            model_prob, xdss_confidence
        )
        
        # 2. XAI BELİRSİZLİK SKORU
        risk_breakdown['xai_uncertainty'] = self._assess_xai_uncertainty(
            shap_values, feature_values
        )
        
        # 3. OPERATÖR PERFORMANS SKORU
        risk_breakdown['operator_performance'] = self._assess_operator_performance(
            operator_profile
        )
        
        # 4. SİSTEM DRIFT/ANOMALİ SKORU
        risk_breakdown['system_drift'] = self._assess_system_drift(
            xsm_anomaly_report, feature_values
        )
        
        # 5. ÇATIŞMA ŞİDDETİ ETKİSİ
        risk_breakdown['conflict_severity_impact'] = self._assess_conflict_impact(
            conflict_severity
        )
        
        return risk_breakdown
    
    def _assess_model_reliability(
        self, 
        model_prob: float,
        xdss_confidence: float
    ) -> Dict:
        """
        Model güvenilirlik riskini değerlendir.
        
        TEZ: Yüksek model güveni + yüksek XDSS güveni = düşük risk
        """
        
        # Model kesinliği (0.5'ten uzaklık)
        model_certainty = abs(model_prob - 0.5) * 2  # 0-1 normalize
        
        # XDSS güveni
        xdss_certainty = xdss_confidence
        
        # Birleşik güvenilirlik (yüksek = güvenilir = düşük risk)
        reliability = (model_certainty + xdss_certainty) / 2
        
        # Risk skoru (1 - güvenilirlik)
        risk_score = 1 - reliability
        
        return {
            'risk_score': risk_score,
            'model_certainty': model_certainty,
            'xdss_certainty': xdss_certainty,
            'overall_reliability': reliability,
            'interpretation': (
                "Yüksek güvenilirlik" if reliability > 0.7 else
                "Orta güvenilirlik" if reliability > 0.4 else
                "Düşük güvenilirlik"
            )
        }
    
    def _assess_xai_uncertainty(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray
    ) -> Dict:
        """
        XAI belirsizlik riskini değerlendir.
        
        TEZ: SHAP değerlerinin tutarlılığı ve dağılımı.
        Yüksek belirsizlik = açıklama güvenilmez = yüksek risk
        """
        
        # 1. SHAP varyansı (normalize edilmiş)
        shap_variance = np.var(shap_values) if len(shap_values) > 0 else 0
        shap_variance_norm = min(shap_variance / 0.1, 1.0)  # 0.1'e normalize
        
        # 2. SHAP entropy (dağılım belirsizliği)
        # Pozitif SHAP değerlerinin dağılımı
        pos_shap = shap_values[shap_values > 0]
        if len(pos_shap) > 0:
            # Normalize et
            pos_shap_norm = pos_shap / (np.sum(np.abs(pos_shap)) + 1e-10)
            entropy = scipy_stats.entropy(pos_shap_norm + 1e-10)
            entropy_norm = min(entropy / 3.0, 1.0)  # 3.0'a normalize
        else:
            entropy_norm = 0.5  # Orta belirsizlik
        
        # 3. SHAP sparsity (çok az feature etkili mi?)
        sparsity = np.sum(np.abs(shap_values) < 0.01) / len(shap_values)
        
        # 4. Top SHAP dominance (tek bir feature çok baskın mı?)
        if len(shap_values) > 0:
            top_shap = np.max(np.abs(shap_values))
            total_shap = np.sum(np.abs(shap_values))
            dominance = top_shap / (total_shap + 1e-10) if total_shap > 0 else 0
        else:
            dominance = 0
        
        # Birleşik belirsizlik skoru
        uncertainty_score = (
            shap_variance_norm * 0.3 +
            entropy_norm * 0.3 +
            sparsity * 0.2 +
            dominance * 0.2
        )
        
        return {
            'risk_score': uncertainty_score,
            'shap_variance': float(shap_variance),
            'entropy': float(entropy_norm),
            'sparsity': float(sparsity),
            'dominance': float(dominance),
            'interpretation': (
                "Yüksek belirsizlik" if uncertainty_score > 0.7 else
                "Orta belirsizlik" if uncertainty_score > 0.4 else
                "Düşük belirsizlik"
            )
        }
    
    def _assess_operator_performance(
        self,
        operator_profile: Dict
    ) -> Dict:
        """
        Operatör performans riskini değerlendir.
        
        TEZ: Yüksek deneyim + yüksek başarı oranı = düşük risk
        """
        
        # Operatör özellikleri
        experience_years = operator_profile.get('experience_years', 3.0)
        historical_accuracy = operator_profile.get('historical_accuracy', 0.75)
        
        # Deneyim skoru (0-1, 10+ yıl = 1.0)
        experience_score = min(experience_years / 10.0, 1.0)
        
        # Başarı skoru (0-1)
        accuracy_score = historical_accuracy
        
        # Birleşik performans (yüksek = iyi = düşük risk)
        performance = (experience_score + accuracy_score) / 2
        
        # Risk skoru (1 - performans)
        risk_score = 1 - performance
        
        return {
            'risk_score': risk_score,
            'experience_score': experience_score,
            'accuracy_score': accuracy_score,
            'overall_performance': performance,
            'interpretation': (
                "Yüksek performans" if performance > 0.75 else
                "Orta performans" if performance > 0.5 else
                "Düşük performans"
            )
        }
    
    def _assess_system_drift(
        self,
        xsm_anomaly_report: Optional[Dict],
        feature_values: np.ndarray
    ) -> Dict:
        """
        Sistem drift/anomali riskini değerlendir.
        
        TEZ: Veri dağılımı kayması ve anomaliler.
        Yüksek drift = sistem güvenilmez = yüksek risk
        """
        
        # XSM raporu varsa kullan
        if xsm_anomaly_report:
            anomaly_count = len(xsm_anomaly_report.get('anomalies', []))
            status = xsm_anomaly_report.get('status', 'NORMAL')
            
            # Status'e göre risk
            status_risk = {
                'NORMAL': 0.1,
                'WARNING': 0.4,
                'INFO': 0.2,
                'CRITICAL': 0.9
            }.get(status, 0.3)
            
            # Anomali sayısına göre risk
            anomaly_risk = min(anomaly_count * 0.15, 0.8)
            
            drift_score = (status_risk + anomaly_risk) / 2
        
        else:
            # Basit outlier tespiti
            extreme_values = np.sum(np.abs(feature_values) > 3.0)
            drift_score = min(extreme_values * 0.1, 0.6)
        
        return {
            'risk_score': drift_score,
            'interpretation': (
                "Yüksek drift" if drift_score > 0.7 else
                "Orta drift" if drift_score > 0.4 else
                "Düşük drift"
            )
        }
    
    def _assess_conflict_impact(self, conflict_severity: Enum) -> Dict:
        """Çatışma şiddetinin risk etkisi."""
        
        severity_scores = {
            'none': 0.0,
            'low': 0.2,
            'moderate': 0.5,
            'critical': 0.9
        }
        
        severity_value = conflict_severity.value if hasattr(conflict_severity, 'value') else 'moderate'
        risk_score = severity_scores.get(severity_value, 0.5)
        
        return {
            'risk_score': risk_score,
            'severity': severity_value
        }
    
    def _calculate_final_risk_score(self, risk_breakdown: Dict) -> float:
        """
        Final risk skorunu hesapla (ağırlıklı kombinasyon).
        
        TEZ: Tüm risk bileşenlerini ağırlıklı olarak birleştir.
        """
        
        final_score = (
            risk_breakdown['model_confidence']['risk_score'] * self.risk_weights['model_confidence'] +
            risk_breakdown['xai_uncertainty']['risk_score'] * self.risk_weights['xai_uncertainty'] +
            risk_breakdown['operator_performance']['risk_score'] * self.risk_weights['operator_performance'] +
            risk_breakdown['system_drift']['risk_score'] * self.risk_weights['system_drift']
        )
        
        # Çatışma şiddeti bonus
        conflict_impact = risk_breakdown['conflict_severity_impact']['risk_score']
        final_score = min(final_score + conflict_impact * 0.15, 1.0)
        
        return final_score
    
    def _make_interlock_decision(
        self,
        risk_score: float,
        risk_breakdown: Dict
    ) -> Tuple[InterlockDecision, float]:
        """
        Risk skoruna göre interlock kararı ver.
        
        TEZ: Threshold-based karar mantığı.
        """
        
        # ALLOW: Düşük risk → Operatör kararına izin
        if risk_score < self.thresholds['allow_max']:
            decision = InterlockDecision.ALLOW
            confidence = 1 - risk_score  # Düşük risk = yüksek güven
        
        # DENY: Yüksek risk → Operatör engellendi
        elif risk_score > self.thresholds['deny_min']:
            decision = InterlockDecision.DENY
            confidence = risk_score  # Yüksek risk = DENY'a yüksek güven
        
        # ESCALATE: Orta risk → İkinci kontrol
        else:
            decision = InterlockDecision.ESCALATE
            # Orta bölgede güven daha düşük
            confidence = 0.6
        
        return decision, confidence
    
    def _determine_final_action(
        self,
        decision: InterlockDecision,
        operator_action: str,
        xdss_action: str
    ) -> str:
        """
        Final aksiyonu belirle.
        
        TEZ: Interlock kararına göre hangi aksiyon yürütülecek.
        """
        
        if decision == InterlockDecision.ALLOW:
            # Operatör kararı yürütülür
            return operator_action
        
        elif decision == InterlockDecision.DENY:
            # XDSS önerisi zorlanır
            return xdss_action
        
        else:  # ESCALATE
            # En güvenli seçenek (genelde XDSS)
            # veya özel bir escalation aksiyonu
            return "ESCALATE_TO_SUPERVISOR"
    
    def _generate_reasoning(
        self,
        decision: InterlockDecision,
        risk_score: float,
        risk_breakdown: Dict,
        conflict_report: Dict
    ) -> str:
        """
        Karar gerekçesini üret.
        
        TEZ: Şeffaf ve açıklanabilir karar mantığı.
        """
        
        reasoning = f"XSM Interlock Kararı: {decision.value.upper()}\n"
        reasoning += f"Final Risk Skoru: {risk_score:.3f}\n\n"
        
        reasoning += "Risk Analizi:\n"
        reasoning += f"  1. Model Güvenilirliği: {risk_breakdown['model_confidence']['risk_score']:.3f} "
        reasoning += f"({risk_breakdown['model_confidence']['interpretation']})\n"
        
        reasoning += f"  2. XAI Belirsizliği: {risk_breakdown['xai_uncertainty']['risk_score']:.3f} "
        reasoning += f"({risk_breakdown['xai_uncertainty']['interpretation']})\n"
        
        reasoning += f"  3. Operatör Performansı: {risk_breakdown['operator_performance']['risk_score']:.3f} "
        reasoning += f"({risk_breakdown['operator_performance']['interpretation']})\n"
        
        reasoning += f"  4. Sistem Drift: {risk_breakdown['system_drift']['risk_score']:.3f} "
        reasoning += f"({risk_breakdown['system_drift']['interpretation']})\n"
        
        reasoning += f"\nKarar Mantığı:\n"
        
        if decision == InterlockDecision.ALLOW:
            reasoning += f"  • Risk kabul edilebilir seviyede ({risk_score:.3f} < {self.thresholds['allow_max']})\n"
            reasoning += f"  • Operatör kararı ({conflict_report['operator_action']}) yürütülür\n"
        
        elif decision == InterlockDecision.DENY:
            reasoning += f"  • Risk çok yüksek ({risk_score:.3f} > {self.thresholds['deny_min']})\n"
            reasoning += f"  • Operatör kararı ENGELLENDİ ✘\n"
            reasoning += f"  • XDSS önerisi ({conflict_report['xdss_action']}) ZORLANDI\n"
        
        else:  # ESCALATE
            reasoning += f"  • Risk orta seviyede (belirsizlik bölgesi)\n"
            reasoning += f"  • Supervisor/manual review gerekli\n"
            reasoning += f"  • Üretim güvenli moda alındı\n"
        
        return reasoning
    
    def get_interlock_statistics(self) -> Dict:
        """
        Interlock istatistiklerini hesapla.
        
        TEZ: Sistemin güvenlik performansını değerlendir.
        """
        
        if not self.interlock_history:
            return {
                'total_interlocks': 0,
                'allow_rate': 0.0,
                'deny_rate': 0.0,
                'escalate_rate': 0.0,
                'avg_risk_score': 0.0
            }
        
        total = len(self.interlock_history)
        
        allows = sum(1 for h in self.interlock_history if h['decision'] == InterlockDecision.ALLOW)
        denies = sum(1 for h in self.interlock_history if h['decision'] == InterlockDecision.DENY)
        escalates = sum(1 for h in self.interlock_history if h['decision'] == InterlockDecision.ESCALATE)
        
        avg_risk = np.mean([h['risk_score'] for h in self.interlock_history])
        
        return {
            'total_interlocks': total,
            'allow_count': allows,
            'deny_count': denies,
            'escalate_count': escalates,
            'allow_rate': allows / total,
            'deny_rate': denies / total,
            'escalate_rate': escalates / total,
            'avg_risk_score': avg_risk
        }
    
    def format_interlock_report(self, report: Dict) -> str:
        """Interlock raporunu formatla."""
        
        output = "\n" + "=" * 70
        output += "\n🔒 XSM INTERLOCK RAPORU (TEZ 4.5.3)"
        output += "\n" + "=" * 70
        
        if not report['interlock_triggered']:
            output += "\n✓ Interlock tetiklenmedi (çatışma yok)"
            return output + "\n" + "=" * 70
        
        decision = report['decision']
        
        if decision == InterlockDecision.ALLOW:
            icon = "✓"
            status = "İZİN VERİLDİ"
        elif decision == InterlockDecision.DENY:
            icon = "✘"
            status = "ENGELLENDİ"
        else:
            icon = "⚠"
            status = "YÜKSELTİLDİ"
        
        output += f"\n\n{icon} KARAR: {status}"
        output += f"\n  Risk Skoru: {report['risk_score']:.3f}"
        output += f"\n  Güven: {report['confidence']:.2f}"
        output += f"\n\n  XDSS → {report['conflict_summary']['xdss_action']}"
        output += f"\n  Operatör → {report['conflict_summary']['operator_action']}"
        output += f"\n  Final Aksiyon → {report['final_action']}"
        
        output += f"\n\n📊 Risk Bileşenleri:"
        rb = report['risk_breakdown']
        output += f"\n  Model Güvenilirliği: {rb['model_confidence']['risk_score']:.3f}"
        output += f"\n  XAI Belirsizliği: {rb['xai_uncertainty']['risk_score']:.3f}"
        output += f"\n  Operatör Performansı: {rb['operator_performance']['risk_score']:.3f}"
        output += f"\n  Sistem Drift: {rb['system_drift']['risk_score']:.3f}"
        
        output += "\n" + "=" * 70
        
        return output


# =============================================================================
# DEMO FONKSİYONU
# =============================================================================

def demo_xsm_interlock():
    """XSM Interlock demo."""
    
    print("\n" + "🎮 " + "=" * 66)
    print("XSM INTERLOCK (GÜVENLİK KİLİDİ) DEMO - TEZ 4.5.3")
    print("=" * 70)
    
    from conflict_detector import ConflictDetector, ConflictSeverity
    
    interlock = XSMInterlock(verbose=False)
    detector = ConflictDetector(verbose=False)
    
    # Test senaryoları
    scenarios = [
        {
            "name": "DÜŞÜK RİSK - ALLOW",
            "xdss": "CHECK",
            "operator": "CONTINUE",
            "xdss_conf": 0.65,
            "operator_conf": 0.80,
            "model_prob": 0.45,
            "operator_profile": {
                'experience_years': 8.0,
                'historical_accuracy': 0.92
            },
            "shap_std": 0.05
        },
        {
            "name": "YÜKSEK RİSK - DENY",
            "xdss": "STOP",
            "operator": "CONTINUE",
            "xdss_conf": 0.90,
            "operator_conf": 0.60,
            "model_prob": 0.88,
            "operator_profile": {
                'experience_years': 2.0,
                'historical_accuracy': 0.70
            },
            "shap_std": 0.20
        },
        {
            "name": "ORTA RİSK - ESCALATE",
            "xdss": "CHECK",
            "operator": "STOP",
            "xdss_conf": 0.60,
            "operator_conf": 0.55,
            "model_prob": 0.62,
            "operator_profile": {
                'experience_years': 5.0,
                'historical_accuracy': 0.82
            },
            "shap_std": 0.12
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SENARYO {i}: {scenario['name']}")
        print(f"{'='*70}")
        
        # Çatışma tespit
        conflict = detector.detect_conflict(
            xdss_action=scenario['xdss'],
            operator_action=scenario['operator'],
            xdss_confidence=scenario['xdss_conf'],
            operator_confidence=scenario['operator_conf'],
            model_prob=scenario['model_prob']
        )
        
        # Dummy SHAP ve features
        shap_values = np.random.randn(50) * scenario['shap_std']
        feature_values = np.random.randn(50)
        
        # Interlock kararı
        interlock_report = interlock.interlock_decision(
            conflict_report=conflict,
            model_prob=scenario['model_prob'],
            xdss_confidence=scenario['xdss_conf'],
            operator_profile=scenario['operator_profile'],
            shap_values=shap_values,
            feature_values=feature_values,
            xsm_anomaly_report=None
        )
        
        print(interlock.format_interlock_report(interlock_report))
    
    # İstatistikler
    stats = interlock.get_interlock_statistics()
    print(f"\n{'='*70}")
    print("GENEL İSTATİSTİKLER")
    print(f"{'='*70}")
    print(f"Toplam Interlock: {stats['total_interlocks']}")
    print(f"ALLOW Oranı: {stats['allow_rate']:.1%}")
    print(f"DENY Oranı: {stats['deny_rate']:.1%}")
    print(f"ESCALATE Oranı: {stats['escalate_rate']:.1%}")
    print(f"Ortalama Risk: {stats['avg_risk_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("✔ Demo tamamlandı!")


if __name__ == "__main__":
    demo_xsm_interlock()