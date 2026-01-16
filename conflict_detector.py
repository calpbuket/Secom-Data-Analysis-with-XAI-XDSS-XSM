"""
================================================================================
CONFLICT DETECTOR
X-HARMONY Mimarisi - Çatışma Tespit Mekanizması
================================================================================

TEZ BAŞLIĞI: 4.5.2 - Çatışma Tespit Mekanizması (Conflict Detection Mechanism)

Bu modül, XDSS önerisi ile Operatör kararı arasındaki uyumsuzlukları tespit eder.
Çatışma tespiti, XSM Interlock'un devreye girmesi için kritik tetikleyicidir.

Çatışma Tipleri:
    - FULL_OVERRIDE: Tam ters karar (STOP → CONTINUE)
    - PARTIAL_MISMATCH: Kısmi uyumsuzluk (CHECK → CONTINUE)
    - NO_CONFLICT: Uyumlu karar

Çatışma Şiddeti (Severity):
    - CRITICAL: Yüksek riskli uyumsuzluk (örn: STOP ignore edildi)
    - MODERATE: Orta düzey uyumsuzluk
    - LOW: Düşük önem

Çıkışlar:
    - conflict_flag: Boolean (True/False)
    - conflict_type: Çatışma tipi
    - severity: Şiddet seviyesi
    - explanation: Açıklama metni
    - risk_factors: İlişkili risk faktörleri

Yazar: X-HARMONY Implementation - Thesis Chapter 4.5.2
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Literal
from enum import Enum


class ConflictType(Enum):
    """Çatışma tipleri."""
    NO_CONFLICT = "no_conflict"           # Uyum var
    PARTIAL_MISMATCH = "partial_mismatch" # Kısmi uyumsuzluk
    FULL_OVERRIDE = "full_override"       # Tam ters karar


class ConflictSeverity(Enum):
    """Çatışma şiddeti seviyeleri."""
    NONE = "none"           # Çatışma yok
    LOW = "low"             # Düşük önem
    MODERATE = "moderate"   # Orta önem
    CRITICAL = "critical"   # Kritik önem


class ConflictDetector:
    """
    Çatışma Tespit Mekanizması.
    
    TEZ 4.5.2: Bu sınıf XDSS-Operatör uyumsuzluklarını tespit eder
    ve XSM Interlock için çatışma raporu üretir.
    """
    
    # Aksiyon hiyerarşisi (risk azalan sırada)
    ACTION_HIERARCHY = {
        "STOP": 5,           # En riskli/kısıtlayıcı
        "INCREASE_QC": 4,
        "CHECK": 3,
        "ADJUST_PARAMS": 2,
        "CONTINUE": 1        # En az kısıtlayıcı
    }
    
    # Uyumsuzluk matrisi: (XDSS, Operatör) → şiddet
    CONFLICT_MATRIX = {
        # XDSS STOP dediğinde
        ("STOP", "STOP"): ConflictSeverity.NONE,
        ("STOP", "CHECK"): ConflictSeverity.MODERATE,
        ("STOP", "INCREASE_QC"): ConflictSeverity.MODERATE,
        ("STOP", "ADJUST_PARAMS"): ConflictSeverity.CRITICAL,
        ("STOP", "CONTINUE"): ConflictSeverity.CRITICAL,
        
        # XDSS CHECK dediğinde
        ("CHECK", "STOP"): ConflictSeverity.LOW,
        ("CHECK", "CHECK"): ConflictSeverity.NONE,
        ("CHECK", "INCREASE_QC"): ConflictSeverity.NONE,
        ("CHECK", "ADJUST_PARAMS"): ConflictSeverity.LOW,
        ("CHECK", "CONTINUE"): ConflictSeverity.MODERATE,
        
        # XDSS CONTINUE dediğinde
        ("CONTINUE", "STOP"): ConflictSeverity.LOW,
        ("CONTINUE", "CHECK"): ConflictSeverity.LOW,
        ("CONTINUE", "INCREASE_QC"): ConflictSeverity.LOW,
        ("CONTINUE", "ADJUST_PARAMS"): ConflictSeverity.NONE,
        ("CONTINUE", "CONTINUE"): ConflictSeverity.NONE,
    }
    
    def __init__(self, verbose: bool = True):
        """
        Çatışma tespit modülünü başlat.
        
        Args:
            verbose: Detaylı çıktı
        """
        self.verbose = verbose
        self.conflict_history = []
        
        if self.verbose:
            print("=" * 70)
            print("ÇATIŞMA TESPİT MEKANİZMASI BAŞLATILDI (TEZ 4.5.2)")
            print("=" * 70)
            print("✓ Çatışma matrisi yüklendi")
            print("✓ Risk analiz motoru hazır")
            print("=" * 70)
    
    def detect_conflict(
        self,
        xdss_action: str,
        operator_action: str,
        xdss_confidence: float,
        operator_confidence: float,
        model_prob: float,
        context: Dict = None
    ) -> Dict:
        """
        Çatışma tespiti yap.
        
        TEZ: Bu fonksiyon XDSS ve Operatör kararları arasındaki
        uyumsuzluğu tespit eder ve şiddet seviyesini belirler.
        
        Args:
            xdss_action: XDSS önerisi
            operator_action: Operatör kararı
            xdss_confidence: XDSS güven skoru
            operator_confidence: Operatör güven skoru
            model_prob: Model fail olasılığı
            context: Ek bağlam bilgisi
            
        Returns:
            conflict_report: {
                'conflict_flag': bool,
                'conflict_type': ConflictType,
                'severity': ConflictSeverity,
                'explanation': str,
                'risk_score': float,
                'risk_factors': List[str]
            }
        """
        
        # 1. Çatışma varlığını kontrol et
        conflict_flag = (xdss_action != operator_action)
        
        # 2. Çatışma tipi belirle
        if not conflict_flag:
            conflict_type = ConflictType.NO_CONFLICT
            severity = ConflictSeverity.NONE
        else:
            conflict_type, severity = self._classify_conflict(
                xdss_action, operator_action
            )
        
        # 3. Risk faktörlerini analiz et
        risk_factors = self._identify_risk_factors(
            xdss_action, operator_action, 
            xdss_confidence, operator_confidence,
            model_prob
        )
        
        # 4. Risk skoru hesapla
        risk_score = self._calculate_conflict_risk(
            severity, xdss_confidence, operator_confidence,
            model_prob, risk_factors
        )
        
        # 5. Açıklama üret
        explanation = self._generate_explanation(
            xdss_action, operator_action,
            conflict_type, severity,
            risk_factors
        )
        
        # Rapor oluştur
        conflict_report = {
            'conflict_flag': conflict_flag,
            'conflict_type': conflict_type,
            'severity': severity,
            'explanation': explanation,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'xdss_action': xdss_action,
            'operator_action': operator_action,
            'xdss_confidence': xdss_confidence,
            'operator_confidence': operator_confidence
        }
        
        # Geçmişe kaydet
        self.conflict_history.append(conflict_report)
        
        return conflict_report
    
    def _classify_conflict(
        self,
        xdss_action: str,
        operator_action: str
    ) -> Tuple[ConflictType, ConflictSeverity]:
        """
        Çatışma tipini ve şiddetini sınıflandır.
        
        TEZ: Çatışma matrisi kullanarak çatışmayı kategorize eder.
        """
        
        # Çatışma matrisi lookup
        key = (xdss_action, operator_action)
        severity = self.CONFLICT_MATRIX.get(key, ConflictSeverity.MODERATE)
        
        # Çatışma tipi
        if severity == ConflictSeverity.NONE:
            conflict_type = ConflictType.NO_CONFLICT
        
        elif severity == ConflictSeverity.CRITICAL:
            conflict_type = ConflictType.FULL_OVERRIDE
        
        else:
            conflict_type = ConflictType.PARTIAL_MISMATCH
        
        return conflict_type, severity
    
    def _identify_risk_factors(
        self,
        xdss_action: str,
        operator_action: str,
        xdss_conf: float,
        operator_conf: float,
        model_prob: float
    ) -> List[str]:
        """
        Risk faktörlerini belirle.
        
        TEZ: Çatışmanın neden tehlikeli olduğunu açıklayan faktörleri listeler.
        """
        
        risk_factors = []
        
        # 1. Yüksek model risk + XDSS ignore
        if model_prob > 0.8 and xdss_action == "STOP" and operator_action != "STOP":
            risk_factors.append(
                f"CRITICAL_RISK_IGNORED: Model yüksek risk tespit etti (p={model_prob:.2f}) "
                f"ama operatör STOP önerisini kabul etmedi"
            )
        
        # 2. XDSS yüksek güvenle öneriyor ama operatör dinlemiyor
        if xdss_conf > 0.8 and xdss_action != operator_action:
            risk_factors.append(
                f"HIGH_CONFIDENCE_OVERRIDE: XDSS yüksek güvenle {xdss_action} öneriyor "
                f"(conf={xdss_conf:.2f}) ama operatör {operator_action} kararı verdi"
            )
        
        # 3. Operatör düşük güvenle override yapıyor
        if operator_conf < 0.6 and xdss_action != operator_action:
            risk_factors.append(
                f"LOW_CONFIDENCE_OVERRIDE: Operatör düşük güvenle (conf={operator_conf:.2f}) "
                f"XDSS önerisini değiştirdi"
            )
        
        # 4. Risk altında gevşetme (STOP → CONTINUE)
        if xdss_action == "STOP" and operator_action == "CONTINUE":
            risk_factors.append(
                "SAFETY_RELAXATION: Operatör kritik STOP önerisini tamamen görmezden geldi"
            )
        
        # 5. Orta risk bölgesinde belirsizlik
        if 0.5 < model_prob < 0.7 and abs(xdss_conf - operator_conf) > 0.3:
            risk_factors.append(
                f"DECISION_UNCERTAINTY: Belirsiz bölgede (p={model_prob:.2f}) "
                f"XDSS ve operatör güven farkı yüksek"
            )
        
        # 6. GüÃ§ hiyerarÅŸisi ihlali
        xdss_level = self.ACTION_HIERARCHY.get(xdss_action, 0)
        operator_level = self.ACTION_HIERARCHY.get(operator_action, 0)
        
        if operator_level < xdss_level - 2:
            risk_factors.append(
                f"HIERARCHY_VIOLATION: Operatör önemli ölçüde daha gevşek aksiyon seçti "
                f"({xdss_action} → {operator_action})"
            )
        
        return risk_factors
    
    def _calculate_conflict_risk(
        self,
        severity: ConflictSeverity,
        xdss_conf: float,
        operator_conf: float,
        model_prob: float,
        risk_factors: List[str]
    ) -> float:
        """
        Çatışma risk skoru hesapla (0-1).
        
        TEZ: Bu skor XSM Interlock'un karar vermesi için kritik girdilerden biridir.
        """
        
        # Severity base skoru
        severity_scores = {
            ConflictSeverity.NONE: 0.0,
            ConflictSeverity.LOW: 0.2,
            ConflictSeverity.MODERATE: 0.5,
            ConflictSeverity.CRITICAL: 0.8
        }
        
        base_score = severity_scores.get(severity, 0.5)
        
        # Faktör ağırlıkları
        # 1. Model risk (fail olasılığı)
        model_risk_weight = model_prob * 0.3
        
        # 2. XDSS güven (yüksek güven = daha ciddi çatışma)
        xdss_weight = xdss_conf * 0.2
        
        # 3. Operatör belirsizlik (düşük güven = daha riskli)
        operator_uncertainty = (1 - operator_conf) * 0.2
        
        # 4. Risk faktör sayısı
        factor_weight = min(len(risk_factors) * 0.1, 0.3)
        
        # Toplam risk
        risk_score = min(
            base_score + model_risk_weight + xdss_weight + 
            operator_uncertainty + factor_weight,
            1.0
        )
        
        return risk_score
    
    def _generate_explanation(
        self,
        xdss_action: str,
        operator_action: str,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        risk_factors: List[str]
    ) -> str:
        """
        İnsan tarafından okunabilir açıklama üret.
        
        TEZ: Bu açıklama sistemin kararını şeffaf kılar.
        """
        
        if conflict_type == ConflictType.NO_CONFLICT:
            return (f"Çatışma tespit edilmedi. XDSS ve operatör kararı uyumlu: {xdss_action}")
        
        explanation = f"Çatışma Tespit Edildi:\n"
        explanation += f"  • XDSS Önerisi: {xdss_action}\n"
        explanation += f"  • Operatör Kararı: {operator_action}\n"
        explanation += f"  • Çatışma Tipi: {conflict_type.value.upper()}\n"
        explanation += f"  • Şiddet: {severity.value.upper()}\n"
        
        if risk_factors:
            explanation += f"\nRisk Faktörleri ({len(risk_factors)}):\n"
            for i, factor in enumerate(risk_factors[:3], 1):  # İlk 3'ü göster
                explanation += f"  {i}. {factor}\n"
        
        return explanation
    
    def get_conflict_statistics(self) -> Dict:
        """
        Çatışma istatistiklerini hesapla.
        
        TEZ: Bu metrikler sistem performansını ve operatör-sistem etkileşimini değerlendirir.
        """
        
        if not self.conflict_history:
            return {
                'total_decisions': 0,
                'conflict_rate': 0.0,
                'critical_conflicts': 0,
                'avg_risk_score': 0.0
            }
        
        total = len(self.conflict_history)
        conflicts = sum(1 for c in self.conflict_history if c['conflict_flag'])
        critical = sum(
            1 for c in self.conflict_history 
            if c['severity'] == ConflictSeverity.CRITICAL
        )
        
        avg_risk = np.mean([c['risk_score'] for c in self.conflict_history])
        
        return {
            'total_decisions': total,
            'conflict_count': conflicts,
            'conflict_rate': conflicts / total if total > 0 else 0.0,
            'critical_conflicts': critical,
            'critical_rate': critical / total if total > 0 else 0.0,
            'avg_risk_score': avg_risk
        }
    
    def format_conflict_report(self, conflict: Dict) -> str:
        """Çatışma raporunu formatla."""
        
        report = "\n" + "=" * 70
        report += "\n⚠️  ÇATIŞMA TESPİT RAPORU (TEZ 4.5.2)"
        report += "\n" + "=" * 70
        
        if not conflict['conflict_flag']:
            report += "\n✓ Çatışma tespit edilmedi. XDSS ve Operatör uyumlu."
        else:
            report += f"\n⚠️  ÇATIŞMA: {conflict['conflict_type'].value.upper()}"
            report += f"\n  Şiddet: {conflict['severity'].value.upper()}"
            report += f"\n  Risk Skoru: {conflict['risk_score']:.3f}"
            report += f"\n\n  XDSS → {conflict['xdss_action']} (güven: {conflict['xdss_confidence']:.2f})"
            report += f"\n  Operatör → {conflict['operator_action']} (güven: {conflict['operator_confidence']:.2f})"
            
            if conflict['risk_factors']:
                report += f"\n\n  Risk Faktörleri ({len(conflict['risk_factors'])}):"
                for i, factor in enumerate(conflict['risk_factors'][:3], 1):
                    report += f"\n    {i}. {factor[:80]}..."
        
        report += "\n" + "=" * 70
        
        return report


# =============================================================================
# DEMO FONKSİYONU
# =============================================================================

def demo_conflict_detector():
    """Çatışma tespit demo."""
    
    print("\n" + "🎮 " + "=" * 66)
    print("ÇATIŞMA TESPİT MEKANİZMASI DEMO - TEZ 4.5.2")
    print("=" * 70)
    
    detector = ConflictDetector(verbose=False)
    
    # Test senaryoları
    scenarios = [
        {
            "name": "Kritik Çatışma",
            "xdss": "STOP",
            "operator": "CONTINUE",
            "xdss_conf": 0.9,
            "operator_conf": 0.7,
            "model_prob": 0.88
        },
        {
            "name": "Orta Çatışma",
            "xdss": "CHECK",
            "operator": "CONTINUE",
            "xdss_conf": 0.7,
            "operator_conf": 0.6,
            "model_prob": 0.55
        },
        {
            "name": "Çatışma Yok",
            "xdss": "STOP",
            "operator": "STOP",
            "xdss_conf": 0.85,
            "operator_conf": 0.80,
            "model_prob": 0.91
        },
        {
            "name": "Düşük Güven Override",
            "xdss": "CHECK",
            "operator": "STOP",
            "xdss_conf": 0.65,
            "operator_conf": 0.45,
            "model_prob": 0.62
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SENARYO {i}: {scenario['name']}")
        print(f"{'='*70}")
        
        conflict = detector.detect_conflict(
            xdss_action=scenario['xdss'],
            operator_action=scenario['operator'],
            xdss_confidence=scenario['xdss_conf'],
            operator_confidence=scenario['operator_conf'],
            model_prob=scenario['model_prob']
        )
        
        print(detector.format_conflict_report(conflict))
    
    # İstatistikler
    stats = detector.get_conflict_statistics()
    print(f"\n{'='*70}")
    print("GENEL İSTATİSTİKLER")
    print(f"{'='*70}")
    print(f"Toplam Karar: {stats['total_decisions']}")
    print(f"Çatışma Oranı: {stats['conflict_rate']:.1%}")
    print(f"Kritik Çatışma Oranı: {stats['critical_rate']:.1%}")
    print(f"Ortalama Risk Skoru: {stats['avg_risk_score']:.3f}")
    
    print("\n" + "=" * 70)
    print("✔ Demo tamamlandı!")


if __name__ == "__main__":
    demo_conflict_detector()
