"""
================================================================================
OPERATOR DECISION MODULE (ODM)
X-HARMONY Mimarisi - Human-in-the-Loop Katmanı
================================================================================

TEZ BAŞLIĞI: 4.5.1 - Operatör Karar Katmanı (Operator Decision Layer)

Bu modül, X-HARMONY mimarisinde insan faktörünü temsil eder.
XDSS'nin önerilerini alan operatör, kendi deneyimi ve durumsal 
yargısıyla nihai kararı verir.

Operatör Aksiyonları:
    - STOP: Üretimi durdur
    - CONTINUE: Devam et
    - CHECK: Kalite kontrole gönder
    - INCREASE_QC: Yoğun kontrol başlat
    - ADJUST_PARAMS: Parametreleri ayarla

Operatör Davranış Modelleri:
    - COMPLIANT: XDSS önerisine tamamen uyar (%95 uyum)
    - CAUTIOUS: Daha muhafazakar kararlar verir (risk odaklı)
    - OPTIMISTIC: Üretimi sürdürmeye eğilimli (verimlilik odaklı)
    - EXPERIENCED: Yaşanmış senaryolara göre karar verir
    - RANDOM: Tutarsız kararlar (sınır durum - test amaçlı)

Karar Metrikleri:
    - decision_confidence: Operatörün kararındaki kesinlik (0-1)
    - experience_years: Operatör deneyimi (yıl)
    - historical_accuracy: Geçmiş doğru karar oranı (0-1)
    - workload_stress: İş yükü baskısı (0-1, yüksek=stresli)

Yazar: X-HARMONY Implementation - Thesis Chapter 4.5.1
================================================================================
"""

import numpy as np
from typing import Dict, List, Literal
from enum import Enum
import random


class OperatorBehaviorType(Enum):
    """Operatör davranış tipleri."""
    COMPLIANT = "compliant"           # XDSS'ye uyumlu
    CAUTIOUS = "cautious"             # Muhafazakar
    OPTIMISTIC = "optimistic"         # İyimser
    EXPERIENCED = "experienced"       # Deneyimli
    RANDOM = "random"                 # Rastgele (test için)


class OperatorProfile:
    """
    Operatör profili - her operatörün özelliklerini tanımlar.
    
    TEZ: Bu sınıf operatörün bireysel özelliklerini ve deneyim seviyesini
    modelleyerek gerçekçi karar simülasyonu sağlar.
    """
    
    def __init__(
        self,
        operator_id: str,
        behavior_type: OperatorBehaviorType,
        experience_years: float,
        historical_accuracy: float,
        risk_tolerance: float = 0.5,
        workload_stress: float = 0.3
    ):
        """
        Operatör profili oluştur.
        
        Args:
            operator_id: Operatör kimliği (ör: "OPR001")
            behavior_type: Davranış tipi
            experience_years: Deneyim (yıl)
            historical_accuracy: Geçmiş başarı oranı (0-1)
            risk_tolerance: Risk toleransı (0=çok temkinli, 1=risk seven)
            workload_stress: İş yükü stresi (0-1)
        """
        self.operator_id = operator_id
        self.behavior_type = behavior_type
        self.experience_years = experience_years
        self.historical_accuracy = historical_accuracy
        self.risk_tolerance = risk_tolerance
        self.workload_stress = workload_stress
        
    def __repr__(self):
        return (f"OperatorProfile(id={self.operator_id}, "
                f"type={self.behavior_type.value}, "
                f"exp={self.experience_years}y, "
                f"acc={self.historical_accuracy:.2f})")


class OperatorDecisionModule:
    """
    Operatör Karar Modülü.
    
    TEZ 4.5.1: Bu modül XDSS önerisini alır ve operatör davranış modeline
    göre nihai kararı verir. Operatör-sistem uyumsuzluğu bu katmanda oluşur.
    """
    
    # Karar aksiyonları
    ACTIONS = ["STOP", "CONTINUE", "CHECK", "INCREASE_QC", "ADJUST_PARAMS"]
    
    # XDSS → Operatör aksiyon eşleştirmesi (varsayılan)
    DEFAULT_ACTION_MAP = {
        "STOP": "STOP",
        "CHECK": "CHECK", 
        "CONTINUE": "CONTINUE"
    }
    
    def __init__(
        self, 
        operator_profile: OperatorProfile,
        verbose: bool = True
    ):
        """
        Operatör karar modülünü başlat.
        
        Args:
            operator_profile: Operatör profili
            verbose: Detaylı çıktı
        """
        self.profile = operator_profile
        self.verbose = verbose
        
        # Karar geçmişi
        self.decision_history = []
        
        if self.verbose:
            print("=" * 70)
            print("OPERATÖR KARAR MODÜLÜ BAŞLATILDI (TEZ 4.5.1)")
            print("=" * 70)
            print(f"✓ Operatör: {self.profile}")
            print("=" * 70)
    
    def make_decision(
        self,
        xdss_recommendation: str,
        xdss_confidence: float,
        model_prob: float,
        context: Dict = None
    ) -> Dict:
        """
        Operatör kararını ver.
        
        TEZ: Bu fonksiyon operatör davranış modeline göre:
        1. XDSS önerisini kabul edebilir
        2. Tersine çevirebilir (override)
        3. Alternatif aksiyon seçebilir
        
        Args:
            xdss_recommendation: XDSS önerisi (STOP/CHECK/CONTINUE)
            xdss_confidence: XDSS güven skoru (0-1)
            model_prob: Model fail olasılığı (0-1)
            context: Ek bağlamsal bilgi (opsiyonel)
            
        Returns:
            decision_dict: {
                'operator_action': str,
                'operator_confidence': float,
                'agreement_with_xdss': bool,
                'reasoning': str,
                'override_flag': bool
            }
        """
        
        # Operatör davranış tipine göre karar
        if self.profile.behavior_type == OperatorBehaviorType.COMPLIANT:
            decision = self._compliant_decision(xdss_recommendation, xdss_confidence)
        
        elif self.profile.behavior_type == OperatorBehaviorType.CAUTIOUS:
            decision = self._cautious_decision(xdss_recommendation, model_prob)
        
        elif self.profile.behavior_type == OperatorBehaviorType.OPTIMISTIC:
            decision = self._optimistic_decision(xdss_recommendation, model_prob)
        
        elif self.profile.behavior_type == OperatorBehaviorType.EXPERIENCED:
            decision = self._experienced_decision(
                xdss_recommendation, model_prob, xdss_confidence
            )
        
        elif self.profile.behavior_type == OperatorBehaviorType.RANDOM:
            decision = self._random_decision()
        
        else:
            # Varsayılan: uyumlu davranış
            decision = self._compliant_decision(xdss_recommendation, xdss_confidence)
        
        # Uyuşma kontrolü
        decision['agreement_with_xdss'] = (
            decision['operator_action'] == self._map_xdss_to_action(xdss_recommendation)
        )
        
        # Override flag
        decision['override_flag'] = not decision['agreement_with_xdss']
        
        # Geçmişe kaydet
        self.decision_history.append({
            'xdss_rec': xdss_recommendation,
            'operator_decision': decision['operator_action'],
            'override': decision['override_flag']
        })
        
        return decision
    
    def _compliant_decision(
        self, 
        xdss_rec: str, 
        xdss_conf: float
    ) -> Dict:
        """
        UYUMLU operatör - XDSS önerisine çoğunlukla uyar (%95).
        
        TEZ: Bu model, sisteme güvenen ve önerilere uyan ideal operatörü temsil eder.
        """
        # %95 olasılıkla XDSS'ye uy
        if random.random() < 0.95:
            action = self._map_xdss_to_action(xdss_rec)
            reasoning = f"XDSS önerisi ({xdss_rec}) kabul edildi. Güven: {xdss_conf:.2f}"
            confidence = xdss_conf * 0.9  # Operatör XDSS'den biraz daha az emin
        else:
            # %5 olasılıkla farklı karar (sezgisel)
            action = random.choice(["CHECK", "STOP"])
            reasoning = f"XDSS önerisi ({xdss_rec}) yerine sezgisel karar: {action}"
            confidence = 0.6
        
        return {
            'operator_action': action,
            'operator_confidence': confidence,
            'reasoning': reasoning
        }
    
    def _cautious_decision(
        self, 
        xdss_rec: str, 
        model_prob: float
    ) -> Dict:
        """
        MUHAFAZAKAR operatör - Risk varsa daha temkinli davranır.
        
        TEZ: Bu model güvenlik öncelikli, hata maliyetini minimize eden operatörü temsil eder.
        """
        # Yüksek risk → her zaman STOP/CHECK
        if model_prob > 0.6:
            if model_prob > 0.8:
                action = "STOP"
                reasoning = f"Yüksek risk (p={model_prob:.2f}) → güvenlik için STOP"
                confidence = 0.9
            else:
                action = "CHECK"
                reasoning = f"Orta-yüksek risk (p={model_prob:.2f}) → CHECK tercih edildi"
                confidence = 0.8
        else:
            # Düşük risk → XDSS'ye uy
            action = self._map_xdss_to_action(xdss_rec)
            reasoning = f"Düşük risk, XDSS önerisi ({xdss_rec}) kabul edildi"
            confidence = 0.75
        
        return {
            'operator_action': action,
            'operator_confidence': confidence,
            'reasoning': reasoning
        }
    
    def _optimistic_decision(
        self, 
        xdss_rec: str, 
        model_prob: float
    ) -> Dict:
        """
        İYİMSER operatör - Üretimi sürdürmeye eğilimli.
        
        TEZ: Bu model verimlilik odaklı, duruşları minimize eden operatörü temsil eder.
        """
        # Düşük-orta risk → CONTINUE
        if model_prob < 0.7:
            action = "CONTINUE"
            reasoning = f"Risk kabul edilebilir (p={model_prob:.2f}) → üretim devam"
            confidence = 0.7
        elif model_prob < 0.85:
            action = "CHECK"
            reasoning = f"Orta-yüksek risk (p={model_prob:.2f}) → CHECK yeterli"
            confidence = 0.65
        else:
            action = "STOP"
            reasoning = f"Kritik risk (p={model_prob:.2f}) → STOP zorunlu"
            confidence = 0.85
        
        return {
            'operator_action': action,
            'operator_confidence': confidence,
            'reasoning': reasoning
        }
    
    def _experienced_decision(
        self,
        xdss_rec: str,
        model_prob: float,
        xdss_conf: float
    ) -> Dict:
        """
        DENEYİMLİ operatör - Bağlamsal karar verir.
        
        TEZ: Bu model yüksek deneyim ve başarı oranına sahip operatörü temsil eder.
        Hem sisteme hem kendi sezgilerine güvenir.
        """
        # Deneyim ve sistem güveni dengeli
        
        # XDSS çok emin ve deneyim yüksek → uy
        if xdss_conf > 0.8 and self.profile.experience_years > 5:
            action = self._map_xdss_to_action(xdss_rec)
            reasoning = (f"XDSS yüksek güvenle {xdss_rec} öneriyor, "
                        f"deneyimim de destekliyor")
            confidence = min(xdss_conf * 1.1, 0.95)
        
        # XDSS belirsiz ama deneyim var → kendi kararı
        elif xdss_conf < 0.6 and self.profile.experience_years > 3:
            if model_prob > 0.7:
                action = "CHECK"
            elif model_prob > 0.85:
                action = "STOP"
            else:
                action = "CONTINUE"
            reasoning = f"XDSS belirsiz, deneyimime göre {action} kararı"
            confidence = 0.75
        
        # Varsayılan → XDSS'ye uy
        else:
            action = self._map_xdss_to_action(xdss_rec)
            reasoning = f"Standart prosedür: {xdss_rec}"
            confidence = xdss_conf * 0.9
        
        return {
            'operator_action': action,
            'operator_confidence': confidence,
            'reasoning': reasoning
        }
    
    def _random_decision(self) -> Dict:
        """
        RASTGELE operatör - Test ve edge case analizi için.
        
        TEZ: Bu model worst-case senaryosu ve sistemin dayanıklılığını test eder.
        """
        action = random.choice(self.ACTIONS)
        confidence = random.uniform(0.3, 0.8)
        reasoning = "Rastgele karar (test modu)"
        
        return {
            'operator_action': action,
            'operator_confidence': confidence,
            'reasoning': reasoning
        }
    
    def _map_xdss_to_action(self, xdss_decision: str) -> str:
        """XDSS kararını operatör aksiyonuna eşle."""
        return self.DEFAULT_ACTION_MAP.get(xdss_decision, "CHECK")
    
    def get_operator_statistics(self) -> Dict:
        """
        Operatör istatistiklerini hesapla.
        
        TEZ: Bu metrikler operatör performansını ve sistem uyumunu değerlendirir.
        """
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'override_rate': 0.0,
                'agreement_rate': 0.0
            }
        
        total = len(self.decision_history)
        overrides = sum(1 for d in self.decision_history if d['override'])
        
        return {
            'total_decisions': total,
            'override_rate': overrides / total,
            'agreement_rate': 1 - (overrides / total),
            'profile': self.profile
        }
    
    def format_decision_report(self, decision: Dict) -> str:
        """Karar raporunu formatla."""
        
        report = "\n" + "=" * 70
        report += "\n📋 OPERATÖR KARARI (TEZ 4.5.1)"
        report += "\n" + "=" * 70
        report += f"\n  Operatör: {self.profile.operator_id}"
        report += f"\n  Profil: {self.profile.behavior_type.value.upper()}"
        report += f"\n  Deneyim: {self.profile.experience_years} yıl"
        report += f"\n\n  Aksiyon: {decision['operator_action']}"
        report += f"\n  Güven: {decision['operator_confidence']:.2f}"
        report += f"\n  XDSS ile Uyum: {'✓ EVET' if decision['agreement_with_xdss'] else '✗ HAYIR (OVERRIDE)'}"
        report += f"\n  Gerekçe: {decision['reasoning']}"
        report += "\n" + "=" * 70
        
        return report


# =============================================================================
# OPERATÖR PROFİL FABRİKASI
# =============================================================================

def create_operator_profiles() -> Dict[str, OperatorProfile]:
    """
    Hazır operatör profilleri oluştur.
    
    TEZ: Bu fonksiyon farklı operatör tiplerini modelleyerek
    gerçekçi simülasyon senaryoları sağlar.
    """
    
    profiles = {
        # Deneyimli ve uyumlu operatör
        "EXPERT_COMPLIANT": OperatorProfile(
            operator_id="OPR_E001",
            behavior_type=OperatorBehaviorType.COMPLIANT,
            experience_years=8.0,
            historical_accuracy=0.92,
            risk_tolerance=0.5,
            workload_stress=0.2
        ),
        
        # Temkinli operatör
        "CAUTIOUS_MID": OperatorProfile(
            operator_id="OPR_C001",
            behavior_type=OperatorBehaviorType.CAUTIOUS,
            experience_years=5.0,
            historical_accuracy=0.88,
            risk_tolerance=0.2,
            workload_stress=0.3
        ),
        
        # İyimser operatör
        "OPTIMISTIC_SENIOR": OperatorProfile(
            operator_id="OPR_O001",
            behavior_type=OperatorBehaviorType.OPTIMISTIC,
            experience_years=6.5,
            historical_accuracy=0.85,
            risk_tolerance=0.7,
            workload_stress=0.4
        ),
        
        # Çok deneyimli
        "VETERAN": OperatorProfile(
            operator_id="OPR_V001",
            behavior_type=OperatorBehaviorType.EXPERIENCED,
            experience_years=12.0,
            historical_accuracy=0.94,
            risk_tolerance=0.5,
            workload_stress=0.15
        ),
        
        # Acemi operatör
        "NOVICE": OperatorProfile(
            operator_id="OPR_N001",
            behavior_type=OperatorBehaviorType.COMPLIANT,
            experience_years=1.5,
            historical_accuracy=0.75,
            risk_tolerance=0.4,
            workload_stress=0.6
        ),
        
        # Test için rastgele
        "RANDOM_TEST": OperatorProfile(
            operator_id="OPR_TEST",
            behavior_type=OperatorBehaviorType.RANDOM,
            experience_years=3.0,
            historical_accuracy=0.50,
            risk_tolerance=0.5,
            workload_stress=0.5
        )
    }
    
    return profiles


# =============================================================================
# DEMO FONKSİYONU
# =============================================================================

def demo_operator_module():
    """Operatör modülü demo."""
    
    print("\n" + "🎮 " + "=" * 66)
    print("OPERATÖR KARARI MODÜLÜ DEMO - TEZ 4.5.1")
    print("=" * 70)
    
    # Profiller oluştur
    profiles = create_operator_profiles()
    
    # Test senaryosu
    xdss_scenarios = [
        {"recommendation": "STOP", "confidence": 0.85, "model_prob": 0.92},
        {"recommendation": "CHECK", "confidence": 0.65, "model_prob": 0.58},
        {"recommendation": "CONTINUE", "confidence": 0.78, "model_prob": 0.25}
    ]
    
    # Her profil için test
    for profile_name in ["EXPERT_COMPLIANT", "CAUTIOUS_MID", "OPTIMISTIC_SENIOR"]:
        profile = profiles[profile_name]
        odm = OperatorDecisionModule(profile, verbose=False)
        
        print(f"\n{'='*70}")
        print(f"PROFİL: {profile_name}")
        print(f"{'='*70}")
        
        for i, scenario in enumerate(xdss_scenarios):
            decision = odm.make_decision(
                xdss_recommendation=scenario['recommendation'],
                xdss_confidence=scenario['confidence'],
                model_prob=scenario['model_prob']
            )
            
            print(f"\nSenaryo {i+1}: XDSS={scenario['recommendation']} (conf={scenario['confidence']:.2f})")
            print(f"  → Operatör: {decision['operator_action']} (conf={decision['operator_confidence']:.2f})")
            print(f"  → Override: {'EVET ⚠️' if decision['override_flag'] else 'HAYIR ✓'}")
            print(f"  → Gerekçe: {decision['reasoning']}")
        
        stats = odm.get_operator_statistics()
        print(f"\nİSTATİSTİKLER:")
        print(f"  Override Oranı: {stats['override_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("✔ Demo tamamlandı!")


if __name__ == "__main__":
    demo_operator_module()