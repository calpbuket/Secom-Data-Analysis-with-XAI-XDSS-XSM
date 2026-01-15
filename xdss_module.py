"""
================================================================================
XDSS - EXPLAINABLE DECISION SUPPORT SYSTEM
X-HARMONY Mimarisi - Karar Destek Katmanı
================================================================================

Bu modül, XGBoost modelinin tahminlerini SHAP açıklamalarıyla birleştirerek
üretim sürecinde aksiyon alabilen açıklanabilir bir karar destek sistemi sağlar.

Karar Seviyeleri:
    - STOP: Üretim durdurulmalı (kritik risk)
    - CHECK: Müdahale gerekli (orta risk)
    - CONTINUE: Devam edilebilir (düşük risk)

Giriş:
    - Model tahmin olasılığı (p_fail)
    - SHAP değerleri
    - Sensör değerleri
    - Kural tabanı (CSV)

Çıkış:
    - Karar seviyesi
    - Risk gerekçeleri
    - Güven skoru
    - Etkilenen sensörler
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class XDSSModule:
    """
    Explainable Decision Support System (XDSS) sınıfı.
    
    X-HARMONY mimarisinin karar destek katmanını implement eder.
    Model tahminlerini SHAP açıklamalarıyla birleştirerek
    açıklanabilir üretim kararları verir.
    """
    
    def __init__(self, rules_path: str, verbose: bool = True):
        """
        XDSS modülünü başlat.
        
        Args:
            rules_path: Kural tabanı CSV dosya yolu
            verbose: Detaylı çıktı için True
        """
        self.verbose = verbose
        self.rules_df = self._load_rules(rules_path)
        self.decision_thresholds = self._set_thresholds()
        
        if self.verbose:
            print("=" * 70)
            print("XDSS MODÜLÜ BAŞLATILDI")
            print("=" * 70)
            print(f"✓ Kural tabanı yüklendi: {len(self.rules_df)} sensör kuralı")
            print(f"✓ Karar eşikleri ayarlandı")
            print("=" * 70)
    
    def _load_rules(self, rules_path: str) -> pd.DataFrame:
        """Kural tabanını yükle ve hazırla."""
        rules_df = pd.read_csv(rules_path)
        
        # Sensör ID'lerini integer'a çevir
        rules_df['Sensor'] = rules_df['Sensor'].astype(str)
        
        # Eşik değerlerini parse et
        rules_df['threshold_lower'] = rules_df['Recommended_Threshold'].apply(
            lambda x: float(str(x).split(' - ')[0])
        )
        rules_df['threshold_upper'] = rules_df['Recommended_Threshold'].apply(
            lambda x: float(str(x).split(' - ')[1]) if ' - ' in str(x) else float(str(x).split(' - ')[0])
        )
        
        return rules_df
    
    def _set_thresholds(self) -> Dict[str, float]:
        """Karar eşiklerini belirle."""
        return {
            'stop_prob': 0.8,      # p_fail >= 0.8 → STOP
            'check_prob': 0.5,     # 0.5 <= p_fail < 0.8 → CHECK
            'critical_sensors': 3,  # >= 3 kritik sensör → STOP
            'warning_sensors': 1,   # >= 1 uyarı sensörü → CHECK
            'shap_threshold': 0.1   # |SHAP| >= 0.1 → anlamlı katkı
        }
    
    def _analyze_sensors(
        self, 
        shap_values: np.ndarray, 
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Sensör analizini gerçekleştir.
        
        SHAP değerlerine göre risk yaratan sensörleri belirle.
        
        Args:
            shap_values: SHAP değerleri vektörü
            feature_values: Sensör değerleri vektörü
            feature_names: Sensör isimleri
            
        Returns:
            critical_sensors: Kritik risk sensörleri
            warning_sensors: Uyarı seviyesi sensörleri
        """
        critical_sensors = []
        warning_sensors = []
        
        # SHAP değerleri pozitif ve anlamlı olan sensörlere odaklan
        # (Pozitif SHAP = Fail riskini artırıyor)
        for i, (shap_val, sensor_val, sensor_name) in enumerate(
            zip(shap_values, feature_values, feature_names)
        ):
            # Sadece pozitif ve anlamlı SHAP değerlerini incele
            if shap_val > self.decision_thresholds['shap_threshold']:
                
                # Kural tabanında bu sensör var mı?
                sensor_rule = self.rules_df[
                    self.rules_df['Sensor'] == sensor_name
                ]
                
                if not sensor_rule.empty:
                    rule = sensor_rule.iloc[0]
                    
                    # Risk yönünü kontrol et
                    risk_direction = rule['Risk_Direction']
                    threshold_lower = rule['threshold_lower']
                    threshold_upper = rule['threshold_upper']
                    
                    is_violating = False
                    severity = "WARNING"
                    
                    # DÜŞÜK değerler risk yaratıyorsa
                    if "DÜŞÜK" in risk_direction:
                        if sensor_val < threshold_lower:
                            is_violating = True
                            # P25'in altındaysa kritik
                            severity = "CRITICAL" if sensor_val < rule['P25'] else "WARNING"
                    
                    # YÜKSEK değerler risk yaratıyorsa  
                    elif "YÜKSEK" in risk_direction:
                        if sensor_val > threshold_upper:
                            is_violating = True
                            severity = "CRITICAL" if sensor_val > rule['P75'] else "WARNING"
                    
                    if is_violating:
                        sensor_info = {
                            'sensor': sensor_name,
                            'value': float(sensor_val),
                            'shap': float(shap_val),
                            'threshold': f"{threshold_lower:.2f} - {threshold_upper:.2f}",
                            'rule': rule['Rule_Condition'],
                            'importance_rank': int(rule['Importance_Rank']),
                            'severity': severity
                        }
                        
                        if severity == "CRITICAL":
                            critical_sensors.append(sensor_info)
                        else:
                            warning_sensors.append(sensor_info)
        
        # Önem sırasına göre sırala
        critical_sensors = sorted(critical_sensors, key=lambda x: x['importance_rank'])
        warning_sensors = sorted(warning_sensors, key=lambda x: x['importance_rank'])
        
        return critical_sensors, warning_sensors
    
    def _calculate_confidence(
        self, 
        p_fail: float, 
        n_critical: int, 
        n_warning: int,
        top_shap_contribution: float
    ) -> float:
        """
        Karar güven skorunu hesapla.
        
        Güven skoru şu faktörlere bağlı:
        - Model olasılığının kesinliği (0.1'e veya 0.9'a yakınlık)
        - Kritik sensör sayısı
        - Top SHAP katkısı
        
        Returns:
            0.0 - 1.0 arası güven skoru
        """
        # Model kesinliği (0.5'ten uzaklık)
        model_certainty = abs(p_fail - 0.5) * 2  # 0-1 arası normalize
        
        # Sensör kanıtı
        sensor_evidence = min((n_critical * 0.3 + n_warning * 0.1), 1.0)
        
        # SHAP kanıtı
        shap_evidence = min(top_shap_contribution / 0.5, 1.0)
        
        # Ağırlıklı ortalama
        confidence = (
            0.4 * model_certainty +
            0.4 * sensor_evidence +
            0.2 * shap_evidence
        )
        
        return round(confidence, 3)
    
    def xdss_decision(
        self,
        pred_prob: float,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        XDSS ana karar fonksiyonu.
        
        Model tahmini, SHAP açıklamaları ve sensör değerlerine dayanarak
        üretim aksiyonu belirler.
        
        Args:
            pred_prob: Model Fail tahmin olasılığı (0-1)
            shap_values: SHAP değerleri (n_features,)
            feature_values: Sensör değerleri (n_features,)
            feature_names: Sensör isimleri listesi
            
        Returns:
            decision_dict: {
                'decision': str,          # STOP / CHECK / CONTINUE
                'p_fail': float,          # Fail olasılığı
                'confidence': float,      # Güven skoru (0-1)
                'critical_sensors': list, # Kritik sensörler
                'warning_sensors': list,  # Uyarı sensörleri
                'reason': list,          # Karar gerekçeleri
                'action': str            # Önerilen aksiyon
            }
        """
        # 1. Sensör analizini yap
        critical_sensors, warning_sensors = self._analyze_sensors(
            shap_values, feature_values, feature_names
        )
        
        n_critical = len(critical_sensors)
        n_warning = len(warning_sensors)
        
        # 2. Karar mantığı
        decision = "CONTINUE"
        reasons = []
        action = "Üretim devam edebilir."
        
        # STOP koşulları
        if (pred_prob >= self.decision_thresholds['stop_prob'] or 
            n_critical >= self.decision_thresholds['critical_sensors']):
            decision = "STOP"
            reasons.append(f"Yüksek fail riski: p_fail={pred_prob:.3f}")
            if n_critical > 0:
                reasons.append(f"{n_critical} kritik sensör eşik dışı")
                top_critical = critical_sensors[:3]
                for s in top_critical:
                    reasons.append(
                        f"  → {s['sensor']}: {s['value']:.3f} "
                        f"(SHAP={s['shap']:.3f})"
                    )
            action = "ÜRETİMİ DURDUR! Kritik risk tespit edildi."
        
        # CHECK koşulları
        elif (pred_prob >= self.decision_thresholds['check_prob'] or
              n_warning >= self.decision_thresholds['warning_sensors']):
            decision = "CHECK"
            reasons.append(f"Orta seviye risk: p_fail={pred_prob:.3f}")
            if n_warning > 0:
                reasons.append(f"{n_warning} sensör uyarı seviyesinde")
                top_warning = warning_sensors[:3]
                for s in top_warning:
                    reasons.append(
                        f"  → {s['sensor']}: {s['value']:.3f} "
                        f"(SHAP={s['shap']:.3f})"
                    )
            action = "Mühendis müdahalesine ihtiyaç var. Sensörleri kontrol et."
        
        # CONTINUE
        else:
            reasons.append(f"Düşük risk: p_fail={pred_prob:.3f}")
            reasons.append("Tüm sensörler normal aralıkta")
            action = "Üretim güvenle devam edebilir."
        
        # 3. Güven skorunu hesapla
        top_shap = np.max(np.abs(shap_values)) if len(shap_values) > 0 else 0
        confidence = self._calculate_confidence(
            pred_prob, n_critical, n_warning, top_shap
        )
        
        # 4. Karar dictionary'sini oluştur
        decision_dict = {
            'decision': decision,
            'p_fail': round(pred_prob, 4),
            'confidence': confidence,
            'critical_sensors': critical_sensors,
            'warning_sensors': warning_sensors,
            'n_critical': n_critical,
            'n_warning': n_warning,
            'reason': reasons,
            'action': action,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return decision_dict
    
    def batch_decision(
        self,
        pred_probs: np.ndarray,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict]:
        """
        Batch (toplu) karar verme.
        
        Birden fazla örnek için aynı anda XDSS kararlarını üret.
        
        Args:
            pred_probs: Fail olasılıkları (n_samples,)
            shap_values: SHAP matrisi (n_samples, n_features)
            feature_values: Sensör matrisi (n_samples, n_features)
            feature_names: Feature isimleri
            
        Returns:
            decisions: Liste of decision dictionaries
        """
        decisions = []
        
        n_samples = len(pred_probs)
        if self.verbose:
            print(f"\n🔄 Batch karar verme başlatıldı: {n_samples} örnek")
        
        for i in range(n_samples):
            decision = self.xdss_decision(
                pred_probs[i],
                shap_values[i],
                feature_values[i],
                feature_names
            )
            decision['sample_id'] = i
            decisions.append(decision)
        
        if self.verbose:
            # Özet istatistikler
            stop_count = sum(1 for d in decisions if d['decision'] == 'STOP')
            check_count = sum(1 for d in decisions if d['decision'] == 'CHECK')
            continue_count = sum(1 for d in decisions if d['decision'] == 'CONTINUE')
            
            print(f"\n📊 Batch Karar Özeti:")
            print(f"  STOP: {stop_count} ({stop_count/n_samples*100:.1f}%)")
            print(f"  CHECK: {check_count} ({check_count/n_samples*100:.1f}%)")
            print(f"  CONTINUE: {continue_count} ({continue_count/n_samples*100:.1f}%)")
        
        return decisions
    
    def format_decision_report(self, decision: Dict) -> str:
        """
        Karar raporunu formatla (konsol çıktısı için).
        
        Args:
            decision: xdss_decision() çıktısı
            
        Returns:
            Formatlanmış rapor string
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("XDSS KARAR RAPORU")
        report.append("=" * 70)
        
        # Karar ve olasılık
        decision_emoji = {
            'STOP': '🛑',
            'CHECK': '⚠️',
            'CONTINUE': '✅'
        }
        emoji = decision_emoji.get(decision['decision'], '❓')
        
        report.append(f"\n{emoji} KARAR: {decision['decision']}")
        report.append(f"📊 Fail Olasılığı: {decision['p_fail']:.4f}")
        report.append(f"🎯 Güven Skoru: {decision['confidence']:.3f}")
        report.append(f"🕐 Zaman: {decision['timestamp']}")
        
        # Kritik sensörler
        if decision['n_critical'] > 0:
            report.append(f"\n🔴 KRİTİK SENSÖRLER ({decision['n_critical']}):")
            for sensor in decision['critical_sensors']:
                report.append(
                    f"  → {sensor['sensor']}: {sensor['value']:.3f} "
                    f"(SHAP={sensor['shap']:.3f}, Rank={sensor['importance_rank']})"
                )
        
        # Uyarı sensörleri
        if decision['n_warning'] > 0:
            report.append(f"\n🟡 UYARI SENSÖRLERİ ({decision['n_warning']}):")
            for sensor in decision['warning_sensors'][:5]:  # Max 5 göster
                report.append(
                    f"  → {sensor['sensor']}: {sensor['value']:.3f} "
                    f"(SHAP={sensor['shap']:.3f})"
                )
        
        # Gerekçeler
        report.append("\n📝 GEREKÇELER:")
        for reason in decision['reason']:
            report.append(f"  • {reason}")
        
        # Aksiyon
        report.append(f"\n💡 ÖNERİLEN AKSİYON:")
        report.append(f"  {decision['action']}")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def export_decisions_to_csv(
        self, 
        decisions: List[Dict], 
        output_path: str
    ) -> None:
        """
        Batch kararları CSV olarak kaydet.
        
        Args:
            decisions: batch_decision() çıktısı
            output_path: CSV dosya yolu
        """
        # Flatten et
        records = []
        for d in decisions:
            record = {
                'sample_id': d.get('sample_id', -1),
                'decision': d['decision'],
                'p_fail': d['p_fail'],
                'confidence': d['confidence'],
                'n_critical': d['n_critical'],
                'n_warning': d['n_warning'],
                'action': d['action'],
                'timestamp': d['timestamp']
            }
            
            # Top 3 kritik sensör
            for i in range(3):
                if i < len(d['critical_sensors']):
                    cs = d['critical_sensors'][i]
                    record[f'critical_{i+1}_sensor'] = cs['sensor']
                    record[f'critical_{i+1}_value'] = cs['value']
                    record[f'critical_{i+1}_shap'] = cs['shap']
                else:
                    record[f'critical_{i+1}_sensor'] = None
                    record[f'critical_{i+1}_value'] = None
                    record[f'critical_{i+1}_shap'] = None
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"✓ Kararlar CSV'ye kaydedildi: {output_path}")


# =============================================================================
# TESTİNG / DEMO
# =============================================================================

def demo_xdss():
    """XDSS modülü demo."""
    
    print("\n" + "=" * 70)
    print("XDSS MODÜLÜ DEMO")
    print("=" * 70)
    
    # Örnek kural tabanı (basitleştirilmiş)
    # Gerçekte CSV'den yüklenecek
    rules_data = {
        'Sensor': ['419', '33', '59', '486', '213'],
        'Importance_Rank': [11, 15, 5, 3, 25],
        'Mean_|SHAP|': [0.282, 0.257, 0.249, 0.247, 0.219],
        'Risk_Direction': [
            'DÜŞÜK DEĞERLER → Fail Riski ARTAR',
            'DÜŞÜK DEĞERLER → Fail Riski ARTAR',
            'DÜŞÜK DEĞERLER → Fail Riski ARTAR',
            'DÜŞÜK DEĞERLER → Fail Riski ARTAR',
            'DÜŞÜK DEĞERLER → Fail Riski ARTAR'
        ],
        'Rule_Condition': [
            'IF 419 < -0.4715 (25th percentile)',
            'IF 33 < -0.3923 (25th percentile)',
            'IF 59 < -0.4999 (25th percentile)',
            'IF 486 < -0.5043 (25th percentile)',
            'IF 213 < -0.4972 (25th percentile)'
        ],
        'Recommended_Threshold': [
            '-0.4715 - -0.0001',
            '-0.3923 - -0.0096',
            '-0.4999 - -0.0501',
            '-0.5043 - 0.0270',
            '-0.4972 - 0.0887'
        ],
        'P25': [-0.472, -0.392, -0.500, -0.504, -0.497],
        'P50': [-0.0001, -0.0096, -0.050, 0.027, 0.089],
        'P75': [0.585, 0.568, 0.435, 0.532, 0.596],
        'P90': [0.899, 1.172, 2.052, 1.058, 1.247]
    }
    
    rules_df = pd.DataFrame(rules_data)
    rules_df.to_csv('./xai_analysis_outputs/xai_analysis_outputs4_xdss_xsm_rules.csv', index=False)
    
    # XDSS'yi başlat
    xdss = XDSSModule(rules_path='./xai_analysis_outputs/xai_analysis_outputs4_xdss_xsm_rules.csv', verbose=True)
    
    # Senaryo 1: STOP durumu (yüksek risk + kritik sensörler)
    print("\n\n🔴 SENARYO 1: KRITIK DURUM (STOP bekleniyor)")
    decision1 = xdss.xdss_decision(
        pred_prob=0.92,
        shap_values=np.array([0.35, 0.28, 0.25, 0.22, 0.18]),  # Pozitif SHAP'lar
        feature_values=np.array([-0.8, -0.7, -0.6, -0.9, -0.5]),  # Eşik altı
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xdss.format_decision_report(decision1))
    
    # Senaryo 2: CHECK durumu (orta risk)
    print("\n\n🟡 SENARYO 2: DİKKAT GEREKTİREN (CHECK bekleniyor)")
    decision2 = xdss.xdss_decision(
        pred_prob=0.65,
        shap_values=np.array([0.15, 0.12, 0.08, 0.05, 0.03]),
        feature_values=np.array([-0.3, -0.2, 0.1, 0.2, 0.3]),  # Karışık
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xdss.format_decision_report(decision2))
    
    # Senaryo 3: CONTINUE durumu (düşük risk)
    print("\n\n🟢 SENARYO 3: NORMAL DURUM (CONTINUE bekleniyor)")
    decision3 = xdss.xdss_decision(
        pred_prob=0.15,
        shap_values=np.array([0.02, 0.01, -0.03, -0.02, 0.01]),
        feature_values=np.array([0.2, 0.3, 0.4, 0.3, 0.5]),  # Normal
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xdss.format_decision_report(decision3))
    
    print("\n✓ XDSS Demo tamamlandı!")


if __name__ == "__main__":
    demo_xdss()