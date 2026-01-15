"""
================================================================================
XSM - EXPLAINABLE SECURITY MODULE
X-HARMONY Mimarisi - Güvenlik Katmanı
================================================================================

Bu modül, üretim sürecinde güvenlik anomalilerini tespit eder ve açıklar.

Güvenlik Kontrolleri:
    1. Anomali Tespiti (sensör değerleri)
    2. Model Güvenilirliği (tahmin tutarlılığı)
    3. SHAP Anomalileri (açıklama tutarlılığı)
    4. Drift Tespiti (veri dağılımı kayması)

Alert Seviyeleri:
    - CRITICAL: Acil müdahale gerekli
    - WARNING: İzleme gerekli
    - INFO: Bilgilendirme

Giriş:
    - Sensör değerleri
    - Model tahminleri
    - SHAP değerleri
    - Referans istatistikler

Çıkış:
    - Alert seviyesi
    - Anomali tipleri
    - Açıklamalar
    - Önerilen aksiyonlar
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class XSMModule:
    """
    Explainable Security Module (XSM) sınıfı.
    
    X-HARMONY mimarisinin güvenlik katmanını implement eder.
    Anomali tespiti, drift kontrolü ve model güvenilirliği kontrolü yapar.
    """
    
    def __init__(
        self, 
        reference_stats_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        XSM modülünü başlat.
        
        Args:
            reference_stats_path: Referans istatistikleri CSV yolu (opsiyonel)
            verbose: Detaylı çıktı için True
        """
        self.verbose = verbose
        self.reference_stats = None
        
        if reference_stats_path:
            self.reference_stats = self._load_reference_stats(reference_stats_path)
        
        self.security_thresholds = self._set_security_thresholds()
        self.alert_history = []
        
        if self.verbose:
            print("=" * 70)
            print("XSM MODÜLÜ BAŞLATILDI")
            print("=" * 70)
            if self.reference_stats is not None:
                print(f"✓ Referans istatistikler yüklendi: {len(self.reference_stats)} sensör")
            print(f"✓ Güvenlik eşikleri ayarlandı")
            print("=" * 70)
    
    def _load_reference_stats(self, stats_path: str) -> pd.DataFrame:
        """Referans istatistikleri yükle (training set'ten)."""
        return pd.read_csv(stats_path)
    
    def _set_security_thresholds(self) -> Dict[str, float]:
        """Güvenlik eşiklerini belirle."""
        return {
            # Anomali tespiti (z-score tabanlı)
            'zscore_critical': 4.0,   # z > 4.0 → CRITICAL
            'zscore_warning': 3.0,    # z > 3.0 → WARNING
            
            # SHAP anomalileri
            'shap_max_value': 2.0,    # Tek bir SHAP > 2.0 → anomali
            'shap_sparsity': 0.95,    # %95'ten fazla sıfır → şüpheli
            
            # Model güvenilirliği
            'confidence_min': 0.3,    # Confidence < 0.3 → belirsiz
            'prediction_flip': 0.1,   # Küçük değişiklikte tahmin değişimi → hassas
            
            # Drift tespiti (KL divergence)
            'kl_divergence_warning': 0.1,
            'kl_divergence_critical': 0.3,
            
            # Batch anomali oranları
            'batch_anomaly_critical': 0.2,  # %20+ anomali → CRITICAL
            'batch_anomaly_warning': 0.1    # %10+ anomali → WARNING
        }
    
    def detect_sensor_anomalies(
        self,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict]:
        """
        Sensör değerlerinde anomali tespit et.
        
        Z-score tabanlı anomali tespiti. Referans istatistikler varsa
        onları kullanır, yoksa basit outlier tespiti yapar.
        
        Args:
            feature_values: Sensör değerleri
            feature_names: Sensör isimleri
            
        Returns:
            anomalies: Liste of anomaly dicts
        """
        anomalies = []
        
        for i, (value, name) in enumerate(zip(feature_values, feature_names)):
            
            # NaN kontrolü
            if pd.isna(value):
                anomalies.append({
                    'type': 'MISSING_VALUE',
                    'sensor': name,
                    'value': None,
                    'severity': 'WARNING',
                    'explanation': f"Sensör {name} değeri eksik"
                })
                continue
            
            # Referans istatistiklerle karşılaştır
            if self.reference_stats is not None:
                sensor_stats = self.reference_stats[
                    self.reference_stats['sensor'] == name
                ]
                
                if not sensor_stats.empty:
                    mean = sensor_stats['mean'].values[0]
                    std = sensor_stats['std'].values[0]
                    
                    # Z-score hesapla
                    if std > 0:
                        z_score = abs((value - mean) / std)
                        
                        if z_score > self.security_thresholds['zscore_critical']:
                            anomalies.append({
                                'type': 'EXTREME_VALUE',
                                'sensor': name,
                                'value': float(value),
                                'z_score': float(z_score),
                                'severity': 'CRITICAL',
                                'explanation': (
                                    f"Sensör {name} aşırı sapma gösteriyor "
                                    f"(z={z_score:.2f}σ). "
                                    f"Değer: {value:.3f}, Beklenen: {mean:.3f}±{std:.3f}"
                                )
                            })
                        
                        elif z_score > self.security_thresholds['zscore_warning']:
                            anomalies.append({
                                'type': 'OUTLIER',
                                'sensor': name,
                                'value': float(value),
                                'z_score': float(z_score),
                                'severity': 'WARNING',
                                'explanation': (
                                    f"Sensör {name} normalden sapıyor "
                                    f"(z={z_score:.2f}σ)"
                                )
                            })
            
            # Basit outlier kontrolü (referans yoksa)
            else:
                # Çok büyük veya çok küçük değerler
                if abs(value) > 10:  # Scaled data için
                    anomalies.append({
                        'type': 'LARGE_VALUE',
                        'sensor': name,
                        'value': float(value),
                        'severity': 'WARNING',
                        'explanation': f"Sensör {name} beklenmedik büyük değer: {value:.3f}"
                    })
        
        return anomalies
    
    def detect_shap_anomalies(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict]:
        """
        SHAP değerlerinde anomali tespit et.
        
        Açıklama tutarlılığını kontrol eder:
        - Aşırı büyük SHAP değerleri
        - Aşırı sparse SHAP vektörleri
        - Beklenmedik SHAP dağılımları
        
        Args:
            shap_values: SHAP değerleri
            feature_names: Feature isimleri
            
        Returns:
            anomalies: SHAP anomalileri
        """
        anomalies = []
        
        # 1. Aşırı büyük SHAP değeri kontrolü
        max_shap = np.max(np.abs(shap_values))
        if max_shap > self.security_thresholds['shap_max_value']:
            max_idx = np.argmax(np.abs(shap_values))
            anomalies.append({
                'type': 'EXTREME_SHAP',
                'feature': feature_names[max_idx],
                'shap_value': float(shap_values[max_idx]),
                'severity': 'WARNING',
                'explanation': (
                    f"Feature {feature_names[max_idx]} aşırı yüksek SHAP değerine sahip: "
                    f"{shap_values[max_idx]:.3f}. Model bu feature'a normalden çok daha fazla "
                    f"önem veriyor olabilir."
                )
            })
        
        # 2. Sparsity kontrolü (çok fazla sıfır)
        n_near_zero = np.sum(np.abs(shap_values) < 0.01)
        sparsity = n_near_zero / len(shap_values)
        
        if sparsity > self.security_thresholds['shap_sparsity']:
            anomalies.append({
                'type': 'SPARSE_SHAP',
                'sparsity': float(sparsity),
                'severity': 'INFO',
                'explanation': (
                    f"SHAP vektörü çok sparse ({sparsity*100:.1f}% sıfıra yakın). "
                    f"Model sadece birkaç feature'a odaklanıyor olabilir."
                )
            })
        
        # 3. SHAP dağılımı kontrolü (normallik testi)
        # Çok skewed bir SHAP dağılımı şüpheli olabilir
        shap_skewness = stats.skew(shap_values)
        if abs(shap_skewness) > 2.0:
            anomalies.append({
                'type': 'SKEWED_SHAP',
                'skewness': float(shap_skewness),
                'severity': 'INFO',
                'explanation': (
                    f"SHAP dağılımı çok çarpık (skewness={shap_skewness:.2f}). "
                    f"Modelin açıklama yapısı dengesiz olabilir."
                )
            })
        
        return anomalies
    
    def check_model_confidence(
        self,
        pred_prob: float,
        shap_values: np.ndarray
    ) -> Optional[Dict]:
        """
        Model güvenilirliğini kontrol et.
        
        Model tahmininin güvenilirliğini SHAP tutarlılığı ile doğrular.
        
        Args:
            pred_prob: Fail olasılığı
            shap_values: SHAP değerleri
            
        Returns:
            alert: Güvenilirlik alerti (varsa)
        """
        # Tahmin belirsizliği (0.5'e yakınlık)
        uncertainty = 1 - abs(pred_prob - 0.5) * 2
        
        if uncertainty > (1 - self.security_thresholds['confidence_min']):
            # SHAP değerleri de belirsiz mi kontrol et
            shap_magnitude = np.sum(np.abs(shap_values))
            
            if shap_magnitude < 1.0:  # Çok düşük SHAP katkısı
                return {
                    'type': 'LOW_CONFIDENCE',
                    'pred_prob': float(pred_prob),
                    'uncertainty': float(uncertainty),
                    'shap_magnitude': float(shap_magnitude),
                    'severity': 'WARNING',
                    'explanation': (
                        f"Model belirsiz tahmin yapıyor (p={pred_prob:.3f}) ve "
                        f"SHAP açıklamaları zayıf (magnitude={shap_magnitude:.3f}). "
                        f"Bu tahmine güvenmek riskli olabilir."
                    )
                }
        
        return None
    
    def detect_drift(
        self,
        current_batch_stats: Dict,
        reference_stats: Dict
    ) -> List[Dict]:
        """
        Veri drift'i tespit et.
        
        Mevcut batch'in istatistiklerini referans ile karşılaştırır.
        Dağılım kayması (covariate shift) kontrolü yapar.
        
        Args:
            current_batch_stats: Mevcut batch istatistikleri
            reference_stats: Referans (training) istatistikleri
            
        Returns:
            drift_alerts: Drift alert listesi
        """
        drift_alerts = []
        
        # Ortalama ve std karşılaştırması
        for sensor in current_batch_stats.keys():
            if sensor in reference_stats:
                
                # Ortalama kayması
                curr_mean = current_batch_stats[sensor]['mean']
                ref_mean = reference_stats[sensor]['mean']
                ref_std = reference_stats[sensor]['std']
                
                if ref_std > 0:
                    mean_shift = abs(curr_mean - ref_mean) / ref_std
                    
                    if mean_shift > 2.0:
                        drift_alerts.append({
                            'type': 'MEAN_DRIFT',
                            'sensor': sensor,
                            'shift': float(mean_shift),
                            'severity': 'CRITICAL' if mean_shift > 3.0 else 'WARNING',
                            'explanation': (
                                f"Sensör {sensor} ortalaması kayıyor. "
                                f"Kayma: {mean_shift:.2f}σ. "
                                f"Mevcut: {curr_mean:.3f}, Referans: {ref_mean:.3f}"
                            )
                        })
                
                # Varyans kayması
                curr_std = current_batch_stats[sensor]['std']
                std_ratio = curr_std / ref_std if ref_std > 0 else 1.0
                
                if std_ratio > 2.0 or std_ratio < 0.5:
                    drift_alerts.append({
                        'type': 'VARIANCE_DRIFT',
                        'sensor': sensor,
                        'ratio': float(std_ratio),
                        'severity': 'WARNING',
                        'explanation': (
                            f"Sensör {sensor} varyansı değişti. "
                            f"Oran: {std_ratio:.2f}x"
                        )
                    })
        
        return drift_alerts
    
    def xsm_security_check(
        self,
        pred_prob: float,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        XSM ana güvenlik kontrolü.
        
        Tüm güvenlik kontrollerini yapar ve alert üretir.
        
        Args:
            pred_prob: Model tahmin olasılığı
            shap_values: SHAP değerleri
            feature_values: Sensör değerleri
            feature_names: Feature isimleri
            
        Returns:
            security_report: {
                'status': str,           # SAFE / WARNING / CRITICAL
                'alerts': list,          # Alert listesi
                'n_critical': int,       # Kritik alert sayısı
                'n_warning': int,        # Uyarı alert sayısı
                'recommendation': str,   # Önerilen aksiyon
                'timestamp': str
            }
        """
        all_alerts = []
        
        # 1. Sensör anomalileri
        sensor_anomalies = self.detect_sensor_anomalies(
            feature_values, feature_names
        )
        all_alerts.extend(sensor_anomalies)
        
        # 2. SHAP anomalileri
        shap_anomalies = self.detect_shap_anomalies(
            shap_values, feature_names
        )
        all_alerts.extend(shap_anomalies)
        
        # 3. Model güvenilirliği
        confidence_alert = self.check_model_confidence(
            pred_prob, shap_values
        )
        if confidence_alert:
            all_alerts.append(confidence_alert)
        
        # Alert istatistikleri
        n_critical = sum(1 for a in all_alerts if a['severity'] == 'CRITICAL')
        n_warning = sum(1 for a in all_alerts if a['severity'] == 'WARNING')
        n_info = sum(1 for a in all_alerts if a['severity'] == 'INFO')
        
        # Genel durum
        if n_critical > 0:
            status = 'CRITICAL'
            recommendation = (
                f"⛔ {n_critical} kritik güvenlik sorunu tespit edildi! "
                f"Üretimi durdurun ve anomalileri araştırın."
            )
        elif n_warning > 0:
            status = 'WARNING'
            recommendation = (
                f"⚠️ {n_warning} uyarı seviyesi anomali var. "
                f"Dikkatli izleme yapın."
            )
        else:
            status = 'SAFE'
            recommendation = "✅ Güvenlik kontrolleri normal. Devam edilebilir."
        
        security_report = {
            'status': status,
            'alerts': all_alerts,
            'n_critical': n_critical,
            'n_warning': n_warning,
            'n_info': n_info,
            'recommendation': recommendation,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Alert geçmişine ekle
        self.alert_history.append(security_report)
        
        return security_report
    
    def batch_security_check(
        self,
        pred_probs: np.ndarray,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[Dict], Dict]:
        """
        Batch (toplu) güvenlik kontrolü.
        
        Args:
            pred_probs: Fail olasılıkları (n_samples,)
            shap_values: SHAP matrisi (n_samples, n_features)
            feature_values: Sensör matrisi (n_samples, n_features)
            feature_names: Feature isimleri
            
        Returns:
            reports: Liste of security reports
            summary: Batch özeti
        """
        reports = []
        n_samples = len(pred_probs)
        
        if self.verbose:
            print(f"\n🔒 Batch güvenlik kontrolü başlatıldı: {n_samples} örnek")
        
        for i in range(n_samples):
            report = self.xsm_security_check(
                pred_probs[i],
                shap_values[i],
                feature_values[i],
                feature_names
            )
            report['sample_id'] = i
            reports.append(report)
        
        # Batch özeti
        n_critical = sum(1 for r in reports if r['status'] == 'CRITICAL')
        n_warning = sum(1 for r in reports if r['status'] == 'WARNING')
        n_safe = sum(1 for r in reports if r['status'] == 'SAFE')
        
        summary = {
            'n_samples': n_samples,
            'n_critical': n_critical,
            'n_warning': n_warning,
            'n_safe': n_safe,
            'critical_rate': n_critical / n_samples,
            'warning_rate': n_warning / n_samples,
            'safe_rate': n_safe / n_samples
        }
        
        if self.verbose:
            print(f"\n📊 Batch Güvenlik Özeti:")
            print(f"  🔴 CRITICAL: {n_critical} ({n_critical/n_samples*100:.1f}%)")
            print(f"  🟡 WARNING: {n_warning} ({n_warning/n_samples*100:.1f}%)")
            print(f"  🟢 SAFE: {n_safe} ({n_safe/n_samples*100:.1f}%)")
            
            # Batch-level anomali oranı kontrolü
            if summary['critical_rate'] > self.security_thresholds['batch_anomaly_critical']:
                print(f"\n  ⚠️ BATCH LEVEL ALERT: Kritik anomali oranı çok yüksek!")
            elif summary['warning_rate'] > self.security_thresholds['batch_anomaly_warning']:
                print(f"\n  ⚠️ BATCH LEVEL ALERT: Uyarı seviyesi anomali oranı yüksek!")
        
        return reports, summary
    
    def format_security_report(self, report: Dict) -> str:
        """
        Güvenlik raporunu formatla.
        
        Args:
            report: xsm_security_check() çıktısı
            
        Returns:
            Formatlanmış rapor string
        """
        output = []
        output.append("\n" + "=" * 70)
        output.append("XSM GÜVENLİK RAPORU")
        output.append("=" * 70)
        
        # Durum
        status_emoji = {
            'SAFE': '🟢',
            'WARNING': '🟡',
            'CRITICAL': '🔴'
        }
        emoji = status_emoji.get(report['status'], '❓')
        
        output.append(f"\n{emoji} DURUM: {report['status']}")
        output.append(f"🕐 Zaman: {report['timestamp']}")
        output.append(f"\n📊 Alert İstatistikleri:")
        output.append(f"  🔴 Kritik: {report['n_critical']}")
        output.append(f"  🟡 Uyarı: {report['n_warning']}")
        output.append(f"  ℹ️  Bilgi: {report['n_info']}")
        
        # Alertleri detaylı göster
        if report['alerts']:
            output.append(f"\n🚨 TESPİT EDİLEN ANOMALILER:")
            
            # Önce kritikler
            critical_alerts = [a for a in report['alerts'] if a['severity'] == 'CRITICAL']
            if critical_alerts:
                output.append("\n  🔴 KRİTİK:")
                for alert in critical_alerts:
                    output.append(f"    • {alert['type']}: {alert['explanation']}")
            
            # Sonra uyarılar
            warning_alerts = [a for a in report['alerts'] if a['severity'] == 'WARNING']
            if warning_alerts:
                output.append("\n  🟡 UYARI:")
                for alert in warning_alerts[:5]:  # Max 5 göster
                    output.append(f"    • {alert['type']}: {alert['explanation']}")
            
            # Bilgilendirme
            info_alerts = [a for a in report['alerts'] if a['severity'] == 'INFO']
            if info_alerts:
                output.append("\n  ℹ️  BİLGİ:")
                for alert in info_alerts[:3]:  # Max 3 göster
                    output.append(f"    • {alert['type']}: {alert['explanation']}")
        
        # Öneri
        output.append(f"\n💡 ÖNERİ:")
        output.append(f"  {report['recommendation']}")
        
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def export_alerts_to_csv(
        self, 
        reports: List[Dict], 
        output_path: str
    ) -> None:
        """
        Batch alert'lerini CSV olarak kaydet.
        
        Args:
            reports: batch_security_check() çıktısı
            output_path: CSV dosya yolu
        """
        records = []
        
        for report in reports:
            for alert in report['alerts']:
                record = {
                    'sample_id': report.get('sample_id', -1),
                    'status': report['status'],
                    'timestamp': report['timestamp'],
                    'alert_type': alert['type'],
                    'severity': alert['severity'],
                    'explanation': alert['explanation']
                }
                
                # Alert tipine göre ekstra bilgiler
                if 'sensor' in alert:
                    record['sensor'] = alert['sensor']
                if 'value' in alert:
                    record['value'] = alert['value']
                if 'z_score' in alert:
                    record['z_score'] = alert['z_score']
                if 'shap_value' in alert:
                    record['shap_value'] = alert['shap_value']
                
                records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            
            if self.verbose:
                print(f"✓ Alert'ler CSV'ye kaydedildi: {output_path}")
        else:
            if self.verbose:
                print("ℹ️  Kaydedilecek alert yok")
    
    def get_alert_statistics(self) -> Dict:
        """
        Geçmiş alert istatistiklerini getir.
        
        Returns:
            stats: Alert istatistikleri
        """
        if not self.alert_history:
            return {'message': 'Henüz alert geçmişi yok'}
        
        total_checks = len(self.alert_history)
        n_critical = sum(1 for r in self.alert_history if r['status'] == 'CRITICAL')
        n_warning = sum(1 for r in self.alert_history if r['status'] == 'WARNING')
        n_safe = sum(1 for r in self.alert_history if r['status'] == 'SAFE')
        
        # En sık görülen anomali tipleri
        all_alert_types = []
        for report in self.alert_history:
            all_alert_types.extend([a['type'] for a in report['alerts']])
        
        from collections import Counter
        most_common_alerts = Counter(all_alert_types).most_common(5)
        
        stats = {
            'total_checks': total_checks,
            'critical_count': n_critical,
            'warning_count': n_warning,
            'safe_count': n_safe,
            'critical_rate': n_critical / total_checks,
            'warning_rate': n_warning / total_checks,
            'safe_rate': n_safe / total_checks,
            'most_common_alert_types': most_common_alerts
        }
        
        return stats


# =============================================================================
# TESTİNG / DEMO
# =============================================================================

def demo_xsm():
    """XSM modülü demo."""
    
    print("\n" + "=" * 70)
    print("XSM MODÜLÜ DEMO")
    print("=" * 70)
    
    # XSM'yi başlat
    xsm = XSMModule(verbose=True)
    
    # Senaryo 1: Kritik sensör anomalisi
    print("\n\n🔴 SENARYO 1: KRİTİK SENSÖR ANOMALİSİ")
    report1 = xsm.xsm_security_check(
        pred_prob=0.75,
        shap_values=np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
        feature_values=np.array([15.0, -8.0, 12.0, 0.5, -0.3]),  # Aşırı değerler
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xsm.format_security_report(report1))
    
    # Senaryo 2: SHAP anomalisi
    print("\n\n🟡 SENARYO 2: SHAP ANOMALİSİ")
    report2 = xsm.xsm_security_check(
        pred_prob=0.5,
        shap_values=np.array([3.5, 0.01, 0.0, -0.01, 0.0]),  # Aşırı büyük SHAP
        feature_values=np.array([0.2, 0.3, 0.1, 0.4, 0.2]),  # Normal değerler
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xsm.format_security_report(report2))
    
    # Senaryo 3: Güvenli durum
    print("\n\n🟢 SENARYO 3: GÜVENLİ DURUM")
    report3 = xsm.xsm_security_check(
        pred_prob=0.2,
        shap_values=np.array([0.15, 0.12, -0.08, 0.05, -0.03]),
        feature_values=np.array([0.2, 0.3, 0.1, 0.4, 0.2]),
        feature_names=['419', '33', '59', '486', '213']
    )
    print(xsm.format_security_report(report3))
    
    # Alert istatistikleri
    print("\n\n📊 ALERT İSTATİSTİKLERİ")
    stats = xsm.get_alert_statistics()
    print(f"Toplam kontrol: {stats['total_checks']}")
    print(f"Kritik: {stats['critical_count']} ({stats['critical_rate']*100:.1f}%)")
    print(f"Uyarı: {stats['warning_count']} ({stats['warning_rate']*100:.1f}%)")
    print(f"Güvenli: {stats['safe_count']} ({stats['safe_rate']*100:.1f}%)")
    
    print("\n✓ XSM Demo tamamlandı!")


if __name__ == "__main__":
    demo_xsm()
