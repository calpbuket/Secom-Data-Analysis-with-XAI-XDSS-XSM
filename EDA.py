"""
================================================================================
SECOM VERİ SETİ - KAPSAMLI KEŞİFSEL VERİ ANALİZİ (EDA)
Yarı İletken Üretim Hatası Tahmini için Akademik Düzeyde Analiz
================================================================================

Bu kod, SECOM (Semiconductor Manufacturing) veri seti üzerinde kapsamlı bir
Exploratory Data Analysis (EDA) gerçekleştirmektedir.

Bölümler:
1. Veri Kümesi Tanıtımı
2. Eksik Veri Analizi
3. Tanımlayıcı İstatistikler
4. Hedef Değişken Analizi (Sınıf Dengesizliği)
5. Korelasyon ve İlişki Analizi
6. Aykırı Değer (Outlier) Analizi
7. Hedef Değişken ile Sensör İlişkisi
8. Sonuç ve Özet

Gerekli Kütüphaneler:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn

Yazar: EDA Analiz Scripti
Tarih: 2024
================================================================================
"""

# =============================================================================
# KÜTÜPHANELER
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# =============================================================================
# VERİ YÜKLEME
# =============================================================================
# Veri dosyasının yolunu kendi sisteminize göre güncelleyin
DATA_PATH = 'Downloads/Buket/uci-secom.csv'  
OUTPUT_DIR = './eda_outputs/'  

# Çıktı klasörünü oluştur
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Veri yükleme
print("Veri yükleniyor...")
df = pd.read_csv(DATA_PATH)

# Temel değişkenler
target_col = 'Pass/Fail'
feature_cols = [col for col in df.columns if col not in ['Time', 'Pass/Fail']]

print(f"✓ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")


# =============================================================================
# BÖLÜM 1: VERİ KÜMESİ TANITIMI
# =============================================================================
def bolum1_veri_tanitimi(df, target_col, feature_cols):
    """Veri kümesinin temel özelliklerini analiz eder"""
    
    print("\n" + "="*80)
    print("BÖLÜM 1: VERİ KÜMESİ TANITIMI")
    print("="*80)
    
    # Temel bilgiler
    print(f"\n📊 Veri Seti Genel Bilgileri:")
    print(f"   • Satır sayısı (Gözlem): {df.shape[0]:,}")
    print(f"   • Sütun sayısı (Değişken): {df.shape[1]:,}")
    print(f"   • Sensör sayısı: {len(feature_cols)}")
    
    # Hedef değişken analizi
    print(f"\n🎯 Hedef Değişken: '{target_col}'")
    print(f"   • Veri tipi: {df[target_col].dtype}")
    print(f"   • Benzersiz değerler: {df[target_col].unique()}")
    print(f"   • Hedef değişken türü: Binary (İkili Sınıflandırma)")
    print(f"   • -1: Hatasız üretim (Pass)")
    print(f"   •  1: Hatalı üretim (Fail)")
    
    # Sütun veri tipleri özeti
    print("\n📋 Veri Tiplerinin Özeti:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} sütun")
    
    # Detaylı veri tipleri tablosu
    print("\n📋 Sütun Veri Tipleri (İlk 30):")
    dtypes_df = pd.DataFrame({
        'Sütun Adı': df.columns[:30],
        'Veri Tipi': df.dtypes[:30].values
    })
    print(dtypes_df.to_string(index=False))
    
    # İlk 10 satır
    print("\n📋 İlk 10 Satır (İlk 10 Sütun):")
    print(df.iloc[:10, :10].to_string())
    
    # Bellek kullanımı
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 Bellek Kullanımı: {memory_usage:.2f} MB")
    
    # Özet istatistikler
    summary_stats = {
        'Satır Sayısı': df.shape[0],
        'Sütun Sayısı': df.shape[1],
        'Sensör Sayısı': len(feature_cols),
        'Bellek (MB)': round(memory_usage, 2)
    }
    
    return summary_stats


# =============================================================================
# BÖLÜM 2: EKSİK VERİ ANALİZİ
# =============================================================================
def bolum2_eksik_veri_analizi(df, target_col, feature_cols, output_dir):
    """Eksik veri analizi yapar ve görselleştirir"""
    
    print("\n" + "="*80)
    print("BÖLÜM 2: EKSİK VERİ ANALİZİ")
    print("="*80)
    
    # Eksik değer hesaplama
    missing_counts = df[feature_cols].isnull().sum()
    missing_percent = (df[feature_cols].isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Sütun': feature_cols,
        'Eksik Sayısı': missing_counts.values,
        'Eksik Oranı (%)': missing_percent.values
    }).sort_values('Eksik Oranı (%)', ascending=False)
    
    # Genel eksik veri istatistikleri
    total_cells = df[feature_cols].size
    total_missing = df[feature_cols].isnull().sum().sum()
    overall_missing_pct = (total_missing / total_cells) * 100
    
    print(f"\n📊 Genel Eksik Veri İstatistikleri:")
    print(f"   • Toplam hücre sayısı: {total_cells:,}")
    print(f"   • Toplam eksik değer: {total_missing:,}")
    print(f"   • Genel eksik oran: %{overall_missing_pct:.2f}")
    
    # Eksik değer içeren sütun sayıları
    cols_with_missing = (missing_counts > 0).sum()
    cols_no_missing = len(feature_cols) - cols_with_missing
    print(f"\n   • Eksik değer içeren sütun: {cols_with_missing}")
    print(f"   • Eksik değer içermeyen sütun: {cols_no_missing}")
    
    # En çok eksik içeren ilk 20 sütun
    print("\n📋 En Çok Eksik Değer İçeren İlk 20 Sütun:")
    top20_missing = missing_df[missing_df['Eksik Oranı (%)'] > 0].head(20)
    print(top20_missing.to_string(index=False))
    
    # Eksik değer kategorileri
    high_missing = missing_df[missing_df['Eksik Oranı (%)'] > 50]
    medium_missing = missing_df[(missing_df['Eksik Oranı (%)'] > 20) & (missing_df['Eksik Oranı (%)'] <= 50)]
    low_missing = missing_df[(missing_df['Eksik Oranı (%)'] > 0) & (missing_df['Eksik Oranı (%)'] <= 20)]
    
    print(f"\n📊 Eksik Veri Kategorileri:")
    print(f"   • Yüksek eksiklik (>50%): {len(high_missing)} sütun")
    print(f"   • Orta eksiklik (20-50%): {len(medium_missing)} sütun")
    print(f"   • Düşük eksiklik (0-20%): {len(low_missing)} sütun")
    
    # Hedef değişken ile eksik veri ilişkisi
    print("\n🔗 Eksik Veri ve Hedef Değişken İlişkisi:")
    df_temp = df.copy()
    df_temp['missing_count'] = df[feature_cols].isnull().sum(axis=1)
    
    # Sınıflara göre eksik veri ortalaması
    missing_by_class = df_temp.groupby(target_col)['missing_count'].agg(['mean', 'std', 'min', 'max'])
    print("\n   Sınıf Bazında Eksik Değer İstatistikleri:")
    print(f"   Pass (-1): Ortalama={missing_by_class.loc[-1, 'mean']:.2f}, Std={missing_by_class.loc[-1, 'std']:.2f}")
    print(f"   Fail (1):  Ortalama={missing_by_class.loc[1, 'mean']:.2f}, Std={missing_by_class.loc[1, 'std']:.2f}")
    
    # T-testi
    pass_missing = df_temp[df_temp[target_col] == -1]['missing_count']
    fail_missing = df_temp[df_temp[target_col] == 1]['missing_count']
    t_stat, p_value = stats.ttest_ind(pass_missing, fail_missing)
    print(f"\n   T-Test Sonucu: t={t_stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("   ⚠️ İstatistiksel olarak anlamlı fark VAR (p < 0.05)")
    else:
        print("   ✓ İstatistiksel olarak anlamlı fark YOK (p >= 0.05)")
    
    # MCAR/MAR/MNAR analizi
    print("\n" + "-"*60)
    print("📊 Eksik Veri Mekanizması Analizi (MCAR/MAR/MNAR)")
    print("-"*60)
    
    missing_pattern = df[feature_cols].isnull().sum(axis=1)
    print(f"\n   Satır bazında eksik değer aralığı: {missing_pattern.min()} - {missing_pattern.max()}")
    print(f"   Satır bazında ortalama eksik: {missing_pattern.mean():.2f}")
    
    # Korelasyon analizi
    sample_cols_with_missing = missing_df[missing_df['Eksik Oranı (%)'] > 5]['Sütun'].head(10).tolist()
    if len(sample_cols_with_missing) > 1:
        missing_indicator = df[sample_cols_with_missing].isnull().astype(int)
        missing_corr = missing_indicator.corr().mean().mean()
        print(f"\n   Eksik değer göstergeleri arası ortalama korelasyon: {missing_corr:.4f}")
    
    print("""
🔬 Bilimsel Yorum - Eksik Veri Yapısı:

   1. MCAR (Missing Completely At Random) DEĞİL:
      • Bazı sütunlarda %40+'ın üzerinde eksiklik bulunması
      • Eksik değerlerin belirli sütunlarda yoğunlaşması
      MCAR varsayımını desteklememektedir.
   
   2. MAR (Missing At Random) olasılığı YÜKSEK:
      • Eksik değerlerin gözlemlenen diğer değişkenlerle ilişkili olması
      • Sensör arızaları veya ölçüm koşullarına bağlı eksiklik
      MAR mekanizmasını desteklemektedir.
   
   3. MNAR (Missing Not At Random) İHTİMALİ:
      • Bazı sensörlerin limit değerlerinde kayıt yapamaması
      • Üretim hatası durumunda sensör çalışmaması
      MNAR olasılığını düşündürmektedir.
   
   ➤ Sonuç: Bu veri seti muhtemelen MAR veya karma bir mekanizma göstermektedir.
""")
    
    print("""
📋 Önerilen Eksik Veri Stratejileri:

   1. SILME (Deletion):
      ✓ %50+'dan fazla eksik içeren sütunlar silinebilir
      ✗ Satır silme önerilmez - veri kaybı çok yüksek olur
      
   2. MEAN/MEDIAN IMPUTATION:
      ✓ Basit ve hızlı uygulama
      ✗ Varyansı küçültür, korelasyonları bozar
      
   3. KNN IMPUTATION:
      ✓ Gözlemler arası benzerliği kullanır
      ✓ Multivariate yapıyı korur
      ✗ Yüksek boyutlu verilerde hesaplama maliyeti
      
   4. ITERATIVE IMPUTER (MICE):
      ✓ MAR varsayımı altında en iyi performans
      ✓ Değişkenler arası ilişkileri modelleyerek impute eder
      ✗ Hesaplama süresi uzun
      
   ➤ ÖNERİ: %40+ eksik sütunları silmek, kalan için IterativeImputer
""")
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Eksik değer dağılımı histogram
    ax1 = axes[0, 0]
    missing_pct_nonzero = missing_df[missing_df['Eksik Oranı (%)'] > 0]['Eksik Oranı (%)']
    ax1.hist(missing_pct_nonzero, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Eksik Değer Oranı (%)')
    ax1.set_ylabel('Sütun Sayısı')
    ax1.set_title('Eksik Değer Oranlarının Dağılımı')
    ax1.axvline(x=20, color='orange', linestyle='--', label='%20 Eşiği')
    ax1.axvline(x=50, color='red', linestyle='--', label='%50 Eşiği')
    ax1.legend()
    
    # 2. En çok eksik içeren 20 sütun
    ax2 = axes[0, 1]
    top20 = missing_df.head(20)
    ax2.barh(range(len(top20)), top20['Eksik Oranı (%)'], color='#3498db', edgecolor='black')
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(top20['Sütun'])
    ax2.set_xlabel('Eksik Değer Oranı (%)')
    ax2.set_title('En Çok Eksik Değer İçeren 20 Sütun')
    ax2.invert_yaxis()
    
    # 3. Sınıf bazında eksik veri boxplot
    ax3 = axes[1, 0]
    data_boxplot = [pass_missing, fail_missing]
    bp = ax3.boxplot(data_boxplot, patch_artist=True, labels=['Pass (-1)', 'Fail (1)'])
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Satır Başına Eksik Değer Sayısı')
    ax3.set_title('Hedef Değişkene Göre Eksik Veri Dağılımı')
    
    # 4. Satır başına eksik değer dağılımı
    ax4 = axes[1, 1]
    ax4.hist(df_temp['missing_count'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Satır Başına Eksik Değer Sayısı')
    ax4.set_ylabel('Gözlem Sayısı')
    ax4.set_title('Gözlemlerdeki Eksik Değer Dağılımı')
    ax4.axvline(x=df_temp['missing_count'].mean(), color='red', linestyle='--', 
                label=f'Ortalama: {df_temp["missing_count"].mean():.1f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig1_eksik_veri_analizi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig1_eksik_veri_analizi.png")
    
    # Özet istatistikler
    missing_stats = {
        'Toplam Eksik': total_missing,
        'Genel Eksik Oranı': overall_missing_pct,
        'Eksik İçeren Sütun': cols_with_missing,
        'Yüksek Eksik Sütun (>50%)': len(high_missing),
        'T-test p-value': p_value
    }
    
    return missing_stats, missing_df


# =============================================================================
# BÖLÜM 3: TANIMLAYICI İSTATİSTİKLER
# =============================================================================
def bolum3_tanimlayici_istatistikler(df, feature_cols, output_dir):
    """Her sensör için tanımlayıcı istatistikleri hesaplar"""
    
    print("\n" + "="*80)
    print("BÖLÜM 3: TANIMLAYICI İSTATİSTİKLER")
    print("="*80)
    
    # Temel istatistikler hesaplama
    stats_list = []
    for col in feature_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats_list.append({
                'Sütun': col,
                'N': len(col_data),
                'Ortalama': col_data.mean(),
                'Medyan': col_data.median(),
                'Std': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Skewness': skew(col_data) if col_data.std() > 0 else np.nan,
                'Kurtosis': kurtosis(col_data) if col_data.std() > 0 else np.nan
            })
    
    stats_df = pd.DataFrame(stats_list)
    
    print("\n📊 Tanımlayıcı İstatistikler Özeti (İlk 20 Sensör):")
    display_cols = ['Sütun', 'N', 'Ortalama', 'Medyan', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis']
    print(stats_df[display_cols].head(20).to_string(index=False))
    
    # Skewness ve Kurtosis analizi
    valid_skew = stats_df['Skewness'].dropna()
    valid_kurt = stats_df['Kurtosis'].dropna()
    
    normal_dist = len(valid_skew[abs(valid_skew) < 0.5])
    mild_skew = len(valid_skew[(abs(valid_skew) >= 0.5) & (abs(valid_skew) < 1)])
    high_skew = len(valid_skew[abs(valid_skew) >= 1])
    
    print(f"\n📈 Dağılım Özellikleri Analizi:")
    print(f"   • Normal dağılım gösteren sütunlar (|skew| < 0.5): {normal_dist}")
    print(f"   • Hafif çarpık dağılımlar (0.5 <= |skew| < 1): {mild_skew}")
    print(f"   • Yüksek çarpık dağılımlar (|skew| >= 1): {high_skew}")
    
    # En çarpık dağılımlar
    print("\n📋 En Yüksek Çarpıklık Gösteren 10 Sütun:")
    top_skew = stats_df.dropna(subset=['Skewness']).nlargest(10, 'Skewness')[['Sütun', 'Ortalama', 'Medyan', 'Skewness', 'Kurtosis']]
    print(top_skew.to_string(index=False))
    
    # Kurtosis yorumu
    leptokurtic = len(valid_kurt[valid_kurt > 3])
    mesokurtic = len(valid_kurt[(valid_kurt >= -3) & (valid_kurt <= 3)])
    platykurtic = len(valid_kurt[valid_kurt < -3])
    
    print(f"\n📊 Kurtosis (Basıklık) Analizi:")
    print(f"   • Leptokurtik (sivri, >3): {leptokurtic} sütun")
    print(f"   • Mesokurtik (normal, -3 ile 3): {mesokurtic} sütun")
    print(f"   • Platikurtik (basık, <-3): {platykurtic} sütun")
    
    # Aykırı değer potansiyeli
    print("\n🔍 Aykırı Değer Potansiyeli Yüksek Sütunlar (Kurtosis > 10):")
    outlier_potential = stats_df[stats_df['Kurtosis'] > 10][['Sütun', 'Min', 'Max', 'Skewness', 'Kurtosis']].head(10)
    print(outlier_potential.to_string(index=False))
    
    print("""
📝 YORUM: Tanımlayıcı İstatistikler
   
   • Sensör verilerinin büyük çoğunluğu normal dağılım göstermemektedir.
   • Yüksek skewness değerleri (>5) logaritmik dönüşüm ihtiyacını gösterir.
   • Yüksek kurtosis değerleri (>10) uç değerlerin varlığına işaret eder.
   • Ortalama-medyan farkları dağılımların simetrik olmadığını gösterir.
   • Ölçeklendirme (StandardScaler/RobustScaler) kesinlikle gereklidir.
""")
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Skewness dağılımı
    ax1 = axes[0, 0]
    skew_clipped = valid_skew.clip(-10, 10)
    ax1.hist(skew_clipped, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', label='Normal (skew=0)')
    ax1.axvline(x=-0.5, color='orange', linestyle=':', label='±0.5 sınırı')
    ax1.axvline(x=0.5, color='orange', linestyle=':')
    ax1.set_xlabel('Skewness (Çarpıklık)')
    ax1.set_ylabel('Sütun Sayısı')
    ax1.set_title('Sensörlerin Çarpıklık Dağılımı')
    ax1.legend()
    
    # 2. Kurtosis dağılımı
    ax2 = axes[0, 1]
    kurt_clipped = valid_kurt.clip(-5, 50)
    ax2.hist(kurt_clipped, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='Normal (kurtosis=0)')
    ax2.set_xlabel('Kurtosis (Basıklık)')
    ax2.set_ylabel('Sütun Sayısı')
    ax2.set_title('Sensörlerin Basıklık Dağılımı')
    ax2.legend()
    
    # 3. Örnek normal dağılım gösteren sensör
    normal_sensors = stats_df[(abs(stats_df['Skewness']) < 0.5) & (abs(stats_df['Kurtosis']) < 3)]['Sütun'].head(3).tolist()
    ax3 = axes[1, 0]
    if normal_sensors:
        for sensor in normal_sensors[:3]:
            data = df[sensor].dropna()
            ax3.hist(data, bins=30, alpha=0.5, label=f'Sensör {sensor}', edgecolor='black')
        ax3.set_xlabel('Değer')
        ax3.set_ylabel('Frekans')
        ax3.set_title('Normal Dağılıma Yakın Sensörler')
        ax3.legend()
    
    # 4. Örnek çarpık dağılım gösteren sensör
    skewed_sensors = stats_df[stats_df['Skewness'] > 5]['Sütun'].head(3).tolist()
    ax4 = axes[1, 1]
    if skewed_sensors:
        for sensor in skewed_sensors[:3]:
            data = df[sensor].dropna()
            ax4.hist(data, bins=30, alpha=0.5, label=f'Sensör {sensor}', edgecolor='black')
        ax4.set_xlabel('Değer')
        ax4.set_ylabel('Frekans')
        ax4.set_title('Yüksek Çarpıklık Gösteren Sensörler')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig2_tanimlayici_istatistikler.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig2_tanimlayici_istatistikler.png")
    
    return stats_df


# =============================================================================
# BÖLÜM 4: HEDEF DEĞİŞKEN ANALİZİ (SINIF DENGESİZLİĞİ)
# =============================================================================
def bolum4_sinif_dengesizligi(df, target_col, output_dir):
    """Hedef değişken ve sınıf dengesizliği analizi"""
    
    print("\n" + "="*80)
    print("BÖLÜM 4: HEDEF DEĞİŞKEN ANALİZİ (SINIF DENGESİZLİĞİ)")
    print("="*80)
    
    # Sınıf dağılımı
    class_counts = df[target_col].value_counts()
    class_percent = df[target_col].value_counts(normalize=True) * 100
    
    print(f"\n📊 Sınıf Dağılımı:")
    print(f"   • Pass (-1): {class_counts[-1]:,} gözlem (%{class_percent[-1]:.2f})")
    print(f"   • Fail (1):  {class_counts[1]:,} gözlem (%{class_percent[1]:.2f})")
    
    # Dengesizlik oranı
    imbalance_ratio = class_counts[-1] / class_counts[1]
    print(f"\n📊 Dengesizlik Oranı: {imbalance_ratio:.2f}:1 (Pass:Fail)")
    
    print("""
⚠️ SINIF DENGESİZLİĞİ SORUNU:
   
   Bu veri setinde ciddi bir sınıf dengesizliği bulunmaktadır:
   • Hatalı ürünler (Fail) toplam verinin sadece ~%6'sını oluşturur
   • Bu tür dengesizlik, modelin çoğunluk sınıfına (Pass) aşırı öğrenmesine neden olur
""")
    
    print("""
📊 METRİK SEÇİMİ AÇIKLAMASI:

   1. ACCURACY (Doğruluk) NEDEN YETERSİZ?
      • Tüm gözlemleri "Pass" tahmin eden bir model %93+ accuracy elde eder
      • Bu yanıltıcıdır çünkü hiçbir hatalı ürün tespit edilemez
      • Üretim hattında kaçırılan her hatalı ürün maliyetli sonuçlar doğurur
   
   2. PRECISION (Kesinlik):
      • Hatalı tahmin edilenlerin ne kadarı gerçekten hatalı?
      • Yanlış alarm (false positive) maliyetini ölçer
   
   3. RECALL (Duyarlılık/Sensitivity):
      • Gerçek hatalı ürünlerin ne kadarı yakalandı?
      • Kaçırılan hata (false negative) maliyetini ölçer
   
   4. F1-SCORE:
      • Precision ve Recall'un harmonik ortalaması
      • Dengesiz sınıflarda tek bir metrik olarak idealdir
   
   5. ROC-AUC / PR-AUC:
      • Sınıf oranlarından bağımsız performans ölçümü
      • Dengesiz verilerde PR-AUC daha bilgilendirici
""")
    
    print("""
🔧 ÖNERİLEN DENGESİZLİK ÇÖZÜM STRATEJİLERİ:

   1. RESAMPLING TEKNİKLERİ:
      • SMOTE (Synthetic Minority Over-sampling)
      • ADASYN (Adaptive Synthetic Sampling)
      • Random Undersampling
      
   2. SINIF AĞIRLIKLANDIRMA:
      • class_weight='balanced' parametresi
      
   3. ENSEMBLE YÖNTEMLER:
      • BalancedRandomForest
      • EasyEnsemble
      
   4. THRESHOLD OPTİMİZASYONU:
      • Precision-Recall eğrisi ile optimal threshold
""")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Bar plot
    ax1 = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Pass (-1)', 'Fail (1)'], [class_counts[-1], class_counts[1]], 
                   color=colors, edgecolor='black')
    ax1.set_ylabel('Gözlem Sayısı')
    ax1.set_title('Hedef Değişken Sınıf Dağılımı')
    for bar, count, pct in zip(bars, [class_counts[-1], class_counts[1]], 
                               [class_percent[-1], class_percent[1]]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                 f'{count:,}\n(%{pct:.1f})', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')
    
    # 2. Pie chart
    ax2 = axes[1]
    explode = (0, 0.1)
    ax2.pie([class_counts[-1], class_counts[1]], labels=['Pass (-1)', 'Fail (1)'], 
            autopct='%1.1f%%', colors=colors, explode=explode, shadow=True, startangle=90)
    ax2.set_title('Sınıf Oranları')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig3_sinif_dengesizligi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig3_sinif_dengesizligi.png")
    
    class_stats = {
        'Pass Count': class_counts[-1],
        'Fail Count': class_counts[1],
        'Pass Percent': class_percent[-1],
        'Fail Percent': class_percent[1],
        'Imbalance Ratio': imbalance_ratio
    }
    
    return class_stats


# =============================================================================
# BÖLÜM 5: KORELASYON VE İLİŞKİ ANALİZİ
# =============================================================================
def bolum5_korelasyon_analizi(df, target_col, feature_cols, output_dir):
    """Korelasyon matrisi ve multicollinearity analizi"""
    
    print("\n" + "="*80)
    print("BÖLÜM 5: KORELASYON VE İLİŞKİ ANALİZİ")
    print("="*80)
    
    # Eksik değerleri geçici olarak doldur
    df_temp = df[feature_cols].fillna(df[feature_cols].median())
    
    # Korelasyon matrisi hesaplama
    print("\n📊 Korelasyon Matrisi Hesaplanıyor...")
    corr_matrix = df_temp.corr()
    
    # En yüksek korelasyonlar
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if pd.notna(upper_tri.loc[idx, col]):
                corr_pairs.append({
                    'Özellik 1': idx,
                    'Özellik 2': col,
                    'Korelasyon': upper_tri.loc[idx, col]
                })
    
    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['Abs_Corr'] = abs(corr_pairs_df['Korelasyon'])
    corr_pairs_df = corr_pairs_df.sort_values('Abs_Corr', ascending=False)
    
    print("\n📋 En Yüksek Korelasyona Sahip İlk 20 Özellik Çifti:")
    top20_corr = corr_pairs_df.head(20)[['Özellik 1', 'Özellik 2', 'Korelasyon']]
    print(top20_corr.to_string(index=False))
    
    # Yüksek korelasyon istatistikleri
    very_high_corr = len(corr_pairs_df[corr_pairs_df['Abs_Corr'] > 0.95])
    high_corr = len(corr_pairs_df[(corr_pairs_df['Abs_Corr'] > 0.8) & (corr_pairs_df['Abs_Corr'] <= 0.95)])
    moderate_corr = len(corr_pairs_df[(corr_pairs_df['Abs_Corr'] > 0.5) & (corr_pairs_df['Abs_Corr'] <= 0.8)])
    
    print(f"\n📊 Korelasyon Düzeyleri:")
    print(f"   • Çok yüksek korelasyon (|r| > 0.95): {very_high_corr:,} çift")
    print(f"   • Yüksek korelasyon (0.8 < |r| <= 0.95): {high_corr:,} çift")
    print(f"   • Orta korelasyon (0.5 < |r| <= 0.8): {moderate_corr:,} çift")
    
    # Hedef değişken ile korelasyonlar
    df_temp['target'] = df[target_col]
    target_corr = df_temp.corr()['target'].drop('target').sort_values(key=abs, ascending=False)
    
    print("\n📋 Hedef Değişken ile En Yüksek Korelasyona Sahip 20 Özellik:")
    target_corr_df = pd.DataFrame({
        'Özellik': target_corr.head(20).index,
        'Korelasyon': target_corr.head(20).values
    })
    print(target_corr_df.to_string(index=False))
    
    print("""
⚠️ MULTICOLLINEARITY (ÇOKLU DOĞRUSALLLIK) RİSKİ:

   Veri setinde ciddi multicollinearity problemi bulunmaktadır:
   
   SORUNLAR:
   • Yüzlerce özellik çifti arasında r > 0.95 korelasyon
   • Bu, özelliklerin birbirinin kopyası veya türevi olduğunu gösterir
   • Regresyon modellerinde katsayı tahminlerini dengesizleştirir
   
   NEDENLER:
   • Aynı sensörün farklı zaman dilimlerindeki ölçümleri
   • Türetilmiş özellikler (ör: ortalama, toplam)
   • Fiziksel olarak ilişkili sensörler
""")
    
    print("""
🔧 PCA VE FEATURE SELECTION İHTİYACI:

   ✓ PCA (Principal Component Analysis) KESİNLİKLE ÖNERİLİR:
   • 590 özellik çok yüksek boyutluluk
   • Önerilen: %95 varyans açıklayan bileşenler (~50-100)
   
   ✓ FEATURE SELECTION STRATEJİLERİ:
   • Variance Threshold: Düşük varyanslı sütunları kaldır
   • Korelasyon Bazlı Eleme: r > 0.95 olan çiftlerden birini kaldır
   • Tree-based Feature Importance
   • LASSO Regularization
""")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Korelasyon dağılımı
    ax1 = axes[0]
    all_corrs = upper_tri.values.flatten()
    all_corrs = all_corrs[~np.isnan(all_corrs)]
    ax1.hist(all_corrs, bins=100, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.8, color='red', linestyle='--', label='|r|=0.8')
    ax1.axvline(x=-0.8, color='red', linestyle='--')
    ax1.axvline(x=0.95, color='darkred', linestyle=':', label='|r|=0.95')
    ax1.axvline(x=-0.95, color='darkred', linestyle=':')
    ax1.set_xlabel('Korelasyon Katsayısı (r)')
    ax1.set_ylabel('Özellik Çifti Sayısı')
    ax1.set_title('Tüm Özellik Çiftlerinin Korelasyon Dağılımı')
    ax1.legend()
    
    # 2. Hedef ile korelasyonlar
    ax2 = axes[1]
    top_target_corr = target_corr.head(15)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_target_corr.values]
    ax2.barh(range(len(top_target_corr)), top_target_corr.values, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_target_corr)))
    ax2.set_yticklabels(top_target_corr.index)
    ax2.set_xlabel('Korelasyon Katsayısı')
    ax2.set_title('Hedef Değişken ile En Yüksek Korelasyonlu Özellikler')
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig4_korelasyon_analizi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig4_korelasyon_analizi.png")
    
    # Heatmap (en önemli 20 özellik)
    top_features = target_corr.head(20).index.tolist()
    small_corr = df_temp[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(small_corr, dtype=bool))
    sns.heatmap(small_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax, annot_kws={'size': 8})
    ax.set_title('Hedef ile En Korele 20 Özelliğin Korelasyon Matrisi')
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig5_korelasyon_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Görsel kaydedildi: {output_dir}fig5_korelasyon_heatmap.png")
    
    corr_stats = {
        'Very High Corr (>0.95)': very_high_corr,
        'High Corr (0.8-0.95)': high_corr,
        'Moderate Corr (0.5-0.8)': moderate_corr,
        'Max Target Corr': target_corr.iloc[0],
        'Top Correlated Feature': target_corr.index[0]
    }
    
    return corr_stats, target_corr


# =============================================================================
# BÖLÜM 6: AYKIRI DEĞER (OUTLIER) ANALİZİ
# =============================================================================
def bolum6_aykiri_deger_analizi(df, feature_cols, output_dir):
    """IQR ve Z-score ile aykırı değer analizi"""
    
    print("\n" + "="*80)
    print("BÖLÜM 6: AYKIRI DEĞER (OUTLIER) ANALİZİ")
    print("="*80)
    
    # IQR yöntemi ile aykırı değer tespiti
    def count_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        return outliers, lower, upper
    
    # Her sütun için aykırı değer analizi
    outlier_stats = []
    for col in feature_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0 and col_data.std() > 0:
            iqr_count, lower, upper = count_outliers_iqr(col_data)
            outlier_stats.append({
                'Sütun': col,
                'Gözlem': len(col_data),
                'IQR_Outlier': iqr_count,
                'IQR_Oran(%)': (iqr_count / len(col_data)) * 100
            })
    
    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df = outlier_df.sort_values('IQR_Oran(%)', ascending=False)
    
    print("\n📊 Aykırı Değer İstatistikleri (IQR Yöntemi):")
    print(f"   • Toplam analiz edilen sütun: {len(outlier_df)}")
    print(f"   • Aykırı değer içeren sütun: {(outlier_df['IQR_Outlier'] > 0).sum()}")
    
    # En çok aykırı değer içeren sütunlar
    print("\n📋 En Çok Aykırı Değer İçeren İlk 20 Sütun (IQR):")
    top20_outlier = outlier_df.head(20)[['Sütun', 'Gözlem', 'IQR_Outlier', 'IQR_Oran(%)']]
    print(top20_outlier.to_string(index=False))
    
    # Kategorizasyon
    high_outlier = len(outlier_df[outlier_df['IQR_Oran(%)'] > 10])
    medium_outlier = len(outlier_df[(outlier_df['IQR_Oran(%)'] > 5) & (outlier_df['IQR_Oran(%)'] <= 10)])
    low_outlier = len(outlier_df[(outlier_df['IQR_Oran(%)'] > 0) & (outlier_df['IQR_Oran(%)'] <= 5)])
    
    print(f"\n📊 Aykırı Değer Kategorileri:")
    print(f"   • Yüksek aykırılık (>%10): {high_outlier} sütun")
    print(f"   • Orta aykırılık (%5-%10): {medium_outlier} sütun")
    print(f"   • Düşük aykırılık (<%5): {low_outlier} sütun")
    
    avg_outlier = outlier_df['IQR_Oran(%)'].mean()
    max_outlier = outlier_df['IQR_Oran(%)'].max()
    
    print(f"\n   • Ortalama aykırı değer oranı: %{avg_outlier:.2f}")
    print(f"   • Maksimum aykırı değer oranı: %{max_outlier:.2f}")
    
    print("""
🔬 AYKIRI DEĞERLERİN OLASI NEDENLERİ:

   1. SENSÖR HATASI:
      • Kalibrasyon sorunları
      • Sensör arızası veya bozulması
      • İletişim hatası
   
   2. ÖLÇÜM ARIZASI:
      • Geçici elektrik kesintileri
      • Ortam koşullarındaki ani değişimler
   
   3. GERÇEK ÜRETİM PROBLEMİ:
      • Anormal üretim koşulları
      • Ham madde kalite sapmaları
      • Bu değerler önemli bilgi taşıyabilir!
""")
    
    print("""
⚠️ AYKIRI DEĞERLERİ SİLMENİN RİSKLERİ:

   1. BİLGİ KAYBI:
      • Aykırı değerler üretim hatası sinyali olabilir
      • Özellikle Fail sınıfı için kritik özellikler silinebilir
   
   2. ÖNERİLEN YAKLAŞIMLAR:
      • Winsorization (%1-99 percentile)
      • RobustScaler kullanımı
      • Tree-based modeller (aykırı değerlere dayanıklı)
      • Aykırı değer göstergesi (flag) yeni özellik olarak
""")
    
    # Görselleştirme - Boxplotlar
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    top_outlier_cols = outlier_df.head(8)['Sütun'].tolist()
    
    for idx, ax in enumerate(axes.flatten()):
        cols_to_plot = top_outlier_cols[idx*2:(idx+1)*2]
        if cols_to_plot:
            data_to_plot = [df[col].dropna() for col in cols_to_plot]
            bp = ax.boxplot(data_to_plot, patch_artist=True, labels=cols_to_plot)
            colors = ['#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors[:len(cols_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(f'Sensör {cols_to_plot[0]} ve {cols_to_plot[1]} - Boxplot')
            ax.set_ylabel('Değer')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig6_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig6_boxplots.png")
    
    # Aykırı değer dağılımı
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(outlier_df['IQR_Oran(%)'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax.axvline(x=5, color='orange', linestyle='--', label='%5 eşiği')
    ax.axvline(x=10, color='red', linestyle='--', label='%10 eşiği')
    ax.set_xlabel('Aykırı Değer Oranı (%)')
    ax.set_ylabel('Sütun Sayısı')
    ax.set_title('Tüm Sensörlerdeki Aykırı Değer Oranları Dağılımı')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig7_outlier_dagilim.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Görsel kaydedildi: {output_dir}fig7_outlier_dagilim.png")
    
    outlier_summary = {
        'Avg Outlier Rate': avg_outlier,
        'Max Outlier Rate': max_outlier,
        'High Outlier Cols (>10%)': high_outlier,
        'Medium Outlier Cols (5-10%)': medium_outlier
    }
    
    return outlier_summary, outlier_df


# =============================================================================
# BÖLÜM 7: HEDEF DEĞİŞKEN İLE SENSÖR İLİŞKİSİ
# =============================================================================
def bolum7_hedef_sensor_iliskisi(df, target_col, feature_cols, target_corr, output_dir):
    """Hedef değişken ile en ilişkili sensörlerin detaylı analizi"""
    
    print("\n" + "="*80)
    print("BÖLÜM 7: HEDEF DEĞİŞKEN İLE SENSÖR İLİŞKİSİ")
    print("="*80)
    
    # En önemli 10 sensör
    important_sensors = target_corr.head(10).index.tolist()
    
    print(f"\n📋 Hedef Değişken ile En İlişkili 10 Sensör:")
    for i, sensor in enumerate(important_sensors, 1):
        corr_val = target_corr[sensor]
        print(f"   {i}. Sensör {sensor}: r = {corr_val:.4f}")
    
    # Pass ve Fail grupları için istatistikler
    print("\n📊 Önemli Sensörlerin Sınıf Bazında İstatistikleri:")
    
    sensor_analysis = []
    for sensor in important_sensors[:5]:
        pass_data = df[df[target_col] == -1][sensor].dropna()
        fail_data = df[df[target_col] == 1][sensor].dropna()
        
        print(f"\n   Sensör {sensor}:")
        print(f"      Pass (-1): Mean={pass_data.mean():.4f}, Std={pass_data.std():.4f}")
        print(f"      Fail (1):  Mean={fail_data.mean():.4f}, Std={fail_data.std():.4f}")
        
        # T-test
        t_stat, p_val = stats.ttest_ind(pass_data, fail_data)
        print(f"      T-test: t={t_stat:.3f}, p={p_val:.4f}", end="")
        if p_val < 0.05:
            print(" ✓ Anlamlı")
        else:
            print(" ✗ Anlamsız")
        
        sensor_analysis.append({
            'Sensor': sensor,
            'Pass Mean': pass_data.mean(),
            'Fail Mean': fail_data.mean(),
            't-stat': t_stat,
            'p-value': p_val
        })
    
    print("""
🏭 OPERASYONEL YORUM:

   Bu analiz sonuçları üretim sürecinde şu anlamları taşımaktadır:
   
   1. KRİTİK SENSÖRLER:
      • En yüksek korelasyonlu sensörler kalite ile doğrudan ilişkili
      • Gerçek zamanlı izleme önceliği bu sensörlere verilmeli
   
   2. ERKEN UYARI SİSTEMİ:
      • Pass ve Fail grupları arasında anlamlı fark gösteren sensörler
      • Threshold değerler belirlenerek alarm sistemi kurulabilir
   
   3. MALİYET ETKİSİ:
      • Hatalı ürün tespiti erken yapılabilir
      • Hurda ve yeniden işleme maliyetleri azaltılabilir
""")
    
    # Görselleştirme - Density plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(important_sensors[:6]):
        ax = axes[idx]
        
        pass_data = df[df[target_col] == -1][sensor].dropna()
        fail_data = df[df[target_col] == 1][sensor].dropna()
        
        if len(pass_data) > 1:
            pass_data.plot(kind='kde', ax=ax, color='#2ecc71', label='Pass (-1)', linewidth=2)
        if len(fail_data) > 1:
            fail_data.plot(kind='kde', ax=ax, color='#e74c3c', label='Fail (1)', linewidth=2)
        
        ax.set_xlabel('Sensör Değeri')
        ax.set_ylabel('Yoğunluk')
        ax.set_title(f'Sensör {sensor} - Sınıf Dağılımları\n(r = {target_corr[sensor]:.4f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig8_density_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Görsel kaydedildi: {output_dir}fig8_density_plots.png")
    
    # Violin plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(important_sensors[:6]):
        ax = axes[idx]
        
        data_pass = df[df[target_col] == -1][sensor].dropna()
        data_fail = df[df[target_col] == 1][sensor].dropna()
        
        plot_data = pd.DataFrame({
            'Değer': pd.concat([data_pass, data_fail]),
            'Sınıf': ['Pass']*len(data_pass) + ['Fail']*len(data_fail)
        })
        
        sns.violinplot(x='Sınıf', y='Değer', data=plot_data, ax=ax, 
                       palette={'Pass': '#2ecc71', 'Fail': '#e74c3c'})
        ax.set_title(f'Sensör {sensor} - Violin Plot')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig9_violin_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Görsel kaydedildi: {output_dir}fig9_violin_plots.png")
    
    # Boxplot karşılaştırması
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(important_sensors[:8]):
        ax = axes[idx]
        
        data_pass = df[df[target_col] == -1][sensor].dropna()
        data_fail = df[df[target_col] == 1][sensor].dropna()
        
        bp = ax.boxplot([data_pass, data_fail], patch_artist=True, labels=['Pass', 'Fail'])
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        
        ax.set_title(f'Sensör {sensor}')
        ax.set_ylabel('Değer')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fig10_boxplot_karsilastirma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Görsel kaydedildi: {output_dir}fig10_boxplot_karsilastirma.png")
    
    return pd.DataFrame(sensor_analysis)


# =============================================================================
# BÖLÜM 8: SONUÇ VE ÖZET
# =============================================================================
def bolum8_sonuc_ozet(df, feature_cols, target_col, missing_stats, class_stats, 
                      corr_stats, outlier_summary):
    """EDA sonuçlarının özeti ve model kurulum önerileri"""
    
    print("\n" + "="*80)
    print("BÖLÜM 8: SONUÇ VE ÖZET")
    print("="*80)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EDA SONUÇ RAPORU - SECOM VERİ SETİ                        ║
║              Yarı İletken Üretim Hatası Tahmin Analizi                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 1. VERİ KALİTESİ
    print("="*60)
    print("1. VERİ KALİTESİ DEĞERLENDİRMESİ")
    print("="*60)
    
    const_cols = df[feature_cols].std().fillna(0).eq(0).sum()
    
    print(f"""
   VERİ KALİTESİ SKORU: ORTA-DÜŞÜK

   OLUMLU YÖNLER:
   • Toplam {len(df):,} gözlem mevcut - yeterli örnek büyüklüğü
   • {len(feature_cols)} sensör verisi - zengin özellik uzayı
   • Hedef değişken tamamen dolu
   
   SORUNLU YÖNLER:
   • Genel eksik veri oranı: %{missing_stats['Genel Eksik Oranı']:.2f}
   • %50+ eksik içeren sütun sayısı: {missing_stats['Yüksek Eksik Sütun (>50%)']}
   • Sabit değerli sütun sayısı: {const_cols}
""")
    
    # 2. EKSİK VERİ PROBLEMİ
    print("="*60)
    print("2. EKSİK VERİ PROBLEMİ CİDDİYETİ: YÜKSEK")
    print("="*60)
    
    print(f"""
   • Toplam eksik hücre: {missing_stats['Toplam Eksik']:,}
   • Eksik değer içeren sütun: {missing_stats['Eksik İçeren Sütun']}
   
   ÖNERİ: %40+ eksik sütunları sil, kalan için IterativeImputer
""")
    
    # 3. SINIF DENGESİZLİĞİ
    print("="*60)
    print("3. SINIF DENGESİZLİĞİ: KRİTİK")
    print("="*60)
    
    print(f"""
   • Pass:Fail oranı = {class_stats['Imbalance Ratio']:.1f}:1
   • Fail oranı: %{class_stats['Fail Percent']:.2f}
   
   ÖNERİ: SMOTE, class_weight='balanced', F1-Score kullan
""")
    
    # 4. BOYUT AZALTMA
    print("="*60)
    print("4. BOYUT AZALTMA GEREKSİNİMİ: KRİTİK")
    print("="*60)
    
    print(f"""
   • |r| > 0.95 korelasyonlu çift sayısı: {corr_stats['Very High Corr (>0.95)']}
   • Özellik/Gözlem oranı: {len(feature_cols)/len(df):.2f}
   
   ÖNERİ: PCA (%95 varyans) veya korelasyon bazlı eleme
""")
    
    # 5. AYKIRI DEĞERLER
    print("="*60)
    print("5. AYKIRI DEĞERLER: ORTA-YÜKSEK")
    print("="*60)
    
    print(f"""
   • Ortalama aykırı oran: %{outlier_summary['Avg Outlier Rate']:.2f}
   • >%10 aykırılık gösteren sütun: {outlier_summary['High Outlier Cols (>10%)']}
   
   ÖNERİ: RobustScaler, winsorization, tree-based modeller
""")
    
    # MODEL KURULUM ÖNERİLERİ
    print("\n" + "="*60)
    print("MODEL KURULUM ÖNERİLERİ")
    print("="*60)
    
    print("""
   1. ÖN İŞLEME ADIMLARI:
      • %40+ eksik sütunları sil
      • Sabit değerli sütunları sil
      • IterativeImputer ile eksik değer doldur
      • RobustScaler ile ölçekleme
      • PCA veya feature selection ile boyut azalt
   
   2. DENGESİZLİK ÇÖZÜMÜ:
      • SMOTE veya class_weight='balanced'
      • Threshold optimizasyonu
   
   3. MODEL SEÇİMİ:
      • Random Forest / XGBoost / LightGBM
      • Tree-based modeller aykırı değerlere dayanıklı
   
   4. DEĞERLENDİRME:
      • Stratified K-Fold Cross Validation
      • F1-Score ve PR-AUC metrikleri
      • Confusion Matrix analizi
""")
    
    print("\n" + "="*80)
    print("                    ANALİZ TAMAMLANDI")
    print("="*80)


# =============================================================================
# ANA FONKSİYON
# =============================================================================
def run_full_eda(data_path, output_dir='./eda_outputs/'):
    """
    Tam EDA analizini çalıştırır.
    
    Parameters:
    -----------
    data_path : str
        Veri dosyasının yolu
    output_dir : str
        Çıktı klasörü
    
    Returns:
    --------
    dict : Tüm analiz sonuçlarını içeren sözlük
    """
    
    # Çıktı klasörünü oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Veri yükleme
    print("Veri yükleniyor...")
    df = pd.read_csv(data_path)
    
    target_col = 'Pass/Fail'
    feature_cols = [col for col in df.columns if col not in ['Time', 'Pass/Fail']]
    
    print(f"✓ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Tüm bölümleri çalıştır
    results = {}
    
    # Bölüm 1
    results['summary'] = bolum1_veri_tanitimi(df, target_col, feature_cols)
    
    # Bölüm 2
    missing_stats, missing_df = bolum2_eksik_veri_analizi(df, target_col, feature_cols, output_dir)
    results['missing'] = missing_stats
    
    # Bölüm 3
    stats_df = bolum3_tanimlayici_istatistikler(df, feature_cols, output_dir)
    results['descriptive'] = stats_df
    
    # Bölüm 4
    class_stats = bolum4_sinif_dengesizligi(df, target_col, output_dir)
    results['class_balance'] = class_stats
    
    # Bölüm 5
    corr_stats, target_corr = bolum5_korelasyon_analizi(df, target_col, feature_cols, output_dir)
    results['correlation'] = corr_stats
    
    # Bölüm 6
    outlier_summary, outlier_df = bolum6_aykiri_deger_analizi(df, feature_cols, output_dir)
    results['outliers'] = outlier_summary
    
    # Bölüm 7
    sensor_analysis = bolum7_hedef_sensor_iliskisi(df, target_col, feature_cols, target_corr, output_dir)
    results['sensor_analysis'] = sensor_analysis
    
    # Bölüm 8
    bolum8_sonuc_ozet(df, feature_cols, target_col, missing_stats, class_stats, 
                      corr_stats, outlier_summary)
    
    print(f"\n✓ Tüm görseller '{output_dir}' klasörüne kaydedildi.")
    
    return results


# =============================================================================
# ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    
    # Veri dosyası yolu - KENDİ YOLUNUZU YAZIN
    DATA_PATH = 'Downloads/Buket/uci-secom.csv'
    OUTPUT_DIR = './eda_outputs/'
    
    # Tam EDA analizini çalıştır
    results = run_full_eda(DATA_PATH, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("EDA ANALİZİ BAŞARIYLA TAMAMLANDI!")
    print("="*80)