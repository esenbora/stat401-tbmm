# STAT 401 Dönem Projesi — Toplanan Veri Seti

TBMM 28. Dönem yazılı soru önergeleri üzerine topladığımız veri setinin özeti.

## Veri kaynağı

Veriyi TBMM resmî sitesinden (tbmm.gov.tr) kendimiz çektik (web scraping). Hazır bir veri seti kullanmadık. Adımlar:

1. 28. dönem 592 milletvekilinin listesini aldık (parti, il bilgisiyle).
2. Her milletvekilinin profilinden yazılı soru önergelerinin linklerini topladık. Toplam 44.485 önerge çıktı.
3. Her önergenin detay sayfasından yapısal alanları (sahip, muhatap bakanlık, tarih, özet, durum) çektik.
4. Önergelerin tam metni PDF olarak tutuluyor. PDF'lerin bir kısmı seçilebilir metin içeriyor, bir kısmı taranmış görüntü. Metin katmanı olanları doğrudan okuduk (pypdf), taranmış olanları PaddleOCR ile (Türkçe model, Colab GPU) metne çevirdik.

## Boyut

| | |
|---|---|
| Toplam önerge | 44.485 |
| Milletvekili | 592 |
| Muhatap bakanlık | 20 |
| Soru sahibi il | 78 |
| Zaman aralığı | 2023–2026 |

Tam metin çıkarımında önergelerin %65'i PDF metin katmanından doğrudan okundu (28.893 belge), %35'i taranmış olduğu için OCR'landı (15.591 belge). OCR ortalama güveni yaklaşık %94.

## Her önergede tutulan alanlar

esas no, geliş tarihi, soru sahibi (il + milletvekili), muhatap bakanlık, durum, önerge özeti, önerge tam metni.

## Birkaç istatistik

Yıllara göre: 2023'te 7.854, 2024'te 14.448, 2025'te 16.144, 2026'da (kısmi) 6.038 önerge.

En çok soru alan bakanlıklar: Adalet (6.140), İçişleri (5.571), Çevre-Şehircilik-İklim (4.587), Tarım-Orman (4.584), Sağlık (3.520), Milli Eğitim (3.508).

Milletvekili dağılımı: AK Parti 275, CHP 138, DEM Parti 56, MHP 46, İYİ Parti 29, Yeni Yol 20, diğer 28.

## Planladığımız analizler (onayınıza)

1. Bakanlık, konu, parti ve zaman ekseninde dağılım. Hangi bakanlık hangi konuda en çok soru alıyor, partiler nasıl ayrışıyor. Tekrar eden (kopya) önergeleri MinHash/LSH ile tespit.
2. İl bazlı ilgi haritası. Hangi iller en çok gündemde, illeri konu profillerine göre K-Means ile kümeleme.
3. Milletvekili ortak imza ağı. PageRank ile etki skoru, Louvain ile topluluk tespiti, parti dışı koalisyonlar.

Kullanacağımız araçlar: Apache Spark, Delta Lake, Spark MLlib (FP-Growth, MinHash/LSH, K-Means, PageRank, Louvain, LDA).

## Ekler

- TBMM_VeriSeti.xlsx — örnek satırlar ve istatistik tabloları
- ornek_veri_40_satir.csv — 38 örnek satır
- tum_metadata_44485.csv — tüm önergelerin metadata'sı (8.7 MB)
- Kaynak kod: github.com/esenbora/stat401-tbmm
