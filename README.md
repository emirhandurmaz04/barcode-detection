# Klasik Görüntü İşleme ile Barkod Tespiti

Derin öğrenme kullanmadan; **gradyan analizi, Otsu eşikleme ve morfolojik işlemlerle** görüntülerdeki barkodun konumunu bulan bir sistem. Tespit sonuçları, veri setindeki gerçek etiketlerle **IoU (Intersection over Union)** üzerinden değerlendirilir.

> **Sonuç:** IoU ≥ 0,5 ölçütüne göre **%98 başarı**

Burdur Mehmet Akif Ersoy Üniversitesi, Görüntü İşleme dersi kapsamında geliştirilmiştir.

---

## Sonuçlar

**Algoritmanın ara adımları**

![Ara adımlar](sonuclar/adimlar.png)

**Veri setindeki IoU dağılımı**

![IoU dağılımı](sonuclar/iou_histogram.png)

**En başarılı ve en başarısız örnekler** (yeşil = tahmin, kırmızı = gerçek konum)

![Örnekler](sonuclar/ornekler.png)

---

## Yöntem

Barkod çizgileri tek bir yönde çok güçlü kenarlar üretir. Algoritma bu özelliği kullanarak barkodu görüntünün geri kalanından ayırır:

1. **Ön işleme:** Görüntü gri tona çevrilir ve Gaussian blur ile gürültü azaltılır.
2. **Gradyan:** Sobel filtresiyle yatay (Gx) ve dikey (Gy) gradyanlar hesaplanır.
3. **Gradyan farkı:** Yatay barkod için `|Gx| - |Gy|`, dikey barkod için `|Gy| - |Gx|` alınır. Ortalama (mean) filtre çizgi aralarını doldurur.
4. **Otsu eşikleme:** Eşik değeri her görüntü için otomatik bulunur.
5. **Morfoloji:** Dikdörtgen çekirdekle *closing* çizgileri tek bloğa birleştirir, ardından *erosion + dilation* küçük lekeleri temizler. Farklı boyuttaki barkodlar için **4 ölçekte** çekirdek denenir.
6. **Aday seçimi:** Bağlı bileşenler (connected components) bulunur ve her aday bölge puanlanır. En yüksek puanlı bölge barkod olarak seçilir.
7. **Son düzeltme:** Seçilen kutu, barkod altındaki rakamları da kapsaması için her kenardan %10 genişletilir.

Tüm işlem hem yatay hem dikey barkodlar için ayrı ayrı yapılır.

### Puanlama fonksiyonu

```
puan = alan^0.6 × ortalama_gradyan_farkı × doluluk × yön_oranı^0.25 × 0.5^(değen_kenar_sayısı)
```

| Terim | Anlamı |
|---|---|
| alan | Büyük bölgeler tercih edilir |
| ortalama gradyan farkı | Çizgi yoğunluğu yüksek bölgeler tercih edilir |
| doluluk | Barkod dikdörtgen olduğu için kutuyu iyi dolduran bölgeler tercih edilir |
| yön oranı | Çizgileri tek yönde olan bölgeler tercih edilir |
| kenar cezası | Görüntünün kenarına değen bölgelerin puanı düşürülür |

---

## Değerlendirme

- Gerçek barkod konumları (**GT**, Ground Truth) YOLO formatındaki `.txt` etiket dosyalarından okunur.
- Tahmin ile GT kutusu arasındaki **IoU** hesaplanır. IoU ≥ 0,5 olan tahminler başarılı sayılır.
- Görüntüde birden fazla barkod varsa tahmin, en çok örtüştüğü GT kutusuyla karşılaştırılır.
- Başarısız tahminler türüne göre sınıflandırılır: *yanlış bölge*, *kutu çok büyük (birleşme)*, *kutu çok küçük (parçalanma)*.
- Görüntü bazlı tüm sonuçlar `sonuclar/iou_sonuclari.csv` dosyasına kaydedilir.

---

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python barkod_tespit.py
```

Veri seti, kodun yanındaki `dataset` klasörüne şu yapıda konmalıdır (veri seti boyutu nedeniyle depoya eklenmemiştir):

```
dataset/
├── ornek1.jpg
├── ornek1.txt    # YOLO formatı: sinif x_merkez y_merkez genislik yukseklik
├── ornek2.jpg
├── ornek2.txt
└── ...
```

## Proje Yapısı

```
├── barkod_tespit.py     # Tespit algoritması ve değerlendirme
├── requirements.txt     # Gerekli kütüphaneler
└── sonuclar/            # Üretilen grafikler ve IoU sonuçları
```

## Kullanılan Teknolojiler

Python · OpenCV · NumPy · Matplotlib

---

**Emirhan Durmaz** · [LinkedIn](https://www.linkedin.com/in/emirhandurmaz04) · [GitHub](https://github.com/emirhandurmaz04)
