"""
Klasik Görüntü İşleme ile Barkod Tespiti
========================================
Derin öğrenme kullanmadan; gradyan analizi, eşikleme ve morfolojik işlemlerle
görüntüdeki barkodun konumunu bulur ve sonucu IoU ile değerlendirir.

Yöntem akışı:
    Gri ton + Gaussian -> Sobel -> |Gx| - |Gy| -> Mean filtre -> Otsu
    -> Morfoloji (4 ölçek, yatay ve dikey) -> Bağlı bileşenler -> Puanlama

Kısaltmalar:
    GT  = Ground Truth (veri setindeki gerçek barkod konumu)
    IoU = Intersection over Union (tahmin ile GT kutusunun örtüşme oranı)

Kullanım:
    pip install -r requirements.txt
    python barkod_tespit.py

Veri seti:
    Görüntüler (.jpg) ve aynı adlı YOLO formatındaki etiket dosyaları (.txt)
    'dataset' klasörüne konmalıdır. Sonuçlar 'sonuclar' klasörüne kaydedilir.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

# Kodun bulunduğu klasör
KOD_KLASORU = os.path.dirname(os.path.abspath(__file__))
# Veri seti klasörü ("dataset"): önce kodun yanında, sonra bir üst klasörde aranır.
# Veri seti başka bir yerdeyse bu satırı tam yolla değiştirin: VERI_KLASORU = r"C:\...\dataset"
VERI_KLASORU = next((k for k in [os.path.join(KOD_KLASORU, 'dataset'),
                                 os.path.join(os.path.dirname(KOD_KLASORU), 'dataset')]
                     if os.path.isdir(k)), os.path.join(KOD_KLASORU, 'dataset'))
SONUC_KLASORU = os.path.join(KOD_KLASORU, 'sonuclar')

# =====================================================================
# PARAMETRELER
# =====================================================================
BLUR_BOYUT = 5
MEAN_BOYUT = 7
# (yükseklik, genişlik) - yatay barkod için; dikey barkodda çekirdek 90° çevrilir
CLOSE_CEKIRDEKLERI = [(3, 9), (5, 15), (7, 25), (9, 35)]   # 4 farklı ölçek
EROZYON_ITER = 4
DILATASYON_ITER = 4
MIN_ALAN_ORANI = 0.01   # Görüntünün %1'inden küçük bölgeler elenir
PAY = 0.10              # Kutu her kenardan %10 genişletilir (GT kutusundaki rakamları ve boşluğu da kapsamak için)
KENAR_CEZASI = 0.5      # Resmin her bir kenarına değen bölgenin puanı bu katsayıyla çarpılır


# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================
def resim_oku(yol):
    """cv2.imread Windows'ta Türkçe karakterli yolları okuyamadığı için
    dosya NumPy ile byte olarak okunup cv2.imdecode ile çözülür."""
    return cv2.imdecode(np.fromfile(yol, dtype=np.uint8), cv2.IMREAD_COLOR)


# --- .txt dosyasındaki gerçek konumu (GT) okuma ---
def yolo_oku(txt_yolu, W, H):
    """YOLO formatındaki (sınıf xc yc w h, normalize) etiketi piksel kutusuna çevirir."""
    kutular = []
    with open(txt_yolu) as f:
        for satir in f:
            p = satir.split()
            if len(p) < 5:
                continue
            _, xc, yc, w, h = map(float, p[:5])
            kutular.append((int((xc - w / 2) * W), int((yc - h / 2) * H),
                            int((xc + w / 2) * W), int((yc + h / 2) * H)))
    return kutular


# --- IoU HESABI ---
def iou(a, b):
    """Intersection over Union = kesişim alanı / birleşim alanı. Kutular (x1, y1, x2, y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    kesisim = max(0, x2 - x1) * max(0, y2 - y1)
    birlesim = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def alan(k):
    return (k[2] - k[0]) * (k[3] - k[1])


# =====================================================================
# TESPİT ALGORİTMASI
# =====================================================================

# --- ADIM 2-4: Sobel -> |Gx|-|Gy| -> Mean -> Otsu ---
def gradyan_farki(gri, yatay):
    """Sobel -> |Gx|-|Gy| -> Mean filtre -> Otsu."""
    # ADIM 2: Sobel X ve Sobel Y ile yatay ve dikey gradyan
    gx = cv2.Sobel(gri, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gri, cv2.CV_32F, 0, 1, ksize=3)
    ax, ay = np.abs(gx), np.abs(gy)

    # ADIM 3a: TEMEL FİKİR - gradyan farkı
    # Barkod çizgileri tek yönde güçlü kenar üretir.
    # Yatay barkod: |Gx| - |Gy|, dikey barkod: |Gy| - |Gx|
    fark = (ax - ay) if yatay else (ay - ax)
    fark = cv2.convertScaleAbs(np.clip(fark, 0, None))

    # ADIM 3b: Mean (ortalama) filtre - çizgi aralarını doldurur
    fark = cv2.blur(fark, (MEAN_BOYUT, MEAN_BOYUT))

    # ADIM 4: Otsu eşikleme - eşik her görüntü için otomatik bulunur
    _, otsu = cv2.threshold(fark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ax, ay, fark, otsu


# --- ADIM 5: MORFOLOJİ ---
def morfoloji(otsu, cekirdek_boyut, yatay):
    """Dikdörtgen çekirdekle closing, ardından erosion + dilation."""
    # Dikdörtgen çekirdek: yatay barkodda örn. 5x15, dikey barkodda 90° çevrilmiş hali (15x5)
    kh, kw = cekirdek_boyut
    cekirdek = np.ones((kh, kw) if yatay else (kw, kh), np.uint8)
    # Closing: barkod çizgilerini tek bloğa birleştirir
    kapali = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, cekirdek)
    # Erosion küçük lekeleri siler, dilation barkodu eski boyutuna getirir
    k3 = np.ones((3, 3), np.uint8)
    temiz = cv2.erode(kapali, k3, iterations=EROZYON_ITER)
    temiz = cv2.dilate(temiz, k3, iterations=DILATASYON_ITER)
    return kapali, temiz


# --- ADIM 6b: PUANLAMA ---
def bolge_puani(maske, kutu, fark, ax, ay, yatay, H, W):
    """Bir bölgenin ne kadar barkoda benzediğini ölçer.
    puan = alan^0.6 * ortalama_fark * doluluk * yon_orani^0.25 * KENAR_CEZASI^(değen kenar sayısı)"""
    x1, y1, x2, y2 = kutu
    bolge_alani = maske.sum()
    ort_fark = fark[maske].mean()                       # alan x yoğunluk: çizgi yoğunluğu
    doluluk = bolge_alani / ((x2 - x1) * (y2 - y1))     # barkod dikdörtgendir -> doluluk yüksek
    sx, sy = ax[maske].sum(), ay[maske].sum()
    yon_orani = sx / (sy + 1) if yatay else sy / (sx + 1)  # tek yönlü çizgi -> oran yüksek
    kenar = int(x1 == 0) + int(y1 == 0) + int(x2 == W) + int(y2 == H)  # kenar cezası: resim kenarına değme
    return (bolge_alani ** 0.6) * ort_fark * doluluk * (yon_orani ** 0.25) * (KENAR_CEZASI ** kenar)


# --- ANA FONKSİYON: tüm adımları sırayla çalıştırır ---
def barkod_bul(img, ara_adimlar=False):
    """Görüntüdeki barkodun kutusunu (x1, y1, x2, y2) döndürür."""
    # ADIM 1: Gri ton + Gaussian blur
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gri = cv2.GaussianBlur(gri, (BLUR_BOYUT, BLUR_BOYUT), 0)
    H, W = gri.shape

    en_iyi = {'puan': -1, 'kutu': None}
    adimlar = {'gri': gri}
    # İKİ YÖN: önce yatay, sonra dikey barkod için
    for yatay in (True, False):
        ax, ay, fark, otsu = gradyan_farki(gri, yatay)        # ADIM 2-4
        # 4 FARKLI ÖLÇEK: her çekirdekle ayrı closing (farklı boyuttaki barkodlar için)
        for cekirdek in CLOSE_CEKIRDEKLERI:
            kapali, temiz = morfoloji(otsu, cekirdek, yatay)  # ADIM 5
            if yatay and cekirdek == (5, 15):
                adimlar.update(gx=ax, fark=fark, otsu=otsu, closing=kapali, temiz=temiz)

            # ADIM 6a: Connected components - beyaz bölgeleri etiketle
            n, etiket, stats, _ = cv2.connectedComponentsWithStats(temiz, connectivity=8)
            for i in range(1, n):  # 0 = arka plan
                x, y, w, h, a = stats[i]
                if a < MIN_ALAN_ORANI * H * W:   # görüntünün %1'inden küçükler elenir
                    continue
                kutu = (x, y, x + w, y + h)
                puan = bolge_puani(etiket == i, kutu, fark, ax, ay, yatay, H, W)  # ADIM 6b
                # En yüksek puanlı bölge barkod olarak seçilir
                if puan > en_iyi['puan']:
                    en_iyi = {'puan': puan, 'kutu': kutu, 'yatay': yatay, 'cekirdek': cekirdek}

    if en_iyi['kutu'] is None:  # hiç bölge yoksa görüntünün ortası
        en_iyi['kutu'] = (W // 4, H // 4, 3 * W // 4, 3 * H // 4)

    # ADIM 7: Kutuyu her kenardan %10 genişlet
    x1, y1, x2, y2 = en_iyi['kutu']
    px, py = int(PAY * (x2 - x1)), int(PAY * (y2 - y1))
    kutu = (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))

    if ara_adimlar:
        adimlar['secim'] = en_iyi
        return kutu, adimlar
    return kutu


# =====================================================================
# DEĞERLENDİRME
# =====================================================================

# --- Başarısız örneklerin hata türü ---
def hata_turu(tahmin, gt, skor):
    """Başarısız (IoU<0.5) tahminin nedenini sınıflandırır."""
    # Tahminin merkezi GT kutusunun dışındaysa başka bir nesne seçilmiştir
    mx, my = (tahmin[0] + tahmin[2]) / 2, (tahmin[1] + tahmin[3]) / 2
    if not (gt[0] <= mx <= gt[2] and gt[1] <= my <= gt[3]):
        return 'Yanlış bölge'
    if alan(tahmin) > alan(gt):
        return 'Kutu çok büyük (birleşme)'
    return 'Kutu çok küçük (parçalanma)'


# --- TÜM VERİ SETİNDE IoU ---
def degerlendir(klasor):
    sonuclar = []  # (dosya, iou, tahmin, gt_listesi, eşleşen_gt)
    for jpg in sorted(glob.glob(os.path.join(klasor, '*.jpg'))):
        txt = jpg[:-4] + '.txt'
        if not os.path.exists(txt):
            continue
        img = resim_oku(jpg)
        if img is None:
            print(f"Uyarı: {jpg} okunamadı, atlanıyor.")
            continue
        H, W = img.shape[:2]
        gt = yolo_oku(txt, W, H)
        if not gt:
            continue
        tahmin = barkod_bul(img)
        # Birden fazla barkod etiketliyse tahmin, en çok örtüştüğü GT kutusuyla karşılaştırılır
        eslesen = max(gt, key=lambda g: iou(tahmin, g))
        sonuclar.append((os.path.basename(jpg), iou(tahmin, eslesen), tahmin, gt, eslesen))
    return sonuclar


def kutu_ciz(img, gt, eslesen, tahmin, skor):
    for g in gt:  # tüm GT kutuları kalın kırmızı; eşleşen GT ayrıca sarı çizgiyle işaretli
        cv2.rectangle(img, g[:2], g[2:], (0, 0, 255), 5)
    cv2.rectangle(img, eslesen[:2], eslesen[2:], (0, 255, 255), 1)
    cv2.rectangle(img, tahmin[:2], tahmin[2:], (0, 255, 0), 2)  # yeşil = tahmin
    cv2.putText(img, f"IoU={skor:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =====================================================================
# ÇALIŞTIRMA: sonuçları yazdır ve grafikleri kaydet
# =====================================================================
if __name__ == '__main__':
    os.makedirs(SONUC_KLASORU, exist_ok=True)
    baslangic = time.perf_counter()
    sonuclar = degerlendir(VERI_KLASORU)
    sure = time.perf_counter() - baslangic
    if not sonuclar:
        raise SystemExit(f"Hata: {VERI_KLASORU} içinde jpg/txt çifti bulunamadı.")

    skorlar = np.array([s[1] for s in sonuclar])
    print("=" * 50)
    print(" BARKOD TESPİTİ - IoU SONUÇLARI")
    print("=" * 50)
    print(f"Görüntü sayısı        : {len(skorlar)}")
    print(f"Ortalama IoU (mIoU)   : {skorlar.mean():.4f}")
    print(f"Medyan IoU            : {np.median(skorlar):.4f}")
    for t in (0.5, 0.7, 0.9):
        print(f"IoU >= {t} olan oran  : %{100 * np.mean(skorlar >= t):.1f}")
    print(f"Görüntü başı süre     : {1000 * sure / len(skorlar):.1f} ms")

    # Başarısız örneklerin hata türü dağılımı
    basarisiz = [s for s in sonuclar if s[1] < 0.5]
    turler = {}
    for ad, s, t, gt, e in basarisiz:
        tur = hata_turu(t, e, s)
        turler[tur] = turler.get(tur, 0) + 1
    print("-" * 50)
    print(f"Başarısız (IoU<0.5)   : {len(basarisiz)}")
    for tur, n in sorted(turler.items(), key=lambda x: -x[1]):
        print(f"  {tur:<28}: {n}")
    print("=" * 50)

    # Görüntü bazlı sonuçlar
    with open(os.path.join(SONUC_KLASORU, 'iou_sonuclari.csv'), 'w', encoding='utf-8') as f:
        f.write("dosya,iou,x1,y1,x2,y2,hata_turu\n")
        for ad, s, t, gt, e in sonuclar:
            tur = hata_turu(t, e, s) if s < 0.5 else ''
            f.write(f"{ad},{s:.4f},{t[0]},{t[1]},{t[2]},{t[3]},{tur}\n")

    # --- Grafik 1: Ara adımlar ---
    ad0, s0, _, gt0, e0 = sonuclar[0]
    ornek = resim_oku(os.path.join(VERI_KLASORU, ad0))
    kutu, a = barkod_bul(ornek, ara_adimlar=True)
    resimler = [a['gri'], cv2.convertScaleAbs(a['gx']), a['fark'], a['otsu'], a['closing'], a['temiz']]
    basliklar = ['1. Gri + Gaussian', '2. Sobel |Gx|', '3-4. |Gx|-|Gy| + Mean',
                 '5. Otsu', '6. Closing (5x15)', '7. Erosion + Dilation']
    plt.figure(figsize=(15, 8))
    for i in range(6):
        plt.subplot(2, 4, i + 1)
        plt.imshow(resimler[i], cmap='gray'); plt.title(basliklar[i]); plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(kutu_ciz(ornek.copy(), gt0, e0, kutu, s0))
    plt.title('8. Sonuç (yeşil=tahmin, kırmızı=GT)'); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, 'adimlar.png'), dpi=120)

    # --- Grafik 2: IoU dağılımı ---
    plt.figure(figsize=(8, 5))
    plt.hist(skorlar, bins=20, range=(0, 1), color='gray', edgecolor='black')
    plt.axvline(skorlar.mean(), color='red', linestyle='--', label=f'Ortalama IoU = {skorlar.mean():.3f}')
    plt.xlabel('IoU'); plt.ylabel('Görüntü sayısı'); plt.title('Veri Seti IoU Dağılımı'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, 'iou_histogram.png'), dpi=120)

    # --- Grafik 3: En iyi 4 ve en kötü 4 örnek ---
    sirali = sorted(sonuclar, key=lambda s: s[1])
    secilen = sirali[-4:][::-1] + sirali[:4]
    plt.figure(figsize=(16, 7))
    for i, (dosya, s, t, gt, e) in enumerate(secilen):
        plt.subplot(2, 4, i + 1)
        plt.imshow(kutu_ciz(resim_oku(os.path.join(VERI_KLASORU, dosya)), gt, e, t, s))
        etiket = 'Başarılı' if i < 4 else 'Başarısız - ' + hata_turu(t, e, s)
        plt.title(f"{etiket}\n{dosya}", fontsize=9); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, 'ornekler.png'), dpi=120)
    plt.show()