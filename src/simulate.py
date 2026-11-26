"""Bir banka subesinde mevcut yonlendirme ile KYC'yi karsilastirir."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simpy

from generate_data import CLASSES


GUN = 8 * 60
GISE_SAYISI = 6
SEED = 495
VARSAYIM_MEVCUT_TANISMA = 2.0
VARSAYIM_KYC_DOGRU_TANISMA = 0.5
VARSAYIM_MEVCUT_YANLIS_GISE = 0.25
VARSAYIM_KYC_DOGRU_YANLIS_GISE = 0.05
VARSAYIM_YANLIS_GISE_KAYBI = 1.0
VARSAYIM_KYC_DOGRU_ISLEM_CARPANI = 0.90
VARSAYIM_YENIDEN_YONLENDIRME_FORM_ISRAFI = 1

# VARSAYIM: Geliş aralıkları ve işlem süreleri üssel, nedenler eşit olasılıklı.
# Tanışma hariç ortalama işlem 4.75 dk. Mevcut senaryoda müşteri başına
# 4.75 + 2 + 0.25 × (2 + 1) = 7.5 dk gişe işi, 40 geliş/saat ve
# 6 gişede teorik olarak %83.3 doluluk verir.
VARSAYIM_ISLEM_ORTALAMASI = {
    "failed_transfer": 4,
    "blocked_card": 3,
    "loan_inquiry": 7,
    "account_opening": 6,
    "card_application": 5,
    "fx_exchange": 3,
    "dispute_charge": 7,
    "app_access_help": 3,
}
METRIKLER = (
    "Ortalama bekleme (dk)", "P90 bekleme (dk)",
    "Ortalama sistemde kalma (dk)", "Günlük yeniden yönlendirme",
    "Gişe doluluğu (%)", "Günlük israf edilen form",
)


def musterileri_uret(gelis, seed):
    rng = np.random.default_rng(seed)
    zaman = rng.exponential(60 / gelis)
    while zaman < GUN:
        neden = rng.choice(CLASSES)
        yield (
            zaman, rng.exponential(VARSAYIM_ISLEM_ORTALAMASI[neden]),
            rng.random(), rng.random(),
        )
        zaman += rng.exponential(60 / gelis)


def sube(musteriler, top1=None):
    ortam = simpy.Environment()
    giseler = simpy.Resource(ortam, capacity=GISE_SAYISI)
    beklemeler, sistem_sureleri = [], []
    mesai_mesgul = 0
    yonlendirme = 0

    def musteri(gelis, islem, tahmin_cekilisi, yonlendirme_cekilisi):
        nonlocal yonlendirme, mesai_mesgul
        yield ortam.timeout(gelis)
        dogru_tahmin = top1 is not None and tahmin_cekilisi < top1
        tanisma = VARSAYIM_MEVCUT_TANISMA
        yanlis_gise_olasiligi = VARSAYIM_MEVCUT_YANLIS_GISE
        if dogru_tahmin:
            tanisma = VARSAYIM_KYC_DOGRU_TANISMA
            yanlis_gise_olasiligi = VARSAYIM_KYC_DOGRU_YANLIS_GISE
            islem *= VARSAYIM_KYC_DOGRU_ISLEM_CARPANI

        yanlis = yonlendirme_cekilisi < yanlis_gise_olasiligi
        bekleme = 0
        while True:
            sira_baslangici = ortam.now
            with giseler.request() as sira:
                yield sira
                servis_baslangici = ortam.now
                bekleme += servis_baslangici - sira_baslangici
                yield ortam.timeout(tanisma + (VARSAYIM_YANLIS_GISE_KAYBI if yanlis else islem))
                mesai_mesgul += max(0, min(ortam.now, GUN) - servis_baslangici)
            if not yanlis:
                break
            # Yeni istek aynı FIFO kuyruğun sonuna girer ve tekrar tanışır.
            yonlendirme += 1
            yanlis = False
        beklemeler.append(bekleme)
        sistem_sureleri.append(ortam.now - gelis)

    for kayit in musteriler:
        ortam.process(musteri(*kayit))
    # Kapanistan sonra yeni gelis yok, siradaki musteriler tamamlanir.
    ortam.run()
    return beklemeler, sistem_sureleri, yonlendirme, mesai_mesgul


def ozetle(gunler, top1=None):
    beklemeler, sistem_sureleri = [], []
    yonlendirmeler, doluluklar = [], []
    for musteriler in gunler:
        bekleme, sure, yonlendirme, mesgul = sube(musteriler, top1)
        beklemeler.extend(bekleme)
        sistem_sureleri.extend(sure)
        yonlendirmeler.append(yonlendirme)
        doluluklar.append(100 * mesgul / (GUN * GISE_SAYISI))
    metrikler = [
        np.mean(beklemeler) if beklemeler else float("nan"),
        np.percentile(beklemeler, 90) if beklemeler else float("nan"),
        np.mean(sistem_sureleri) if sistem_sureleri else float("nan"),
        np.mean(yonlendirmeler),
        np.mean(doluluklar),
        np.mean(yonlendirmeler) * VARSAYIM_YENIDEN_YONLENDIRME_FORM_ISRAFI,
    ]
    return metrikler, beklemeler


def main():
    parser = argparse.ArgumentParser(description="Banka şubesi bekleme simülasyonu")
    parser.add_argument("--gelis", type=float, default=40, help="Saatlik ortalama geliş")
    parser.add_argument("--top1", type=float, default=0.70, help="Top-1 doğruluğu (0–1)")
    parser.add_argument("--tekrar", type=int, default=30, help="Simüle edilen gün sayısı")
    parser.add_argument("--duyarlilik", action="store_true", help="Top-1 duyarlılık grafiği")
    args = parser.parse_args()
    if not np.isfinite(args.gelis) or args.gelis <= 0:
        parser.error("--gelis pozitif ve sonlu olmalı")
    if not 0 <= args.top1 <= 1:
        parser.error("--top1 0 ile 1 arasında olmalı")
    if args.tekrar < 1:
        parser.error("--tekrar en az 1 olmalı")

    # Ayni gelisleri, nedenleri ve islem surelerini her senaryoda kullan.
    gunler = [list(musterileri_uret(args.gelis, SEED + i)) for i in range(args.tekrar)]
    mevcut, bekleme_a = ozetle(gunler)
    kyc, bekleme_b = ozetle(gunler, args.top1)
    data = Path(__file__).resolve().parent.parent / "data"
    data.mkdir(exist_ok=True)

    with (data / "sim_sonuc.csv").open("w", newline="", encoding="utf-8") as dosya:
        writer = csv.writer(dosya)
        writer.writerow(["Metrik", "A (Mevcut)", "B (KYC)", "İyileşme (%)"])
        print(f"{'Metrik':<33} {'A (Mevcut)':>12} {'B (KYC)':>12} {'İyileşme (%)':>14}")
        print("-" * 74)
        for ad, a, b in zip(METRIKLER, mevcut, kyc):
            # Doluluk bir kapasite gostergesidir, yuksek veya dusuk olmasi
            # tek basina bir iyilesme sayilmaz.
            iyilesme = f"{100 * (a - b) / a:.2f}" if a and np.isfinite(a) and np.isfinite(b) else ""
            a_yazi = f"{a:.2f}" if np.isfinite(a) else ""
            b_yazi = f"{b:.2f}" if np.isfinite(b) else ""
            writer.writerow([ad, a_yazi, b_yazi, iyilesme])
            print(f"{ad:<33} {a_yazi or '—':>12} {b_yazi or '—':>12} {iyilesme or '—':>14}")
    print("Pozitif yüzde düşüşü gösterir. Doluluk satırı göreli değişimi gösterir.")

    fig, ax = plt.subplots()
    araliklar = np.histogram_bin_edges(bekleme_a + bekleme_b, bins=30)
    ax.hist(bekleme_a, bins=araliklar, alpha=0.6, label="A (Mevcut)")
    ax.hist(bekleme_b, bins=araliklar, alpha=0.6, label="B (KYC)")
    ax.set(xlabel="Toplam kuyrukta bekleme (dk)", ylabel="Müşteri sayısı")
    ax.legend()
    fig.savefig(data / "sim_bekleme.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if args.duyarlilik:
        oranlar = [0.5, 0.6, 0.7, 0.8, 0.9]
        ortalamalar = [ozetle(gunler, p)[0][0] for p in oranlar]
        fig, ax = plt.subplots()
        ax.plot(oranlar, ortalamalar, marker="o", label="B (KYC)")
        ax.axhline(mevcut[0], linestyle="--", color="gray", label="A (Mevcut)")
        ax.set(xlabel="Top-1 doğruluğu", ylabel="Ortalama bekleme (dk)", xticks=oranlar)
        ax.legend()
        fig.savefig(data / "sim_duyarlilik.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
