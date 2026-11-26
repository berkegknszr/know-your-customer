"""Yuz tanima basarisi: FAR / FRR olcumu (DeepFace test seti, 299 etiketli cift).

Veri: DeepFace kutuphanesinin acik test seti (tests/unit/dataset, master.csv).
  data/yuz_test/ klasorune indirilir (README'ye bakin).
FAR (False Accept Rate)  = farkli kisiyi ayni sanma orani  (guvenlik riski)
FRR (False Reject Rate)  = ayni kisiyi reddetme orani      (musteri normal siraya duser)

Calistir: PYTHONPATH=src python src/face_eval.py
Cikti: terminal tablosu + data/yuz_far_frr.png + data/yuz_sonuc.csv
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from face import embed, ESIK

KLASOR = "data/yuz_test/"

ciftler = list(csv.DictReader(open(KLASOR + "master.csv")))
vektor = {}  # her resim bir kez hesaplanir
for c in ciftler:
    for f in (c["file_x"], c["file_y"]):
        if f not in vektor:
            try:
                vektor[f] = embed(KLASOR + f)
            except ValueError:
                vektor[f] = None  # yuz bulunamadi

mesafe, ayni = [], []
atlanan = 0
for c in ciftler:
    a, b = vektor[c["file_x"]], vektor[c["file_y"]]
    if a is None or b is None:
        atlanan += 1
        continue
    mesafe.append(1 - float(a @ b))
    ayni.append(c["Decision"] == "Yes")
mesafe, ayni = np.array(mesafe), np.array(ayni)


def far_frr(esik):
    far = np.mean(mesafe[~ayni] < esik)   # farkli kisi ama kabul edildi
    frr = np.mean(mesafe[ayni] >= esik)   # ayni kisi ama reddedildi
    return far, frr


print(f"Cift sayisi: {len(mesafe)} (ayni kisi {ayni.sum()}, farkli {(~ayni).sum()}), yuz bulunamayan cift: {atlanan}")
print(f"{'Esik':>6} {'FAR %':>8} {'FRR %':>8}")
satirlar = []
for e in [0.40, 0.50, 0.60, ESIK, 0.75]:
    far, frr = far_frr(e)
    satirlar.append((e, far, frr))
    print(f"{e:>6.2f} {far*100:>8.2f} {frr*100:>8.2f}" + ("   <- sistemde kullanilan" if e == ESIK else ""))

with open("data/yuz_sonuc.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["esik", "FAR", "FRR"]); w.writerows(satirlar)

es = np.linspace(0.2, 1.0, 81)
FAR, FRR = zip(*[far_frr(e) for e in es])
plt.figure(figsize=(6, 4))
plt.plot(es, np.array(FAR) * 100, label="FAR (yanlis kabul)")
plt.plot(es, np.array(FRR) * 100, label="FRR (yanlis ret)")
plt.axvline(ESIK, color="gray", ls="--", label=f"esik {ESIK}")
plt.xlabel("Kosinus mesafesi esigi"); plt.ylabel("Oran (%)")
plt.title("ArcFace yuz dogrulama: FAR / FRR"); plt.legend(); plt.tight_layout()
plt.savefig("data/yuz_far_frr.png", dpi=150)
print("Grafik: data/yuz_far_frr.png")
