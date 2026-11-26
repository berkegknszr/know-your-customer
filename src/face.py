"""Opt-in yuz tanima (DeepFace / ArcFace) + SQLite.

KVKK: Ham fotograf ASLA saklanmaz. Yalnizca acik riza veren musterinin
512 boyutlu yuz vektoru (embedding) data/kyc.db icindeki faces tablosuna yazilir.
Musteri rizasini geri cekerse delete() vektoru siler.

Kullanim:
  PYTHONPATH=src python src/face.py kaydet 7 foto.jpg
  PYTHONPATH=src python src/face.py tani foto.jpg
  PYTHONPATH=src python src/face.py sil 7
  PYTHONPATH=src python src/face.py liste
"""
import argparse, sqlite3
from datetime import datetime
import numpy as np
import cv2

DB = "data/kyc.db"
MODEL = "ArcFace"
ESIK = 0.68  # ArcFace + kosinus icin DeepFace'in onerdigi esik


def _db():
    db = sqlite3.connect(DB)
    db.execute("""create table if not exists faces(
        customer_id integer primary key,
        embedding blob,
        consent_at text,
        consent integer)""")
    return db


def _resim(img):
    """Dosya yolu, BGR numpy dizisi veya JPEG/PNG bytes kabul eder -> BGR numpy."""
    if isinstance(img, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("resim okunamadi")
    elif isinstance(img, str):
        import os
        if not os.path.exists(img):
            raise FileNotFoundError(img)
    return img  # dosya yolu ise DeepFace kendisi okur


def embed(img):
    """Resimdeki yuzun ArcFace vektorunu dondurur. Yuz yoksa ValueError."""
    from deepface import DeepFace  # agir kutuphane, yalniz gerekince yuklenir
    try:
        r = DeepFace.represent(img_path=_resim(img), model_name=MODEL,
                               detector_backend="opencv", enforce_detection=True)
    except ValueError:
        raise ValueError("yuz bulunamadi")
    v = np.array(r[0]["embedding"], dtype=np.float32)
    return v / np.linalg.norm(v)  # birim vektor: kosinus = ic carpim


def enroll(customer_id: int, img, consent: bool):
    """Acik riza varsa musterinin yuz vektorunu kaydeder (fotograf kaydedilmez)."""
    if not consent:
        raise ValueError("acik riza yok, kayit yapilmadi")
    v = embed(img)
    db = _db()
    db.execute("insert or replace into faces values (?,?,?,1)",
               (int(customer_id), v.tobytes(), datetime.now().isoformat(timespec="seconds")))
    db.commit()
    return True


def identify(img, esik=ESIK):
    """En yakin kayitli yuzu bulur. Esigin altindaysa (customer_id, mesafe), degilse (None, mesafe)."""
    v = embed(img)
    rows = _db().execute("select customer_id, embedding from faces where consent=1").fetchall()
    if not rows:
        return None, 1.0
    ids = [r[0] for r in rows]
    M = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    mesafe = 1 - M @ v  # kosinus mesafesi
    i = int(np.argmin(mesafe))
    d = float(mesafe[i])
    return (ids[i] if d < esik else None), round(d, 3)


def delete(customer_id: int):
    """Riza geri cekildi: vektoru tamamen siler."""
    db = _db()
    n = db.execute("delete from faces where customer_id=?", (int(customer_id),)).rowcount
    db.commit()
    return n > 0


def list_enrolled():
    return [{"customer_id": r[0], "consent_at": r[1]}
            for r in _db().execute("select customer_id, consent_at from faces order by customer_id")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Opt-in yuz tanima")
    sp = ap.add_subparsers(dest="komut", required=True)
    k = sp.add_parser("kaydet"); k.add_argument("id", type=int); k.add_argument("resim")
    t = sp.add_parser("tani"); t.add_argument("resim")
    s = sp.add_parser("sil"); s.add_argument("id", type=int)
    sp.add_parser("liste")
    a = ap.parse_args()
    if a.komut == "kaydet":
        enroll(a.id, a.resim, consent=True); print("kaydedildi:", a.id)
    elif a.komut == "tani":
        print("sonuc (musteri, mesafe):", identify(a.resim))
    elif a.komut == "sil":
        print("silindi" if delete(a.id) else "kayit yok")
    else:
        for r in list_enrolled(): print(r)
