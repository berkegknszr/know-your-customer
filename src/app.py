"""CENG 495 şube demosu."""

import sqlite3
from contextlib import closing
from datetime import datetime
from importlib.util import find_spec

import streamlit as st

from predict import predict
from simulate import GISE_SAYISI, METRIKLER, SEED, musterileri_uret, ozetle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import face
except ImportError:
    face = None
else:
    # Yüz modülü DeepFace'i ilk fotoğraf işlenirken yükler.
    if find_spec("deepface") is None:
        face = None


BASLIK = "Know Your Customer — Before They Reach the Desk"
SON_ISLEMLER = {
    "login": "Giriş",
    "balance_view": "Bakiye görüntüleme",
    "transfer_ok": "Başarılı havale",
    "transfer_fail": "Başarısız havale",
    "card_payment": "Kart ödemesi",
    "card_declined": "Reddedilen kart ödemesi",
    "card_blocked": "Kart blokesi",
    "loan_calc_view": "Kredi hesaplama",
    "loan_page_view": "Kredi sayfası",
    "fx_rate_view": "Kur görüntüleme",
    "fx_trade": "Döviz işlemi",
    "bill_pay": "Fatura ödemesi",
    "otp_fail": "Başarısız doğrulama kodu",
    "password_reset": "Şifre sıfırlama",
    "atm_withdraw": "ATM para çekme",
    "atm_fail": "Başarısız ATM işlemi",
    "statement_view": "Ekstre görüntüleme",
    "chargeback_view": "İşlem itirazı görüntüleme",
    "product_page_view": "Ürün sayfası",
    "support_chat": "Destek görüşmesi",
}
NEDENLER = {
    "basarisiz havale": "Başarısız havale",
    "bloke kart": "Bloke kart",
    "kredi bilgisi": "Kredi bilgisi",
    "hesap acma": "Hesap açma",
    "kart basvurusu": "Kart başvurusu",
    "doviz islemi": "Döviz işlemi",
    "islem itirazi": "İşlem itirazı",
    "uygulama erisim sorunu": "Uygulama erişim sorunu",
}


def fotograf_al(alan):
    kamera = st.camera_input("Fotoğraf çek", key=f"{alan}_kamera")
    dosya = st.file_uploader(
        "Ya da fotoğraf yükle", type=["jpg", "jpeg", "png"], key=f"{alan}_dosya"
    )
    return dosya if dosya is not None else kamera


def musteri_no(alan):
    return int(st.number_input("Müşteri no", 0, 1999, 7, key=alan))


def tahminleri_goster(sonuc):
    for tahmin in sonuc["top3"]:
        olasilik = tahmin["olasilik"]
        neden = NEDENLER.get(tahmin["neden"], tahmin["neden"])
        st.progress(olasilik, text=f"{neden} · %{olasilik * 100:.0f}")


def sms_kaydet(customer_id, metin):
    with closing(sqlite3.connect("data/kyc.db")) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sms_log (
                zaman TEXT NOT NULL,
                customer_id INTEGER NOT NULL,
                metin TEXT NOT NULL
            )
        """)
        db.execute(
            "INSERT INTO sms_log (zaman, customer_id, metin) VALUES (?, ?, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"), customer_id, metin),
        )


def gelisi_kaydet(customer_id):
    try:
        sonuc = predict(customer_id, llm=False)
        sonuc["sms"] = sonuc["sms"].replace("Sirano", "Sıra no").replace("Gise", "Gişe")
        sonuc["sms"] = sonuc["sms"].replace("simule SMS", "simüle SMS")
        sms_kaydet(customer_id, sonuc["sms"])
    except ValueError:
        st.error("Müşteri bulunamadı.")
        return
    except sqlite3.Error:
        st.error("SMS kaydı yazılamadı. Tekrar deneyin.")
        return
    st.session_state["son_gelis"] = sonuc
    st.session_state.pop("temsilci_ozeti", None)
    st.session_state.pop("tanima_mesaji", None)


def sube_girisi():
    if face is None:
        st.warning("Yüz tanıma kullanılamıyor. Müşteri numarasıyla devam edin.")
    else:
        fotograf = fotograf_al("giris")
        if st.button("Fotoğrafla tanı", disabled=fotograf is None):
            try:
                with st.spinner("Fotoğraf kontrol ediliyor…"):
                    customer_id, mesafe = face.identify(fotograf.getvalue(), esik=0.68)
            except ImportError:
                customer_id, mesafe = None, None
                st.warning("Yüz tanıma kullanılamıyor. Müşteri numarasıyla devam edin.")
            except ValueError:
                customer_id, mesafe = None, None
            if customer_id is None:
                st.session_state["tanima_mesaji"] = (
                    "Opt-in değil / tanınmadı, normal sıra."
                )
            else:
                gelisi_kaydet(int(customer_id))
                st.success(f"Yüz tanındı: Müşteri {customer_id} (mesafe {mesafe})")

    customer_id = musteri_no("elle_musteri")
    if st.button("Müşteri no ile devam et"):
        gelisi_kaydet(customer_id)

    if "tanima_mesaji" in st.session_state:
        st.info(st.session_state["tanima_mesaji"])
        return

    sonuc = st.session_state.get("son_gelis")
    if sonuc:
        st.subheader(f'Müşteri {sonuc["customer_id"]} · Top-3')
        tahminleri_goster(sonuc)
        st.write("Simüle SMS")
        with st.chat_message("assistant", avatar="📱"):
            st.write(sonuc["sms"])


def temsilci_ekrani():
    sonuc = st.session_state.get("son_gelis")
    if sonuc is None:
        st.info("Şube girişinden bir müşteri kaydedin.")
        return

    customer_id = sonuc["customer_id"]
    st.subheader(f"Müşteri {customer_id}")
    st.write("Son işlemler (eskiden yeniye)")
    st.write(" → ".join(SON_ISLEMLER.get(islem, islem) for islem in sonuc["last_events"]))
    st.subheader("Top-3 tahmin")
    tahminleri_goster(sonuc)

    if st.button("Özeti hazırla"):
        with st.spinner("Yerel Qwen3 özeti hazırlanıyor…"):
            st.session_state["temsilci_ozeti"] = predict(customer_id, llm=True)["ozet"]

    ozet = st.session_state.get("temsilci_ozeti")
    if ozet and ozet.startswith("(LLM kullanilamiyor:"):
        ayrinti = ozet.removeprefix("(LLM kullanilamiyor: ").removesuffix(")")
        st.warning(f"Yerel özet hazırlanamadı. {ayrinti}")
    elif ozet:
        st.markdown(ozet)


def opt_in_kayit():
    if face is None:
        st.warning("Yüz kaydı kullanılamıyor. Şube girişinde müşteri numarasıyla devam edin.")
        return

    customer_id = musteri_no("kayit_musteri")
    consent = st.checkbox(
        "Açık rıza veriyorum. Yalnız yüz vektörü saklanır, ham fotoğraf saklanmaz. "
        "İstediğimde yüz vektörüm silinir."
    )
    fotograf = fotograf_al("kayit") if consent else None
    if st.button("Yüzü kaydet", disabled=fotograf is None):
        try:
            with st.spinner("Yüz kaydediliyor…"):
                face.enroll(customer_id, fotograf.getvalue(), consent=consent)
        except ImportError:
            st.warning("Yüz kaydı kullanılamıyor. Şube girişinde müşteri numarasıyla devam edin.")
        except ValueError as hata:
            st.error(f"Yüz kaydedilemedi. {hata}")
        else:
            st.success(f"Müşteri {customer_id} için yüz kaydı tamamlandı.")

    st.subheader("Kayıtlı müşteriler")
    kayitlar = face.list_enrolled()
    if kayitlar:
        silinecek = st.selectbox(
            "Rızası geri çekilecek müşteri no", [kayit["customer_id"] for kayit in kayitlar]
        )
        if st.button("Rızayı geri çek"):
            face.delete(silinecek)
            kayitlar = face.list_enrolled()
            st.success(f"Müşteri {silinecek} için yüz kaydı silindi.")
    if kayitlar:
        st.table(pd.DataFrame(kayitlar).rename(columns={
            "customer_id": "Müşteri no", "consent_at": "Rıza zamanı",
        }))
    else:
        st.caption("Yüz kaydı bulunmuyor.")


def sube_simulasyonu():
    gelis = st.slider("Saatlik ortalama müşteri gelişi", 20, 60, 40)
    top1 = st.slider("Top-1 doğruluğu", 0.50, 0.95, 0.815, step=0.005, format="%.3f")
    tekrar = st.slider("Tekrar (simüle edilen gün)", 1, 30, 10)
    st.caption(f"Her iki senaryo için aynı müşteri gelişleri, 8 saat ve {GISE_SAYISI} gişe.")

    if st.button("Simülasyonu çalıştır"):
        with st.spinner("Şube senaryoları karşılaştırılıyor…"):
            gunler = [list(musterileri_uret(gelis, SEED + i)) for i in range(tekrar)]
            mevcut, bekleme_a = ozetle(gunler)
            kyc, bekleme_b = ozetle(gunler, top1)
            st.session_state["simulasyon"] = (gelis, top1, tekrar, mevcut, kyc, bekleme_a, bekleme_b)

    sonuc = st.session_state.get("simulasyon")
    if sonuc is None or sonuc[:3] != (gelis, top1, tekrar):
        return

    _, _, _, mevcut, kyc, bekleme_a, bekleme_b = sonuc
    tablo = pd.DataFrame({"Mevcut": mevcut, "KYC": kyc}, index=METRIKLER)
    st.table(tablo.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.boxplot([bekleme_a, bekleme_b], vert=False, showfliers=False, widths=0.5)
    ax.set_yticks([1, 2], ["Mevcut", "KYC"])
    ax.set(xlabel="Kuyrukta bekleme (dk), uç değerler hariç")
    st.pyplot(fig)
    plt.close(fig)


def model_sonuclari():
    st.caption("Sentetik veri · 2.000 müşteri × 20 işlem · 8 sınıf")
    st.table(pd.DataFrame(
        {"Top-1": ["0.705", "0.815"], "Top-3": ["0.930", "0.922"], "Makro-F1": ["0.700", "0.811"]},
        index=["LightGBM", "GRU"],
    ))


st.set_page_config(page_title=BASLIK, layout="wide")
st.title(BASLIK)
giris, temsilci, kayit, simulasyon, modeller = st.tabs([
    "Şube girişi",
    "Temsilci ekranı",
    "Opt-in kayıt (KVKK)",
    "Şube simülasyonu",
    "Model sonuçları",
])
with giris:
    sube_girisi()
with temsilci:
    temsilci_ekrani()
with kayit:
    opt_in_kayit()
with simulasyon:
    sube_simulasyonu()
with modeller:
    model_sonuclari()
