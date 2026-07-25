import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import numpy as np
import os, glob, time, cv2

st.set_page_config(page_title="KKD Denetim Sistemi", layout="wide")

def ihlal_mi(ad):
    a = ad.lower()
    return a.startswith("no-") or a in {"head", "fall-detected"}

@st.cache_resource
def model_yukle(yol):
    return YOLO(yol)

# ---------- Kenar cubugu ----------
st.sidebar.title("Ayarlar")

kip = st.sidebar.radio("Calisma kipi",
                       ["Tek model", "Birlesik (baret + KKD)"])

esik = st.sidebar.slider("Guven esigi", 0.10, 0.90, 0.35, 0.05)

modeller = [os.path.basename(p) for p in glob.glob("*.pt")]

# Birlesik kipte baret modeli 'head' -> 'NO-Hardhat', 'helmet' -> 'Hardhat'
BARET_ESLEME = {"head": "NO-Hardhat", "helmet": "Hardhat"}

if kip == "Tek model":
    if not modeller:
        st.error("Klasorde .pt model bulunamadi.")
        st.stop()
    secilen = st.sidebar.selectbox("Model", modeller)
    aktif = [(model_yukle(secilen), None)]
else:
    if "baret_model.pt" not in modeller:
        st.error("baret_model.pt bulunamadi.")
        st.stop()
    kkd_secenek = [m for m in modeller if m.startswith("kkd_model")]
    kkd_secilen = st.sidebar.selectbox("KKD modeli", kkd_secenek)
    aktif = [
        (model_yukle("baret_model.pt"), BARET_ESLEME),
        (model_yukle(kkd_secilen), None),
    ]
    st.sidebar.caption("Baret tespiti yuksek dogruluklu baret modeline, "
                       "diger ekipmanlar KKD modeline yaptirilir.")

def analiz_et(kaynak_img):
    """Verilen goruntude tum aktif modelleri calistirir, birlesik sonuc doner."""
    tum_isimler = []
    ilk_cizim = None
    for m, esleme in aktif:
        r = m.predict(kaynak_img, conf=esik, verbose=False)[0]
        if ilk_cizim is None:
            ilk_cizim = r.plot()
        else:
            # ikinci modelin kutularini ayni goruntuye ciz
            ilk_cizim = r.plot(img=ilk_cizim)
        for c in r.boxes.cls:
            ad = r.names[int(c)]
            if esleme is not None:
                if ad not in esleme:
                    continue
                ad = esleme[ad]
            tum_isimler.append(ad)
    return ilk_cizim, tum_isimler

# ---------- Ana ekran ----------
st.title("KKD Denetim Sistemi")

sekme1, sekme2 = st.tabs(["Goruntu analizi", "Canli kamera"])

with sekme1:
    dosya = st.file_uploader("Denetlenecek goruntuyu yukleyin",
                             type=["jpg", "jpeg", "png", "bmp"])
    if dosya:
        img = Image.open(dosya).convert("RGB")
        cizim, isimler = analiz_et(img)
        sayim = Counter(isimler)
        ihlaller = Counter({k: v for k, v in sayim.items() if ihlal_mi(k)})
        uygunlar = Counter({k: v for k, v in sayim.items() if not ihlal_mi(k)})

        k1, k2, k3 = st.columns(3)
        k1.metric("Toplam tespit", len(isimler))
        k2.metric("Ihlal sayisi", sum(ihlaller.values()))
        k3.metric("Durum", "IHLAL VAR" if ihlaller else "UYGUN")

        sol, sag = st.columns([3, 2])
        with sol:
            st.image(cizim[:, :, ::-1], use_container_width=True)
        with sag:
            st.subheader("Denetim raporu")
            if ihlaller:
                for ad, adet in ihlaller.most_common():
                    st.error(f"{ad} — {adet} adet")
            else:
                st.success("Kural ihlali tespit edilmedi.")
            if uygunlar:
                st.markdown("**Uygun kullanim**")
                for ad, adet in uygunlar.most_common():
                    st.write(f"- {ad}: {adet}")
    else:
        st.info("Baslamak icin bir goruntu yukleyin.")

with sekme2:
    kaynak_tipi = st.radio("Kaynak", ["Webcam", "IP kamera / video"],
                           horizontal=True)
    if kaynak_tipi == "Webcam":
        kaynak = st.number_input("Kamera indeksi", 0, 5, 0)
    else:
        kaynak = st.text_input("RTSP adresi veya video dosyasi", "")

    c1, c2, c3 = st.columns(3)
    kare_atla = c1.number_input("Kare atlama", 1, 10, 2)
    cooldown = c2.number_input("Alarm bekleme (sn)", 1, 60, 10)
    kaydet = c3.checkbox("Ihlalleri kaydet", True)

    if "calisiyor" not in st.session_state:
        st.session_state.calisiyor = False
    b1, b2 = st.columns(2)
    if b1.button("Baslat", use_container_width=True):
        st.session_state.calisiyor = True
    if b2.button("Durdur", use_container_width=True):
        st.session_state.calisiyor = False

    ekran = st.empty()
    durum = st.empty()
    kayit_alani = st.empty()

    if st.session_state.calisiyor:
        os.makedirs("alarmlar", exist_ok=True)
        cap = cv2.VideoCapture(int(kaynak) if kaynak_tipi == "Webcam" else kaynak)
        if not cap.isOpened():
            st.error("Kamera acilamadi.")
            st.session_state.calisiyor = False
        else:
            sayac, son_alarm, kayitlar = 0, 0, []
            while st.session_state.calisiyor:
                ok, kare = cap.read()
                if not ok:
                    st.warning("Goruntu alinamadi.")
                    break
                sayac += 1
                if sayac % kare_atla:
                    continue
                cizim, isimler = analiz_et(kare)
                ihlaller = sorted({i for i in isimler if ihlal_mi(i)})
                ekran.image(cizim[:, :, ::-1], channels="RGB",
                            use_container_width=True)
                if ihlaller:
                    durum.error("IHLAL: " + ", ".join(ihlaller))
                    if kaydet and time.time() - son_alarm > cooldown:
                        son_alarm = time.time()
                        ad = time.strftime("alarmlar/%Y%m%d_%H%M%S.jpg")
                        cv2.imwrite(ad, cizim)
                        kayitlar.insert(0, f"{time.strftime('%H:%M:%S')} — {', '.join(ihlaller)}")
                        kayit_alani.write("**Alarm gecmisi**\n\n" +
                                          "\n".join(f"- {k}" for k in kayitlar[:10]))
                else:
                    durum.success("Uygun")
            cap.release()