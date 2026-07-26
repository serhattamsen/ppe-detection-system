import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import pandas as pd
import os, glob, time, cv2
import auth

st.set_page_config(page_title="KKD Denetim Sistemi", layout="wide")

def ihlal_mi(ad):
    a = ad.lower()
    return a.startswith("no-") or a in {"head", "fall-detected"}

@st.cache_resource
def model_yukle(yol):
    return YOLO(yol)

# ================== GIRIS EKRANI ==================
if "kullanici" not in st.session_state:
    st.session_state.kullanici = None

if st.session_state.kullanici is None:
    st.title("KKD Denetim Sistemi")
    sekme_giris, sekme_kayit = st.tabs(["Giris", "Kayit Ol"])

    with sekme_giris:
        k = st.text_input("Kullanici adi", key="g_k")
        s = st.text_input("Sifre", type="password", key="g_s")
        if st.button("Giris yap"):
            if auth.giris_yap(k, s):
                st.session_state.kullanici = k
                st.rerun()
            else:
                st.error("Kullanici adi veya sifre hatali")

    with sekme_kayit:
        yk = st.text_input("Yeni kullanici adi", key="k_k")
        ys = st.text_input("Yeni sifre", type="password", key="k_s")
        if st.button("Kayit ol"):
            if len(yk) < 3 or len(ys) < 4:
                st.warning("Kullanici adi en az 3, sifre en az 4 karakter olmali")
            else:
                ok, mesaj = auth.kayit_ol(yk, ys)
                if ok:
                    st.success(mesaj + ". Simdi giris yapabilirsiniz.")
                else:
                    st.error(mesaj)
    st.stop()

# ================== GIRIS YAPILDI ==================
kullanici = st.session_state.kullanici

st.sidebar.title(f"Merhaba, {kullanici}")
if st.sidebar.button("Cikis yap"):
    st.session_state.kullanici = None
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Ayarlar")

kip = st.sidebar.radio("Calisma kipi", ["Tek model", "Birlesik (baret + KKD)"])
esik = st.sidebar.slider("Guven esigi", 0.10, 0.90, 0.35, 0.05)

modeller = [os.path.basename(p) for p in glob.glob("*.pt")]
BARET_ESLEME = {"head": "NO-Hardhat", "helmet": "Hardhat"}

if kip == "Tek model":
    secilen = st.sidebar.selectbox("Model", modeller)
    aktif = [(model_yukle(secilen), None)]
else:
    kkd_secenek = [m for m in modeller if m.startswith("kkd_model")]
    kkd_secilen = st.sidebar.selectbox("KKD modeli", kkd_secenek)
    aktif = [
        (model_yukle("baret_model.pt"), BARET_ESLEME),
        (model_yukle(kkd_secilen), None),
    ]

def analiz_et(kaynak_img):
    tum_isimler, tum_guven = [], []
    cizim = None
    for m, esleme in aktif:
        r = m.predict(kaynak_img, conf=esik, verbose=False)[0]
        cizim = r.plot() if cizim is None else r.plot(img=cizim)
        for c, cf in zip(r.boxes.cls, r.boxes.conf):
            ad = r.names[int(c)]
            if esleme is not None:
                if ad not in esleme:
                    continue
                ad = esleme[ad]
            tum_isimler.append(ad)
            tum_guven.append(float(cf))
    return cizim, tum_isimler, tum_guven

# ================== ANA SEKMELER ==================
st.title("KKD Denetim Sistemi")
s1, s2, s3 = st.tabs(["Goruntu analizi", "Canli kamera", "Dashboard"])

# ---------- Goruntu ----------
with s1:
    dosya = st.file_uploader("Goruntu yukleyin", type=["jpg", "jpeg", "png", "bmp"])
    if dosya:
        img = Image.open(dosya).convert("RGB")
        cizim, isimler, guvenler = analiz_et(img)
        sayim = Counter(isimler)
        ihlaller = Counter({k: v for k, v in sayim.items() if ihlal_mi(k)})

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
                # ihlalleri veritabanina kaydet
                if st.button("Bu denetimi kaydet"):
                    os.makedirs("alarmlar", exist_ok=True)
                    zaman = time.strftime("%Y-%m-%d %H:%M:%S")
                    yol = time.strftime("alarmlar/%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(yol, cizim)
                    for ad, cf in zip(isimler, guvenler):
                        if ihlal_mi(ad):
                            auth.alarm_kaydet(kullanici, "Goruntu yukleme",
                                              ad, cf, zaman, yol)
                    st.success("Denetim kaydedildi. Dashboard'da gorebilirsiniz.")
            else:
                st.success("Kural ihlali tespit edilmedi.")
    else:
        st.info("Baslamak icin bir goruntu yukleyin.")

# ---------- Canli kamera ----------
with s2:
    st.caption("Birden fazla kamera tanimlayabilirsiniz. Her satir: isim = kaynak")
    varsayilan = "Webcam = 0"
    kamera_metni = st.text_area("Kamera listesi", varsayilan, height=100,
        help="Ornek:\nGiris = 0\nDepo = rtsp://kullanici:sifre@192.168.1.64:554/Streaming/Channels/101")

    kameralar = {}
    for satir in kamera_metni.strip().splitlines():
        if "=" in satir:
            isim, kaynak = satir.split("=", 1)
            kameralar[isim.strip()] = kaynak.strip()

    if kameralar:
        secili_kamera = st.selectbox("Aktif kamera", list(kameralar.keys()))
        kaynak = kameralar[secili_kamera]

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

        if st.session_state.calisiyor:
            os.makedirs("alarmlar", exist_ok=True)
            kaynak_deger = int(kaynak) if kaynak.isdigit() else kaynak
            cap = cv2.VideoCapture(kaynak_deger)
            if not cap.isOpened():
                st.error("Kamera acilamadi.")
                st.session_state.calisiyor = False
            else:
                sayac, son_alarm = 0, 0
                while st.session_state.calisiyor:
                    ok, kare = cap.read()
                    if not ok:
                        st.warning("Goruntu alinamadi.")
                        break
                    sayac += 1
                    if sayac % kare_atla:
                        continue
                    cizim, isimler, guvenler = analiz_et(kare)
                    ihlaller = sorted({i for i in isimler if ihlal_mi(i)})
                    ekran.image(cizim[:, :, ::-1], channels="RGB",
                                use_container_width=True)
                    if ihlaller:
                        durum.error("IHLAL: " + ", ".join(ihlaller))
                        if kaydet and time.time() - son_alarm > cooldown:
                            son_alarm = time.time()
                            zaman = time.strftime("%Y-%m-%d %H:%M:%S")
                            yol = time.strftime("alarmlar/%Y%m%d_%H%M%S.jpg")
                            cv2.imwrite(yol, cizim)
                            for ad, cf in zip(isimler, guvenler):
                                if ihlal_mi(ad):
                                    auth.alarm_kaydet(kullanici, secili_kamera,
                                                      ad, cf, zaman, yol)
                    else:
                        durum.success("Uygun")
                cap.release()

# ---------- Dashboard ----------
with s3:
    st.subheader("Denetim Analizleri")
    kayitlar = auth.alarmlari_getir(kullanici)

    if not kayitlar:
        st.info("Henuz kayitli alarm yok. Analiz yapip kaydettikce burada gorunecek.")
    else:
        df = pd.DataFrame(kayitlar,
            columns=["Kamera", "Ekipman", "Guven", "Tarih", "Resim"])
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df["Saat"] = df["Tarih"].dt.hour

        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam ihlal", len(df))
        m2.metric("Kamera sayisi", df["Kamera"].nunique())
        m3.metric("En sik ihlal", df["Ekipman"].mode()[0])

        st.markdown("**En cok eksik kullanilan KKD**")
        st.bar_chart(df["Ekipman"].value_counts())

        st.markdown("**Kamera bazli ihlal dagilimi**")
        st.bar_chart(df["Kamera"].value_counts())

        st.markdown("**Gunun saatlerine gore ihlaller**")
        saat_dagilim = df["Saat"].value_counts().sort_index()
        st.bar_chart(saat_dagilim)

        st.markdown("**Son alarmlar**")
        st.dataframe(df[["Tarih", "Kamera", "Ekipman", "Guven"]].head(20),
                     use_container_width=True)