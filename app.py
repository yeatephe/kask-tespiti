import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Kask Tespiti", page_icon="🦺", layout="centered")

# ---- Görünüm (koyu + turuncu, güvenlik teması) ----
st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
h2 { border-left: 4px solid #F59E0B; padding-left: 12px; }
.stButton > button {
    background: linear-gradient(90deg, #D97706, #F59E0B);
    color: white; border: none; border-radius: 10px;
    padding: 0.55rem 1.4rem; font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(245,158,11,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(245,158,11,0.5);
}
div[data-testid="stMetric"] {
    background: #1A2332; border: 1px solid #2A3B52;
    border-radius: 12px; padding: 12px 16px;
}
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def modeli_yukle():
    # Egitilmis kendi modelimiz
    return YOLO("best.pt")


model = modeli_yukle()

st.title("İş Güvenliği Kask Tespiti")
st.markdown(
    "<p style='font-size:1.1rem; color:#94A3B8; margin-top:-0.5rem;'>"
    "Şantiye fotoğrafını yükle; yapay zeka kask takan ve takmayan işçileri saniyeler içinde tespit etsin."
    "</p>", unsafe_allow_html=True)

# 3 adım şeridi
st.markdown("""
<div style='display:flex; gap:10px; margin:1rem 0 1.5rem 0;'>
  <div style='flex:1; background:#1A2332; border:1px solid #2A3B52; border-radius:12px; padding:14px; text-align:center;'>
    <div style='font-size:1.6rem;'>📸</div>
    <div style='font-weight:600; color:#E5EDF5;'>1. Fotoğraf Yükle</div>
    <div style='font-size:0.8rem; color:#94A3B8;'>Şantiye görüntüsü seç</div>
  </div>
  <div style='flex:1; background:#1A2332; border:1px solid #2A3B52; border-radius:12px; padding:14px; text-align:center;'>
    <div style='font-size:1.6rem;'>🤖</div>
    <div style='font-weight:600; color:#E5EDF5;'>2. AI Tespit Etsin</div>
    <div style='font-size:0.8rem; color:#94A3B8;'>Kask ve kafalar işaretlenir</div>
  </div>
  <div style='flex:1; background:#1A2332; border:1px solid #2A3B52; border-radius:12px; padding:14px; text-align:center;'>
    <div style='font-size:1.6rem;'>🦺</div>
    <div style='font-weight:600; color:#E5EDF5;'>3. Raporu Gör</div>
    <div style='font-size:0.8rem; color:#94A3B8;'>Uyumluluk oranı hesaplanır</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.header("Fotoğraf Yükle")
yuklenen = st.file_uploader("Şantiye fotoğrafı", type=["jpg", "jpeg", "png"])

guven_esigi = st.slider(
    "Güven eşiği", 0.1, 0.9, 0.4, 0.05,
    help="Düşük değer daha çok tespit yapar ama hata payı artar."
)

if yuklenen is not None:
    resim = Image.open(yuklenen).convert("RGB")
    st.image(resim, caption="Yüklenen fotoğraf", use_container_width=True)

    if st.button("Tespit Et", type="primary"):
        with st.spinner("Yapay zeka fotoğrafı inceliyor..."):
            sonuc = model.predict(np.array(resim), conf=guven_esigi, verbose=False)[0]

            # Kutulu goruntuyu ciz
            kutulu = sonuc.plot()  # BGR numpy dizisi
            kutulu = kutulu[:, :, ::-1]  # BGR -> RGB
            st.image(kutulu, caption="Tespit sonucu", use_container_width=True)

            # Sayim
            sayim = {}
            for kutu in sonuc.boxes:
                ad = sonuc.names[int(kutu.cls)]
                sayim[ad] = sayim.get(ad, 0) + 1

            kaskli = sayim.get("helmet", 0)
            kasksiz = sayim.get("head", 0)
            toplam = kaskli + kasksiz

            st.header("Güvenlik Raporu")
            c1, c2, c3 = st.columns(3)
            c1.metric("🦺 Kasklı", kaskli)
            c2.metric("⚠️ Kasksız", kasksiz)
            c3.metric("👷 Toplam kişi", toplam)

            if toplam > 0:
                oran = kaskli / toplam * 100
                st.progress(oran / 100)
                st.write(f"**Kask uyumluluk oranı: %{oran:.0f}**")
                if kasksiz > 0:
                    st.error(f"⚠️ GÜVENLİK İHLALİ: {kasksiz} kişi kask takmıyor!")
                else:
                    st.success("✅ Tüm tespit edilen kişiler kask takıyor.")
            else:
                st.warning("Fotoğrafta kask veya kafa tespit edilemedi. "
                           "Daha net bir şantiye fotoğrafı deneyebilirsin.")

            st.caption("⚠️ Bu bir yapay zeka ön değerlendirmesidir; "
                       "resmi iş güvenliği denetimi yerine geçmez.")
