import streamlit as st
import sys, os, io, tempfile

sys.path.insert(0, os.path.dirname(__file__))
from tespit import TespitModeli, MODEL_DOSYA, VERI_DIR
import pandas as pd
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder

# ─── Sayfa ayarları ───────────────────────────────────────
st.set_page_config(
    page_title="Saldırı Niyeti Tespiti",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Saldırı Niyeti Tespit Sistemi")
st.caption("TF-IDF N-gram + Markov Zinciri · Türkçe 42K tweet ile eğitildi")

# ─── Model yükle (cache) ──────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor…")
def model_getir():
    if os.path.exists(MODEL_DOSYA):
        return TespitModeli.yukle()
    # Model yoksa eğit
    train_df = pd.read_csv(os.path.join(VERI_DIR, "train.csv"))
    valid_df  = pd.read_csv(os.path.join(VERI_DIR, "valid.csv"))
    df = pd.concat([train_df, valid_df], ignore_index=True)[["text", "label"]].dropna()
    tehdit = pd.DataFrame({
        "text": [
            "seni öldüreceğim","seni geberteceim","canına okuyacağım",
            "hepinizi mahvedeceğim","kafanı kıracağım","sana zarar vereceğim",
            "evini yakacağım","seni bitireceğim","boğazını sıkacağım",
            "seni parçalayacağım","kanını dökeceğim","seni ezeceğim",
            "seni yok edeceğim","hepsini öldüreceğim","seni bıçaklayacağım",
            "seni vuracağım","bulursam öldürürüm","kafanı uçuracağım",
            "ağzını burnunu kırarım","bir daha görürsem sonun olur",
        ],
        "label": [1] * 20
    })
    df = pd.concat([df, tehdit], ignore_index=True)
    m = TespitModeli()
    m.egit(df)
    m.kaydet()
    return m

model = model_getir()

# ─── Mikrofon ─────────────────────────────────────────────
st.divider()
st.markdown("**🎤 Sesle Gir**")
audio_bytes = audio_recorder(
    text="Kaydet",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_size="2x",
)

ses_metin = ""
if audio_bytes:
    with st.spinner("Ses tanınıyor…"):
        try:
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
            ses_metin = recognizer.recognize_google(audio_data, language="tr-TR")
            st.success(f"Tanınan: **{ses_metin}**")
        except sr.UnknownValueError:
            st.warning("Ses anlaşılamadı, tekrar deneyin.")
        except Exception as e:
            st.error(f"Hata: {e}")
        finally:
            os.unlink(tmp_path)

# ─── Giriş alanı ─────────────────────────────────────────
st.markdown("**✏️ Yazıyla Gir**")
metin = st.text_area(
    "Analiz edilecek metni girin:",
    value=ses_metin,
    placeholder="Örnek: seni bulursam hesabını sorarım",
    height=100,
)

analiz_btn = st.button("🔎 Analiz Et", type="primary", use_container_width=True)

# ─── Sonuç ───────────────────────────────────────────────
if analiz_btn and metin.strip():
    r = model.tahmin(metin.strip())

    if r["etiket"] == 1:
        st.error(f"### {r['label_str']}", icon="🚨")
    else:
        st.success(f"### {r['label_str']}", icon="✅")

    col1, col2, col3 = st.columns(3)
    col1.metric("Normal",     f"{r['p_normal']:.1%}")
    col2.metric("Saldırgan",  f"{r['p_saldirgan']:.1%}")
    col3.metric("Markov Δ",   f"{r['markov_delta']:+.3f}")

    st.divider()

    st.markdown("**Olasılık Dağılımı**")
    st.progress(r["p_saldirgan"], text=f"🔴 Saldırgan  {r['p_saldirgan']:.1%}")
    st.progress(r["p_normal"],    text=f"🟢 Normal     {r['p_normal']:.1%}")

    with st.expander("Detay"):
        st.write(f"**Temizlenmiş metin:** `{r['temiz']}`")
        markov_yorum = "saldırgan dile yakın" if r["markov_delta"] > 0 else "normal dile yakın"
        st.write(f"**Markov Δ:** `{r['markov_delta']:+.3f}` → {markov_yorum}")

elif analiz_btn:
    st.warning("Lütfen bir metin girin.")

# ─── Toplu analiz ────────────────────────────────────────
st.divider()
with st.expander("📋 Toplu Analiz (birden fazla satır)"):
    toplu = st.text_area(
        "Her satıra bir metin girin:",
        height=150,
        key="toplu",
    )
    if st.button("Toplu Analiz Et", use_container_width=True):
        satirlar = [s.strip() for s in toplu.splitlines() if s.strip()]
        if satirlar:
            sonuclar = [model.tahmin(s) for s in satirlar]
            rows = []
            for r in sonuclar:
                rows.append({
                    "Metin": r["metin"],
                    "Sonuç": r["label_str"],
                    "Saldırgan %": f"{r['p_saldirgan']:.1%}",
                    "Normal %": f"{r['p_normal']:.1%}",
                    "Markov Δ": f"{r['markov_delta']:+.3f}",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.warning("Metin girilmedi.")
