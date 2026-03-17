# ─────────────────────────────────────────────────────────────────
# Saldırı Niyeti Tespit Sistemi
# TF-IDF N-gram + Markov Zinciri + Lojistik Regresyon
# Dataset: archive-12/train.csv (42K Türkçe tweet)
# ─────────────────────────────────────────────────────────────────
# pip install scikit-learn pandas nltk speechrecognition pyaudio
# ─────────────────────────────────────────────────────────────────

import os, re, pickle, math
from collections import defaultdict

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

VERI_DIR    = os.path.join(os.path.dirname(__file__), "archive-12")
MODEL_DOSYA = os.path.join(os.path.dirname(__file__), "niyet_model.pkl")

ETIKETLER = {0: "🟢 Normal", 1: "🔴 Saldırgan / Tehdit"}

# ─────────────────────────────────────────
# 1. METIN ÖN İŞLEME
# ─────────────────────────────────────────

try:
    TR_STOP = set(stopwords.words("turkish"))
except OSError:
    TR_STOP = set()

def temizle(metin: str) -> str:
    metin = str(metin).lower()
    metin = re.sub(r"@\w+|http\S+|#\w+", " ", metin)  # mention, url, hashtag
    metin = re.sub(r"[^\w\s]", " ", metin)
    metin = re.sub(r"\d+", " ", metin)
    kelimeler = word_tokenize(metin, language="turkish")
    kelimeler = [k for k in kelimeler if k not in TR_STOP and len(k) > 1]
    return " ".join(kelimeler)


# ─────────────────────────────────────────
# 2. MARKOV ZİNCİRİ
#    Saldırgan metinlerden bigram geçiş
#    olasılıklarını öğrenir.  Bir metin
#    için log-olasılık skoru hesaplar.
# ─────────────────────────────────────────

def _dd_int():
    return defaultdict(int)


class MarkovDil:
    """Bigram Markov dil modeli (log-olasılık skoru)."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.bigram: dict[str, dict[str, int]] = defaultdict(_dd_int)
        self.unigram: dict[str, int] = defaultdict(int)

    def egit(self, metinler: list[str]):
        for metin in metinler:
            kelimeler = metin.split()
            for k in kelimeler:
                self.unigram[k] += 1
            for i in range(len(kelimeler) - 1):
                self.bigram[kelimeler[i]][kelimeler[i + 1]] += 1

    def log_olasilik(self, metin: str) -> float:
        """Metnin bu modele göre normalize log-olasılığı."""
        kelimeler = metin.split()
        if len(kelimeler) < 2:
            return 0.0
        V = len(self.unigram)
        toplam = 0.0
        for i in range(len(kelimeler) - 1):
            onceki = kelimeler[i]
            sonraki = kelimeler[i + 1]
            pay = self.bigram[onceki][sonraki] + self.smoothing
            payda = self.unigram[onceki] + self.smoothing * V
            toplam += math.log(pay / payda)
        return toplam / (len(kelimeler) - 1)   # uzunluktan bağımsız hale getir


# ─────────────────────────────────────────
# 3. BÜTÜNLEŞIK MODEL
# ─────────────────────────────────────────

class TespitModeli:
    """
    TF-IDF (unigram→trigram) + Lojistik Regresyon pipeline'ı
    ile Markov skor farkını ek özellik olarak birleştirir.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=20_000,
                sublinear_tf=True,
            )),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
            )),
        ])
        self.markov_saldirgan = MarkovDil()
        self.markov_normal    = MarkovDil()
        self.markov_agirlik   = 0.15   # Markov etkisinin karıştırma oranı

    # — Eğitim ————————————————————————————

    def egit(self, df: pd.DataFrame):
        df["temiz"] = df["text"].apply(temizle)

        saldirgan = df[df["label"] == 1]["temiz"].tolist()
        normal    = df[df["label"] == 0]["temiz"].tolist()

        print(f"  Markov modelleri eğitiliyor… ({len(saldirgan)} saldırgan, {len(normal)} normal)")
        self.markov_saldirgan.egit(saldirgan)
        self.markov_normal.egit(normal)

        X_train, X_test, y_train, y_test = train_test_split(
            df["temiz"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
        )

        print("  TF-IDF pipeline eğitiliyor…")
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        print("\n📊 Model Performansı (TF-IDF + LR):")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Saldırgan"]))

    # — Tahmin ————————————————————————————

    def tahmin(self, metin: str) -> dict:
        temiz = temizle(metin)

        # TF-IDF + LR olasılıkları
        olasiliklar = self.pipeline.predict_proba([temiz])[0]
        p_normal, p_saldirgan = float(olasiliklar[0]), float(olasiliklar[1])

        # Markov delta skoru: saldırgan dile ne kadar benziyor?
        ms = self.markov_saldirgan.log_olasilik(temiz)
        mn = self.markov_normal.log_olasilik(temiz)
        markov_delta = ms - mn          # pozitif → saldırgan dile daha yakın

        # Yumuşak birleştirme: Markov farkını saldırgan olasılığına ek
        # 0–1 arasına sıkıştır (sigmoid benzeri normalizasyon)
        def sigmoid(x): return 1 / (1 + math.exp(-x))
        markov_katkisi = sigmoid(markov_delta) - 0.5   # -0.5 … +0.5

        p_saldirgan_ayarli = p_saldirgan + self.markov_agirlik * markov_katkisi
        p_saldirgan_ayarli = max(0.0, min(1.0, p_saldirgan_ayarli))
        p_normal_ayarli    = 1.0 - p_saldirgan_ayarli

        etiket = 1 if p_saldirgan_ayarli >= 0.5 else 0

        return {
            "metin": metin,
            "temiz": temiz,
            "etiket": etiket,
            "label_str": ETIKETLER[etiket],
            "p_saldirgan": p_saldirgan_ayarli,
            "p_normal": p_normal_ayarli,
            "markov_delta": markov_delta,
        }

    # — Kayıt / Yükleme ———————————————————

    def kaydet(self, dosya: str = MODEL_DOSYA):
        with open(dosya, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model kaydedildi → {dosya}")

    @staticmethod
    def yukle(dosya: str = MODEL_DOSYA) -> "TespitModeli":
        with open(dosya, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────
# 4. ÇIKTI YARDIMCISI
# ─────────────────────────────────────────

def sonuc_yazdir(r: dict):
    bar = lambda p: "█" * int(p * 30)
    print(f"\n{'─'*55}")
    print(f"📝 Metin        : {r['metin']}")
    print(f"🔍 Sonuç        : {r['label_str']}")
    print(f"📈 Olasılıklar  :")
    print(f"   {'🟢 Normal':22s} {bar(r['p_normal']):<32s} {r['p_normal']:.1%}")
    print(f"   {'🔴 Saldırgan':22s} {bar(r['p_saldirgan']):<32s} {r['p_saldirgan']:.1%}")
    print(f"🧮 Markov Δ     : {r['markov_delta']:+.3f}  "
          f"({'saldırgan dile yakın' if r['markov_delta'] > 0 else 'normal dile yakın'})")
    print(f"{'─'*55}")


# ─────────────────────────────────────────
# 5. SES TANIMA (opsiyonel)
# ─────────────────────────────────────────

def konusmayi_dinle() -> str | None:
    try:
        import speech_recognition as sr
    except ImportError:
        print("⚠️  speechrecognition kurulu değil: pip install speechrecognition pyaudio")
        return None

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Konuşun… (Ctrl+C ile iptal)")
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=8)
        except sr.WaitTimeoutError:
            print("⏱ Zaman aşımı")
            return None

    try:
        metin = r.recognize_google(audio, language="tr-TR")
        print(f"✅ Tanınan: '{metin}'")
        return metin
    except sr.UnknownValueError:
        print("❌ Ses anlaşılamadı")
        return None
    except sr.RequestError as e:
        print(f"❌ API hatası: {e}")
        return None


# ─────────────────────────────────────────
# 6. ANA PROGRAM
# ─────────────────────────────────────────

def main():
    print("=" * 55)
    print("   Saldırı Niyeti Tespit Sistemi  (N-gram + Markov)")
    print("=" * 55)

    # Model yükle veya eğit
    if os.path.exists(MODEL_DOSYA):
        print(f"\n📂 Kayıtlı model yükleniyor: {MODEL_DOSYA}")
        model = TespitModeli.yukle()
    else:
        print("\n⚙️  Model eğitiliyor (ilk çalıştırma)…")
        train_df = pd.read_csv(os.path.join(VERI_DIR, "train.csv"))
        # valid seti de eğitime ekle → daha fazla veri
        valid_df = pd.read_csv(os.path.join(VERI_DIR, "valid.csv"))
        df = pd.concat([train_df, valid_df], ignore_index=True)
        df = df[["text", "label"]].dropna()

        # Açık tehdit örnekleri — model bu kalıpları dataset'te yeterince görmüyor
        tehdit_ornekleri = pd.DataFrame({
            "text": [
                "seni öldüreceğim", "seni geberteceim", "canına okuyacağım",
                "hepinizi mahvedeceğim", "kafanı kıracağım", "sana zarar vereceğim",
                "evini yakacağım", "ailenle birlikte mahvedeceğim", "seni bitireceğim",
                "boğazını sıkacağım", "seni parçalayacağım", "kanını dökeceğim",
                "seni ezeceğim", "seni yok edeceğim", "kellenizi alacağım",
                "hepsini öldüreceğim", "seni bıçaklayacağım", "seni vuracağım",
                "seni bulup hesabını soracağım", "pişman edeceğim sizi",
                "bir daha görürsem canına okuyacağım", "bulursam öldürürüm",
                "senin gibi adamları ortadan kaldırırım", "sizi ezip geçeceğim",
                "seni mahvedeceğim hiç izi kalmayacak", "kafanı uçuracağım",
                "gözlerini oymak istiyorum", "sana işkence yapacağım",
                "ölmeden önce pişman olacaksın", "seni bir daha göremez hale getireceğim",
                # Ek: saldırgan sosyal medya dili
                "defol yoksa pişman ederim", "bok ye", "seni kırarım",
                "ağzını burnunu kırarım", "buradan git yoksa yanarın",
                "bir daha uğrarsan sonun olur", "seni silip süpüreceğim",
                "bu gece sonun", "gözünü açarsan ölürsün", "sesini kesmezsen patlarsın",
            ],
            "label": [1] * 40
        })
        df = pd.concat([df, tehdit_ornekleri], ignore_index=True)

        model = TespitModeli()
        model.egit(df)
        model.kaydet()

    print("\n Komutlar: 'mikrofon' | 'çıkış' | (doğrudan metin yaz)")

    while True:
        try:
            komut = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not komut:
            continue
        if komut.lower() == "çıkış":
            break
        elif komut.lower() == "mikrofon":
            metin = konusmayi_dinle()
            if metin:
                sonuc_yazdir(model.tahmin(metin))
        else:
            sonuc_yazdir(model.tahmin(komut))

    print("\nÇıkılıyor…")


if __name__ == "__main__":
    main()
