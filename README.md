# Attack Intent Detection System

A Turkish-language NLP system that detects aggressive/threatening speech in text and voice input using TF-IDF N-gram features combined with a Markov Chain language model.

## Demo

![Streamlit UI](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

## How It Works

The system combines two models for final prediction:

1. **TF-IDF + Logistic Regression** — extracts unigram, bigram, and trigram features from cleaned text, trained on 42K Turkish tweets
2. **Bigram Markov Chain** — learns transition probabilities separately from aggressive and normal corpora, then computes a delta score indicating which distribution the input text resembles more

Final probability:
```
p_aggressive_final = p_lr + 0.15 × sigmoid(markov_delta)
```

## Dataset

[OffComTR-style](https://archive-12/) Turkish social media dataset with binary labels:
- `0` — Normal
- `1` — Aggressive / Threatening

| Split | Samples |
|-------|---------|
| Train | 42,398  |
| Valid | ~5,000  |
| Test  | ~5,000  |

## Results

```
              precision    recall  f1-score
Normal            0.86      0.96      0.91
Aggressive        0.95      0.81      0.88
accuracy                              0.90
```

## Installation

```bash
git clone https://github.com/BerkeTozkoparam/saldiri-niyeti-tespiti.git
cd saldiri-niyeti-tespiti
pip install -r requirements.txt
```

> **macOS:** PyAudio requires PortAudio. Install it first:
> ```bash
> brew install portaudio
> ```

## Usage

### Streamlit App (recommended)

```bash
streamlit run app.py
```

Features:
- **Voice input** — record audio directly in the browser, transcribed via Google Speech API (Turkish)
- **Text input** — type or paste any text
- **Batch analysis** — analyze multiple lines at once, results shown as a table
- **Probability bars** + Markov Δ score for each prediction

### CLI

```bash
python tespit.py
```

Type text directly or use the `mikrofon` command for microphone input.

## Project Structure

```
├── app.py           # Streamlit web interface
├── tespit.py        # Core model: preprocessing, Markov, TF-IDF pipeline
├── archive-12/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── requirements.txt
```

The trained model (`niyet_model.pkl`) is not committed — it is generated automatically on first run.
