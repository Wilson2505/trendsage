# TrendSage (Starter Platform)

A minimal end‑to‑end platform for collecting tweets by hashtag, engineering features, training a classifier, and visualising results in a Streamlit app.

> Best for quick screenshots/figures for your report and a working demo video.

## 0) Prerequisites
- Python 3.10+
- VS Code (with the **Python** extension)
- Twitter/X API access (bearer token at minimum).

## 1) Open in VS Code
```bash
# macOS / Linux
cd trendsage
code .

# Windows (PowerShell)
cd trendsage
code .
```

## 2) Create a virtual environment & install deps
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Configure your secrets
Copy `.env.example` to `.env` and fill in your keys:
```
cp .env.example .env
# or on Windows:
copy .env.example .env
```

Open `.env` and add your values (bearer token needed for recent search).

## 4) Collect tweets for a hashtag
```bash
# Example: collect last 6 hours of tweets for #F1 (max ~300 by API limits)
python -m src.twitter_client --hashtag "F1" --hours 6 --max-results 300
```
This writes a CSV into `data/raw/` (e.g., `F1_2025-08-19_06h.csv`).

## 5) Build features from the collected CSV
```bash
python -m src.features --input data/raw/F1_2025-08-19_06h.csv --output data/features/F1_features.csv
```

## 6) (Optional) Train a model (needs labels)
Create a CSV `data/labels.csv` with two columns: `hashtag,label` (label is 0/1 for not-trending/trending). Example:
```csv
hashtag,label
F1,1
CatsOfTwitter,0
```
Then run:
```bash
python -m src.model.train --features-glob "data/features/*_features.csv" --labels-csv data/labels.csv
```

## 7) Run the Streamlit app
```bash
streamlit run app_streamlit.py
```
- Enter a hashtag and hours, click **Fetch Tweets**, then **Compute Features**.  
- If a trained model exists in `src/model/models/random_forest.joblib`, you can **Predict Trend Likelihood**.

## 8) Exporting figures for your report
The app and CLI save PNGs in `exports/`. You can also right‑click images in Streamlit and save.

---

## Project Structure
```
trendsage/
├─ .env.example
├─ requirements.txt
├─ README.md
├─ exports/                # saved charts/images
├─ data/
│  ├─ raw/                 # collected raw tweets (CSV)
│  └─ features/            # per‑hashtag feature CSVs
├─ src/
│  ├─ __init__.py
│  ├─ config.py            # loads secrets from .env
│  ├─ twitter_client.py    # collects tweets via Tweepy (v2)
│  ├─ preprocess.py        # text cleaning + sentiment
│  ├─ features.py          # feature engineering (engagement, sentiment, network)
│  ├─ viz.py               # plotting utilities (matplotlib)
│  └─ model/
│     ├─ __init__.py
│     ├─ train.py          # trains LogisticRegression & RandomForest
│     ├─ predict.py        # uses saved model to predict
│     └─ models/           # saved .joblib models & metadata
└─ app_streamlit.py        # simple UI to run everything
```

## Notes
- API access and rate limits vary by account tier. If you hit limits, reduce `--max-results` or try another time window.
- This starter avoids complex DBs — CSVs keep things simple for a student project.
- Network features are approximated from mentions/retweets available in the search results.
- For time‑series forecasting (ARIMA/LSTM) you’d typically need longer historical windows; this starter focuses on early‑signal classification.
