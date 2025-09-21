import os, glob, ast, json
from pathlib import Path
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Optional API imports (keep app responsive without them)
try:
    import tweepy
    HAS_TWEEPY = True
except Exception:
    HAS_TWEEPY = False

# Optional VADER for on-the-fly sentiment if your raw CSVs lack a sentiment column
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except Exception:
    HAS_VADER = False

# Optional network features
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

st.set_page_config(page_title="TrendSage", layout="wide")

RAW_DIR = Path("data/raw")
EXPORTS = Path("exports"); EXPORTS.mkdir(exist_ok=True)
MODEL_PATH = EXPORTS / "model.pkl"
MODEL_COLS = EXPORTS / "model_columns.json"

# -------------------------- Utilities --------------------------

def _to_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def list_local_tags() -> list[str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(RAW_DIR / "*.csv"))
    tags = sorted({Path(f).name.split("_")[0] for f in files})
    return tags

@st.cache_data(show_spinner=False)
def load_latest_raw(hashtag: str):
    files = sorted(glob.glob(str(RAW_DIR / f"{hashtag}_*.csv")), key=os.path.getmtime)
    if not files:
        return None, None
    path = files[-1]
    df = pd.read_csv(path)
    # Parse UTC timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    # Coerce numeric counters
    for c in ["like_count","retweet_count","reply_count","quote_count","followers_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    return df, path

def get_compound_series(df: pd.DataFrame) -> pd.Series:
    # Prefer precomputed sentiment column if present
    for cand in ("sentiment", "compound", "vader_compound"):
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce").dropna()
    # Otherwise compute on the fly with VADER (if installed)
    if not HAS_VADER:
        return pd.Series([], dtype=float)
    sid = SentimentIntensityAnalyzer()
    return df["text"].fillna("").map(lambda t: sid.polarity_scores(str(t))["compound"])

def plot_volume(df: pd.DataFrame):
    s = df.set_index("created_at").resample("15T")["tweet_id"].count()
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(s.index, s.values)
    ax.set_title("Tweet Volume Over Time (15-min bins)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Tweet Count")
    fig.savefig(EXPORTS / "ui_volume.png", dpi=200, bbox_inches="tight")
    st.pyplot(fig, clear_figure=True)

def plot_sentiment(df: pd.DataFrame):
    comp = get_compound_series(df)
    if comp.empty:
        st.info("No precomputed sentiment found and VADER not installed — skipping histogram.")
        return
    fig, ax = plt.subplots(figsize=(7,3))
    ax.hist(comp, bins=20)
    ax.set_title("Sentiment Distribution (compound)")
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Frequency")
    fig.savefig(EXPORTS / "ui_sentiment.png", dpi=200, bbox_inches="tight")
    st.pyplot(fig, clear_figure=True)

def base_feature_card(df: pd.DataFrame):
    """Show a small, human-readable subset of features."""
    if df.empty: return
    t_end = df["created_at"].max()
    c1h  = df[df["created_at"] >= t_end - timedelta(hours=1)].shape[0]
    c6h  = df[df["created_at"] >= t_end - timedelta(hours=6)].shape[0]
    c12h = df[df["created_at"] >= t_end - timedelta(hours=12)].shape[0]
    users = df["author_id"].nunique() if "author_id" in df.columns else None
    avg_rt = df["retweet_count"].mean() if "retweet_count" in df.columns else None
    avg_like = df["like_count"].mean() if "like_count" in df.columns else None

    feat = pd.DataFrame({
        "tweet_count_1h":[c1h],
        "tweet_count_6h":[c6h],
        "tweet_count_12h":[c12h],
        "total_tweets":[len(df)],
        "unique_users":[users],
        "avg_retweets":[avg_rt],
        "avg_likes":[avg_like],
    }).T.rename(columns={0:"value"})
    st.caption("Computed features (subset)")
    st.dataframe(feat, use_container_width=True, height=260)

def compute_feature_row_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return 1-row DataFrame aligned with training features."""
    if df.empty:
        return pd.DataFrame()

    t_end = df["created_at"].max()
    w1  = df[df["created_at"] >= t_end - timedelta(hours=1)]
    w6  = df[df["created_at"] >= t_end - timedelta(hours=6)]
    w12 = df[df["created_at"] >= t_end - timedelta(hours=12)]

    tweet_count_1h  = int(len(w1))
    tweet_count_6h  = int(len(w6))
    tweet_count_12h = int(len(w12))
    total_tweets    = int(len(df))
    acceleration    = (tweet_count_1h - tweet_count_6h) / 5.0

    unique_users = int(df["author_id"].nunique()) if "author_id" in df.columns else 0
    avg_retweets = float(df["retweet_count"].mean()) if "retweet_count" in df.columns else 0.0
    avg_likes    = float(df["like_count"].mean())    if "like_count"    in df.columns else 0.0
    avg_followers = float(df["followers_count"].mean()) if "followers_count" in df.columns else 0.0

    if "verified" in df.columns and unique_users > 0:
        verified_share = float(df.drop_duplicates("author_id")["verified"].astype(bool).mean())
    else:
        verified_share = 0.0

    comp = get_compound_series(df)
    mean_sentiment = float(comp.mean()) if not comp.empty else 0.0
    std_sentiment  = float(comp.std(ddof=0)) if not comp.empty else 0.0

    avg_degree_centrality, num_components = 0.0, 0
    if HAS_NX and ("mentions" in df.columns or "entities.mentions" in df.columns):
        col = "mentions" if "mentions" in df.columns else "entities.mentions"
        edges = []
        for _, row in df.iterrows():
            src = str(row.get("author_id", ""))
            for m in _to_list(row.get(col, [])):
                dst = str(m.get("username")) if isinstance(m, dict) else str(m)
                if src and dst:
                    edges.append((src, dst))
        if edges:
            G = nx.DiGraph(); G.add_edges_from(edges)
            if len(G) > 0:
                cent = nx.degree_centrality(G)
                avg_degree_centrality = float(sum(cent.values()) / len(cent))
                num_components = nx.number_weakly_connected_components(G)

    feat = {
        "tweet_count_1h": tweet_count_1h,
        "tweet_count_6h": tweet_count_6h,
        "tweet_count_12h": tweet_count_12h,
        "total_tweets": total_tweets,
        "avg_retweets": avg_retweets,
        "avg_likes": avg_likes,
        "unique_users": unique_users,
        "avg_followers": avg_followers,
        "verified_share": verified_share,
        "mean_sentiment": mean_sentiment,
        "std_sentiment": std_sentiment,
        "acceleration": acceleration,
        "avg_degree_centrality": avg_degree_centrality,
        "num_components": num_components,
    }
    return pd.DataFrame([feat])

@st.cache_resource(show_spinner=False)
def load_model():
    if not (MODEL_PATH.exists() and MODEL_COLS.exists()):
        return None, None
    model = joblib.load(MODEL_PATH)
    cols = json.loads(MODEL_COLS.read_text())
    return model, cols

def predict_probability(feat_row: pd.DataFrame):
    model, cols = load_model()
    if model is None:
        return None, None
    X = feat_row.reindex(columns=cols, fill_value=0.0)
    proba = float(model.predict_proba(X)[:,1][0])
    return proba, cols

# ---------------------------- UI ----------------------------

st.title("TrendSage — Early Trend Predictor (Starter)")

with st.sidebar:
    st.header("Fetch Tweets")
    source = st.radio("Data source", ["Local CSV (offline)", "Live API (may be rate-limited)"], index=0)
    tags = list_local_tags()
    default_tag = "F1" if "F1" in tags else (tags[0] if tags else "")
    hashtag = st.text_input("Hashtag (without #)", value=default_tag)
    hours = st.number_input("Lookback hours", min_value=1, max_value=12, value=2, step=1)
    max_results = st.slider("Max tweets to fetch", min_value=50, max_value=500, value=100, step=50)
    go = st.button("Fetch Tweets")

if go:
    if source.startswith("Local"):
        with st.spinner("Loading latest local CSV…"):
            df, path = load_latest_raw(hashtag)
        if df is None:
            st.warning(f"No local CSV found for #{hashtag} in {RAW_DIR}/")
        else:
            st.success(f"Loaded {len(df)} tweets from {Path(path).name}")
            plot_volume(df)
            plot_sentiment(df)
            base_feature_card(df)

            # Full model feature vector + probability
            feat_row = compute_feature_row_for_model(df)
            with st.expander("Computed features (full vector passed to model)"):
                st.dataframe(feat_row.T.rename(columns={0:"value"}), use_container_width=True, height=320)

            proba, cols = predict_probability(feat_row)
            if proba is None:
                st.warning("Model not found. Train & save model to exports/model.pkl to enable probability & threshold.")
            else:
                st.subheader("Trend probability")
                st.metric("Estimated probability", f"{proba:.2%}")
                thresh = st.slider("Decision threshold τ", min_value=0.00, max_value=1.00, value=0.50, step=0.01)
                label = "TREND" if proba >= thresh else "Not trend"
                st.success(f"Decision at τ={thresh:.2f}: **{label}**")
                st.caption(f"Model columns: {', '.join(cols)}")

    else:
        # Keep UI responsive; do not block for 900s on 429
        st.info("Live API disabled here to avoid long rate-limit sleeps. Switch to 'Local CSV (offline)'.")
else:
    st.info("Use the sidebar to load a local CSV (recommended) or try Live API when limits allow.")
