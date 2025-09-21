import matplotlib.pyplot as plt
import pandas as pd

def save_volume_plot(df: pd.DataFrame, out_path: str):
    if df.empty:
        return
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at")
    counts = df.set_index("created_at").resample("15T")["tweet_id"].count()
    plt.figure(figsize=(8,4))
    counts.plot()
    plt.title("Tweet Volume Over Time (15-min bins)")
    plt.xlabel("Time")
    plt.ylabel("Tweet Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_sentiment_hist(df: pd.DataFrame, out_path: str):
    if df.empty or "sentiment" not in df.columns:
        return
    plt.figure(figsize=(6,4))
    df["sentiment"].plot(kind="hist", bins=30)
    plt.title("Sentiment Distribution (compound)")
    plt.xlabel("VADER compound score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
