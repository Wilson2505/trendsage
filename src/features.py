import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from .preprocess import add_sentiment
from .viz import save_volume_plot, save_sentiment_hist

def _safe_dt(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def build_features(df: pd.DataFrame, assumed_hashtag: str) -> pd.DataFrame:
    """Aggregate tweet-level data into a single-row feature vector for a hashtag."""
    if df.empty:
        return pd.DataFrame()

    df = add_sentiment(df)
    df["created_at"] = _safe_dt(df["created_at"])
    df = df.sort_values("created_at")

    # Basic engagement aggregates
    total_tweets = len(df)
    avg_retweets = df["retweet_count"].mean()
    avg_likes = df["like_count"].mean()
    unique_users = df["author_id"].nunique()
    avg_followers = df["followers_count"].replace(0, np.nan).mean() if "followers_count" in df else np.nan
    verified_share = df["verified"].mean() if "verified" in df else 0.0
    mean_sent = df["sentiment"].mean()
    std_sent = df["sentiment"].std(ddof=0)

    # Time windows — compute counts in 1h, 6h, 12h relative to the first tweet
    if total_tweets:
        t0 = df["created_at"].min()
        df["mins_from_start"] = (df["created_at"] - t0).dt.total_seconds() / 60.0
        count_1h = (df["mins_from_start"] <= 60).sum()
        count_6h = (df["mins_from_start"] <= 360).sum()
        count_12h = (df["mins_from_start"] <= 720).sum()
        # simple acceleration: last hour vs first hour (if span long enough)
        df["hour_bin"] = ((df["mins_from_start"]) // 60).astype(int)
        hour_counts = df.groupby("hour_bin")["tweet_id"].count().sort_index()
        accel = 0.0
        if len(hour_counts) >= 2:
            accel = hour_counts.iloc[-1] - hour_counts.iloc[0]
    else:
        count_1h = count_6h = count_12h = 0
        accel = 0.0

    # Network (approximate): build mention graph
    edges = []
    if "mentions" in df.columns:
        for _, row in df.iterrows():
            author = row.get("username") or str(row.get("author_id"))
            for m in (row.get("mentions") or []):
                if author and m:
                    edges.append((author, m))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    deg_cent = nx.degree_centrality(G) if len(G) else {}

    # Simple network stats
    avg_degree_centrality = float(np.mean(list(deg_cent.values()))) if deg_cent else 0.0
    num_components = nx.number_weakly_connected_components(G) if len(G) else 0

    t_start = df["created_at"].min()
    t_end   = df["created_at"].max()
    duration_hours = float((t_end - t_start).total_seconds() / 3600.0)

    features = {
        "hashtag": assumed_hashtag,
        "t_start": t_start,
        "t_end": t_end,
        "duration_hours": duration_hours,
        "tweet_count_1h": int(count_1h),
        "tweet_count_6h": int(count_6h),
        "tweet_count_12h": int(count_12h),
        "total_tweets": int(total_tweets),
        "avg_retweets": float(avg_retweets) if not np.isnan(avg_retweets) else 0.0,
        "avg_likes": float(avg_likes) if not np.isnan(avg_likes) else 0.0,
        "unique_users": int(unique_users),
        "avg_followers": float(avg_followers) if avg_followers==avg_followers else 0.0,
        "verified_share": float(verified_share),
        "mean_sentiment": float(mean_sent) if mean_sent==mean_sent else 0.0,
        "std_sentiment": float(std_sent) if std_sent==std_sent else 0.0,
        "acceleration": float(accel),
        "avg_degree_centrality": float(avg_degree_centrality),
        "num_components": int(num_components),
    }

    return pd.DataFrame([features])

def main():
    parser = argparse.ArgumentParser(description="Build features from a raw tweets CSV")
    parser.add_argument("--input", required=True, help="Path to raw tweets CSV from data/raw/")
    parser.add_argument("--output", required=True, help="Path to write features CSV, e.g., data/features/XYZ_features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Derive hashtag from filename if possible
    assumed_hashtag = Path(args.input).stem.split("_")[0]

    feats = build_features(df, assumed_hashtag=assumed_hashtag)
    feats.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")

    # Also save quick charts for your report
    save_volume_plot(df, out_path=f"exports/{assumed_hashtag}_volume.png")
    save_sentiment_hist(df, out_path=f"exports/{assumed_hashtag}_sentiment.png")
    print("Saved figures into exports/")

if __name__ == "__main__":
    main()
