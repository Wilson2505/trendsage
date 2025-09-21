import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW = Path("data/raw")
FEAT = Path("data/features")
EXP  = Path("exports"); EXP.mkdir(exist_ok=True)

FEATURE_COLS = [
    "tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
    "avg_retweets","avg_likes","unique_users","avg_followers",
    "verified_share","mean_sentiment","std_sentiment","acceleration",
    "avg_degree_centrality","num_components"
]

def main():
    # --- dataset summary from raw ---
    rows = []
    for rp in sorted(RAW.glob("*.csv")):
        tag = rp.stem.split("_")[0]
        df = pd.read_csv(rp)
        if df.empty: continue
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        rows.append({
            "hashtag": tag,
            "file": rp.name,
            "tweets": len(df),
            "unique_users": df["author_id"].nunique(),
            "avg_likes": df["like_count"].mean(),
            "avg_retweets": df["retweet_count"].mean(),
            "verified_share": df["verified"].mean() if "verified" in df else np.nan,
            "t_start": df["created_at"].min(),
            "t_end": df["created_at"].max(),
        })
    raw_summary = pd.DataFrame(rows).drop_duplicates(subset=["hashtag"], keep="last")
    raw_summary.to_csv(EXP / "dataset_summary.csv", index=False)

    # --- feature correlations (across hashtags) ---
    feats = []
    for fp in FEAT.glob("*_features.csv"):
        feats.append(pd.read_csv(fp, parse_dates=["t_start","t_end"]))
    if feats:
        F = pd.concat(feats, ignore_index=True)
        corr = F[FEATURE_COLS].corr()
        plt.figure(figsize=(8,6))
        plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(FEATURE_COLS)), FEATURE_COLS, rotation=45, ha="right")
        plt.yticks(range(len(FEATURE_COLS)), FEATURE_COLS)
        plt.colorbar()
        plt.title("Feature Correlations")
        plt.tight_layout(); plt.savefig(EXP / "feature_correlations.png", dpi=150); plt.close()

        plt.figure(figsize=(6,4))
        F["total_tweets"].plot(kind="hist", bins=20)
        plt.title("Histogram of Tweet Counts per Hashtag")
        plt.xlabel("total_tweets"); plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(EXP / "tweetcount_hist.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
