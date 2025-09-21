import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = [
    "tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
    "avg_retweets","avg_likes","unique_users","avg_followers",
    "verified_share","mean_sentiment","std_sentiment","acceleration",
    "avg_degree_centrality","num_components"
]
EXP = Path("exports"); EXP.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-glob", required=True)
    ap.add_argument("--labels-csv", required=True)
    args = ap.parse_args()

    feats = pd.concat([pd.read_csv(p) for p in Path().glob(args.features_glob)], ignore_index=True)
    labels = pd.read_csv(args.labels_csv)
    df = feats.merge(labels, on="hashtag", how="inner")

    y = df["label"].astype(int).values

    # Majority baseline
    majority = int(pd.Series(y).mode()[0])
    yhat_major = np.full_like(y, majority)
    base = dict(model="majority",
                accuracy=accuracy_score(y,yhat_major),
                precision=precision_score(y,yhat_major,zero_division=0),
                recall=recall_score(y,yhat_major,zero_division=0),
                f1=f1_score(y,yhat_major,zero_division=0))

    # Simple volume-threshold on tweet_count_1h
    xs, f1s = [], []
    best = (0.5, -1)
    x = df["tweet_count_1h"].values
    for k in sorted(set(x)):
        yhat = (x >= k).astype(int)
        f1v = f1_score(y,yhat,zero_division=0); f1s.append(f1v); xs.append(k)
        if f1v > best[1]: best = (k, f1v)
    plt.figure(figsize=(6,4))
    plt.plot(xs, f1s)
    plt.xlabel("tweet_count_1h threshold k"); plt.ylabel("F1")
    plt.title("Threshold Sweep (volume baseline)"); plt.tight_layout()
    plt.savefig(EXP / "threshold_sweep.png", dpi=150); plt.close()

    k, f1v = best
    yhat_vol = (x >= k).astype(int)
    vol = dict(model=f"volume>= {k}",
               accuracy=accuracy_score(y,yhat_vol),
               precision=precision_score(y,yhat_vol,zero_division=0),
               recall=recall_score(y,yhat_vol,zero_division=0),
               f1=f1_score(y,yhat_vol,zero_division=0))

    out = pd.DataFrame([base, vol])
    out.to_csv(EXP / "baselines.csv", index=False)

    # Compare to  trained model if metrics exist
    mfile = Path("src/model/models/metrics.json")
    if mfile.exists():
        info = json.load(open(mfile))
        rf = info["random_forest"]; lr = info["logreg"]
        compare = pd.DataFrame([base, vol,
            {"model":"logreg", **lr},
            {"model":"random_forest", **rf}])
        cmp_csv = EXP / "baseline_vs_models.csv"; compare.to_csv(cmp_csv, index=False)
        # simple bar for F1
        plt.figure(figsize=(6,4))
        plt.bar(compare["model"], compare["f1"])
        plt.ylabel("F1"); plt.title("Baselines vs Models")
        plt.tight_layout(); plt.savefig(EXP / "baseline_vs_models.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()