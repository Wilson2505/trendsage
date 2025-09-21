import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score

FEATS = ["tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
         "avg_retweets","avg_likes","unique_users","avg_followers",
         "verified_share","mean_sentiment","std_sentiment","acceleration",
         "avg_degree_centrality","num_components"]
EXP = Path("exports"); EXP.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-glob", required=True); ap.add_argument("--labels-csv", required=True)
    args = ap.parse_args()
    feats = pd.concat([pd.read_csv(p) for p in Path().glob(args.features_glob)], ignore_index=True)
    labels = pd.read_csv(args.labels_csv)
    df = feats.merge(labels,on="hashtag",how="inner")

    X = df[FEATS].values; y = df["label"].astype(int).values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    m = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr,ytr)
    prob = m.predict_proba(Xte)[:,1]

    # calibration curve
    fracs, mean_pred = calibration_curve(yte, prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5,4)); plt.plot(mean_pred, fracs, marker="o")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("Predicted"); plt.ylabel("Observed")
    plt.title("Calibration Curve"); plt.tight_layout()
    plt.savefig(EXP/"calibration_curve.png", dpi=150); plt.close()

    with open(EXP/"brier_score.txt","w") as f:
        f.write(f"Brier score: {brier_score_loss(yte, prob):.4f}\n")

    # F1 vs threshold
    ths = np.linspace(0.05,0.95,19); f1s=[]
    for t in ths:
        f1s.append(f1_score(yte,(prob>=t).astype(int),zero_division=0))
    plt.figure(figsize=(6,4)); plt.plot(ths,f1s,marker="o")
    plt.xlabel("Threshold"); plt.ylabel("F1"); plt.title("F1 vs Threshold")
    plt.tight_layout(); plt.savefig(EXP/"f1_vs_threshold.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
