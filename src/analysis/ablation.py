import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

GROUPS = {
    "engagement": ["avg_retweets","avg_likes","unique_users","avg_followers","verified_share"],
    "temporal":   ["tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets","acceleration"],
    "sentiment":  ["mean_sentiment","std_sentiment"],
    "network":    ["avg_degree_centrality","num_components"],
}
ALL = sum(GROUPS.values(), [])
EXP = Path("exports"); EXP.mkdir(exist_ok=True)

def run_subset(df, cols):
    X = df[cols].values; y = df["label"].astype(int).values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    m = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr,ytr)
    yhat = m.predict(Xte)
    return f1_score(yte,yhat,zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-glob", required=True); ap.add_argument("--labels-csv", required=True)
    args = ap.parse_args()
    feats = pd.concat([pd.read_csv(p) for p in Path().glob(args.features_glob)], ignore_index=True)
    labels = pd.read_csv(args.labels_csv)
    df = feats.merge(labels,on="hashtag",how="inner")

    rows=[]
    rows.append({"setup":"Full", "f1": run_subset(df, ALL)})
    for gname, gcols in GROUPS.items():
        kept = [c for c in ALL if c not in gcols]
        rows.append({"setup": f"Minus {gname}", "f1": run_subset(df, kept)})

    out = pd.DataFrame(rows); out.to_csv(EXP/"ablation_table.csv", index=False)
    plt.figure(figsize=(6,4)); plt.bar(out["setup"], out["f1"])
    plt.xticks(rotation=20, ha="right"); plt.ylabel("F1"); plt.title("Ablation Study")
    plt.tight_layout(); plt.savefig(EXP/"ablation_f1.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
