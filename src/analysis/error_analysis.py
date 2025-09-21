import argparse
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    Xtr,Xte,ytr,yte, idx_tr, idx_te = train_test_split(
        X,y,df.index, test_size=0.3, stratify=y, random_state=42
    )
    m = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr,ytr)
    prob = m.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)

    out = df.loc[idx_te, ["hashtag"]+FEATS].copy()
    out["true"]=yte; out["pred"]=pred; out["prob"]=prob
    out["error_type"] = np.where((out["true"]==1)&(out["pred"]==0),"FN",
                         np.where((out["true"]==0)&(out["pred"]==1),"FP","OK"))
    out.sort_values(["error_type","prob"], ascending=[True,False]).to_csv(EXP/"misclassifications.csv", index=False)

if __name__ == "__main__":
    main()
