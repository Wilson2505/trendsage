import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

EXP = Path("exports"); EXP.mkdir(exist_ok=True)

def build_features_subset(raw_csv: Path, hours: float):
    from src.features import build_features
    from src.preprocess import add_sentiment
    df = pd.read_csv(raw_csv)
    if df.empty: return None
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.sort_values("created_at")
    t0 = df["created_at"].min()
    df = df[df["created_at"] <= t0 + pd.Timedelta(hours=hours)]
    df = add_sentiment(df)
    tag = raw_csv.stem.split("_")[0]
    return build_features(df, assumed_hashtag=tag)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--hours", nargs="+", type=float, default=[1,3,6])
    args = ap.parse_args()

    L = pd.read_csv(args.labels_csv)
    # build temporary feature table at multiple hours from all raw CSVs
    rows=[]
    for h in args.hours:
        feats=[]
        for rp in Path("data/raw").glob("*.csv"):
            f = build_features_subset(rp, h)
            if f is not None and not f.empty: feats.append(f)
        if not feats: continue
        F = pd.concat(feats, ignore_index=True)
        df = F.merge(L,on="hashtag",how="inner")
        X=df.drop(columns=["hashtag","t_start","t_end"]).select_dtypes("number")
        y=df["label"].astype(int).values
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
        m = RandomForestClassifier(n_estimators=300,random_state=42).fit(Xtr,ytr)
        yhat = m.predict(Xte)
        rows.append({"hours":h,"f1":f1_score(yte,yhat,zero_division=0)})
    out = pd.DataFrame(rows); out.to_csv(EXP/"sensitivity_hours_table.csv", index=False)
    plt.figure(figsize=(6,4)); plt.plot(out["hours"], out["f1"], marker="o")
    plt.xlabel("Look-back hours"); plt.ylabel("F1"); plt.title("Sensitivity to hours")
    plt.tight_layout(); plt.savefig(EXP/"sensitivity_hours.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
