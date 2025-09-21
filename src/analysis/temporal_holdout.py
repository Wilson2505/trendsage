import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

FEATS = [
    "tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
    "avg_retweets","avg_likes","unique_users","avg_followers",
    "verified_share","mean_sentiment","std_sentiment","acceleration",
    "avg_degree_centrality","num_components",
]
EXP = Path("exports"); EXP.mkdir(exist_ok=True)

def compute_metrics(y_true, y_pred, y_prob):
    has_both = len(set(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }

def safe_fit(model, X, y):
    if len(np.unique(y)) < 2:
        return None, "train_only_one_class"
    try:
        model.fit(X, y)
        return model, None
    except Exception as e:
        return None, f"fit_error: {type(e).__name__}"

def pick_split_time(df):
    """
    Try to pick a split where TRAIN and TEST both contain both classes.
    Fall back to: TRAIN has both classes; TEST may be single-class.
    If still impossible, use a safe median/penultimate time.
    """
    df = df.sort_values("t_end").reset_index(drop=True)
    n = len(df)

    # 1) ideal: both sides have both classes
    for i in range(2, n-1):
        tr, te = df.iloc[:i], df.iloc[i:]
        if tr["label"].nunique() >= 2 and te["label"].nunique() >= 2:
            return tr["t_end"].max(), "both_sides_ok"

    # 2) relaxed: train has both classes; test may be single-class
    for i in range(2, n-1):
        tr, te = df.iloc[:i], df.iloc[i:]
        if tr["label"].nunique() >= 2 and len(te) >= 1:
            return tr["t_end"].max(), "train_ok"

    # 3) safest fallback: median t_end; if that makes one side empty,
    #    use the penultimate timestamp so test has at least one row.
    t_sorted = df["t_end"].sort_values().to_list()
    split = t_sorted[len(t_sorted)//2]
    if (df["t_end"] <= split).sum() < 2 or (df["t_end"] > split).sum() < 1:
        if len(t_sorted) >= 2:
            split = t_sorted[-2]  # before the last time
    return split, "fallback"

def main():
    ap = argparse.ArgumentParser(description="Temporal holdout with robust splitting")
    ap.add_argument("--features-glob", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--split-time", help="ISO timestamp; earlier=train, later=test (overrides auto)")
    args = ap.parse_args()

    # Load and merge
    feats = []
    for p in Path().glob(args.features_glob):
        df = pd.read_csv(p)
        feats.append(df)
    if not feats:
        raise SystemExit("No feature files found. Run src.batch_features first.")
    F = pd.concat(feats, ignore_index=True)

    # Ensure t_end is datetime and not null
    F["t_end"] = pd.to_datetime(F["t_end"], utc=True, errors="coerce")
    F = F.dropna(subset=["t_end"])

    L = pd.read_csv(args.labels_csv)
    df = F.merge(L, on="hashtag", how="inner")
    if df["label"].nunique() < 2:
        raise SystemExit("labels.csv has only one class. Add at least one 0 and one 1.")

    # Choose split time
    if args.split_time:
        split_time = pd.to_datetime(args.split_time, utc=True, errors="coerce")
        if pd.isna(split_time):
            raise SystemExit("--split-time could not be parsed.")
        mode = "manual"
    else:
        split_time, mode = pick_split_time(df)

    # Split
    train = df[df["t_end"] <= split_time].copy()
    test  = df[df["t_end"] >  split_time].copy()
    if len(train) < 2 or len(test) < 1:
        raise SystemExit(
            f"Temporal split too small (train={len(train)}, test={len(test)}). "
            "Label more hashtags or pass a different --split-time."
        )

    Xtr, ytr = train[FEATS].values, train["label"].astype(int).values
    Xte, yte = test[FEATS].values,  test["label"].astype(int).values

    models = {
        "logreg": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results = {}
    chosen = None; best_f1 = -1.0

    for name, mdl in models.items():
        fitted, err = safe_fit(mdl, Xtr, ytr)
        if fitted is None:
            print(f"[skip] {name}: {err}")
            continue
        prob = fitted.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        met = compute_metrics(yte, pred, prob)
        results[name] = met
        if met["f1"] > best_f1:
            best_f1, chosen = met["f1"], (name, fitted, prob, pred)

    # Save metrics
    out = pd.DataFrame(results).T
    out.to_csv(EXP / "temporal_holdout_metrics.csv")

    # Save split info for appendix
    (EXP / "temporal_holdout_split.txt").write_text(
        f"mode={mode}\nsplit_time={pd.to_datetime(split_time)}\n"
        f"train_n={len(train)}, test_n={len(test)}\n"
        f"train_counts=\n{train['label'].value_counts().to_string()}\n\n"
        f"test_counts=\n{test['label'].value_counts().to_string()}\n"
    )

    if chosen is None:
        print("[error] No model could be trained (train single-class). Label more data.")
        return

    name, model, prob, pred = chosen

    # ROC/PR if test has both classes
    if len(set(yte)) > 1:
        fpr, tpr, _ = roc_curve(yte, prob)
        plt.figure(figsize=(5,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Temporal Holdout ROC"); plt.tight_layout()
        plt.savefig(EXP / "temporal_holdout_roc.png", dpi=150); plt.close()

        prec, rec, _ = precision_recall_curve(yte, prob)
        plt.figure(figsize=(5,4)); plt.plot(rec,prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Temporal Holdout PR"); plt.tight_layout()
        plt.savefig(EXP / "temporal_holdout_pr.png", dpi=150); plt.close()
    else:
        print("[warn] Test set single-class; ROC/PR undefined (ROC_AUC=NaN). Metrics file still written.")

if __name__ == "__main__":
    main()
