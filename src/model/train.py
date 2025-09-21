# import argparse, glob, json
# from pathlib import Path
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
# )
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# FEATURE_COLUMNS = [
#     "tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
#     "avg_retweets","avg_likes","unique_users","avg_followers",
#     "verified_share","mean_sentiment","std_sentiment","acceleration",
#     "avg_degree_centrality","num_components",
# ]

# EXP_DIR = Path("exports")
# MODELS_DIR = Path(__file__).parent / "models"
# EXP_DIR.mkdir(parents=True, exist_ok=True)
# MODELS_DIR.mkdir(parents=True, exist_ok=True)

# def load_features(features_glob: str) -> pd.DataFrame:
#     rows = []
#     for path in glob.glob(features_glob):
#         df = pd.read_csv(path)
#         rows.append(df)
#     if not rows:
#         raise FileNotFoundError(f"No feature files matched {features_glob}")
#     return pd.concat(rows, ignore_index=True)

# def compute_metrics(y_true, y_pred, y_prob):
#     return {
#         "accuracy": float(accuracy_score(y_true, y_pred)),
#         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
#         "recall": float(recall_score(y_true, y_pred, zero_division=0)),
#         "f1": float(f1_score(y_true, y_pred, zero_division=0)),
#         "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true))>1 else float("nan"),
#     }

# def plot_roc(y_true, y_prob, out_png):
#     fpr, tpr, _ = roc_curve(y_true, y_prob)
#     plt.figure(figsize=(5,4))
#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot([0,1],[0,1], linestyle="--")
#     plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.tight_layout()
#     plt.savefig(out_png, dpi=150); plt.close()

# def plot_pr(y_true, y_prob, out_png):
#     prec, rec, _ = precision_recall_curve(y_true, y_prob)
#     plt.figure(figsize=(5,4))
#     plt.plot(rec, prec, label="PR")
#     plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve"); plt.tight_layout()
#     plt.savefig(out_png, dpi=150); plt.close()

# def plot_confmat(y_true, y_pred, out_png):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(4,4))
#     plt.imshow(cm, interpolation="nearest")
#     plt.title("Confusion Matrix"); plt.colorbar()
#     plt.xticks([0,1], ["Not Trend","Trend"]); plt.yticks([0,1], ["Not Trend","Trend"])
#     for (i,j), v in np.ndenumerate(cm):
#         plt.text(j, i, str(v), ha="center", va="center")
#     plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# def plot_feature_importance(model, cols, out_png):
#     if not hasattr(model, "feature_importances_"):
#         return
#     importances = model.feature_importances_
#     order = np.argsort(importances)[::-1]
#     plt.figure(figsize=(7,4))
#     plt.bar(range(len(cols)), importances[order])
#     plt.xticks(range(len(cols)), [cols[i] for i in order], rotation=45, ha="right")
#     plt.title("Random Forest Feature Importance"); plt.tight_layout()
#     plt.savefig(out_png, dpi=150); plt.close()

# def main():
#     parser = argparse.ArgumentParser(description="Train TrendSage models and export plots")
#     parser.add_argument("--features-glob", required=True, help='e.g., "data/features/*_features.csv"')
#     parser.add_argument("--labels-csv", required=True, help="CSV with columns: hashtag,label")
#     args = parser.parse_args()

#     feats = load_features(args.features_glob)
#     labels = pd.read_csv(args.labels_csv)
#     df = feats.merge(labels, on="hashtag", how="inner")

#     X = df[FEATURE_COLUMNS].values
#     y = df["label"].astype(int).values

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=42
#     )

#     # Logistic Regression
#     logreg = LogisticRegression(max_iter=200)
#     logreg.fit(X_train, y_train)
#     lr_probs = logreg.predict_proba(X_test)[:,1]
#     lr_pred  = (lr_probs >= 0.5).astype(int)
#     lr_metrics = compute_metrics(y_test, lr_pred, lr_probs)

#     # Random Forest
#     rf = RandomForestClassifier(n_estimators=300, random_state=42)
#     rf.fit(X_train, y_train)
#     rf_probs = rf.predict_proba(X_test)[:,1]
#     rf_pred  = (rf_probs >= 0.5).astype(int)
#     rf_metrics = compute_metrics(y_test, rf_pred, rf_probs)

#     # Save metrics table
#     metrics_df = pd.DataFrame([
#         {"model":"logreg", **lr_metrics},
#         {"model":"random_forest", **rf_metrics},
#     ])
#     metrics_csv = EXP_DIR / "model_metrics.csv"
#     metrics_df.to_csv(metrics_csv, index=False)

#     # Choose best by F1 and save
#     best_name, best_model, best_probs, best_pred = (
#         ("random_forest", rf, rf_probs, rf_pred)
#         if rf_metrics["f1"] >= lr_metrics["f1"]
#         else ("logreg", logreg, lr_probs, lr_pred)
#     )
#     joblib.dump(best_model, MODELS_DIR / f"{best_name}.joblib")
#     with open(MODELS_DIR / "metrics.json", "w") as f:
#         json.dump({"logreg": lr_metrics, "random_forest": rf_metrics, "chosen": best_name}, f, indent=2)
#     with open(MODELS_DIR / "feature_columns.json", "w") as f:
#         json.dump(FEATURE_COLUMNS, f, indent=2)

#     # Plots for the chosen model
#     plot_roc(y_test, best_probs, EXP_DIR / "model_roc.png")
#     plot_pr(y_test,  best_probs, EXP_DIR / "model_pr.png")
#     plot_confmat(y_test, best_pred, EXP_DIR / "model_confusion.png")
#     if best_name == "random_forest":
#         plot_feature_importance(best_model, FEATURE_COLUMNS, EXP_DIR / "model_feature_importance.png")

#     print("Saved:")
#     print(f" - model -> {MODELS_DIR / (best_name + '.joblib')}")
#     print(f" - metrics table -> {metrics_csv}")
#     print(" - ROC -> exports/model_roc.png")
#     print(" - PR  -> exports/model_pr.png")
#     print(" - Confusion -> exports/model_confusion.png")
#     if best_name == "random_forest":
#         print(" - Feature importance -> exports/model_feature_importance.png")

# if __name__ == "__main__":
#     main()

import argparse, glob, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------- Config ----------------------------

FEATURE_COLUMNS = [
    "tweet_count_1h","tweet_count_6h","tweet_count_12h","total_tweets",
    "avg_retweets","avg_likes","unique_users","avg_followers",
    "verified_share","mean_sentiment","std_sentiment","acceleration",
    "avg_degree_centrality","num_components",
]

EXP_DIR = Path("exports")
MODELS_DIR = Path(__file__).parent / "models"
EXP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------- IO ----------------------------

def load_features(features_glob: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(features_glob):
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No feature files matched {features_glob}")
    return pd.concat(rows, ignore_index=True)

# ---------------------------- Metrics & Plots ----------------------------

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
    }

def plot_roc(y_true, y_prob, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_pr(y_true, y_prob, out_png):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve"); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_confmat(y_true, y_pred, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks([0, 1], ["Not Trend", "Trend"]); plt.yticks([0, 1], ["Not Trend", "Trend"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_feature_importance(model, cols, out_png):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(cols)), importances[order])
    plt.xticks(range(len(cols)), [cols[i] for i in order], rotation=45, ha="right")
    plt.title("Random Forest Feature Importance"); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TrendSage models and export plots")
    parser.add_argument("--features-glob", required=True, help='e.g., "data/features/*_features.csv"')
    parser.add_argument("--labels-csv", required=True, help="CSV with columns: hashtag,label")
    args = parser.parse_args()

    # Load & join
    feats = load_features(args.features_glob)
    labels = pd.read_csv(args.labels_csv)
    df = feats.merge(labels, on="hashtag", how="inner")

    # Train/test
    X = df[FEATURE_COLUMNS].values
    y = df["label"].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # ---------------- Logistic Regression ----------------
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    lr_probs = logreg.predict_proba(X_test)[:, 1]
    lr_pred  = (lr_probs >= 0.5).astype(int)
    lr_metrics = compute_metrics(y_test, lr_pred, lr_probs)

    # ---------------- Random Forest ----------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_pred  = (rf_probs >= 0.5).astype(int)
    rf_metrics = compute_metrics(y_test, rf_pred, rf_probs)

    # Save metrics table
    metrics_df = pd.DataFrame([
        {"model": "logreg",         **lr_metrics},
        {"model": "random_forest",  **rf_metrics},
    ])
    metrics_csv = EXP_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Choose best by F1
    best_name, best_model, best_probs, best_pred = (
        ("random_forest", rf, rf_probs, rf_pred)
        if rf_metrics["f1"] >= lr_metrics["f1"]
        else ("logreg", logreg, lr_probs, lr_pred)
    )

    # ---------------- Persist models & metadata ----------------
    # 1) Keep your original artifacts under src/model/models/
    joblib.dump(best_model, MODELS_DIR / f"{best_name}.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"logreg": lr_metrics, "random_forest": rf_metrics, "chosen": best_name}, f, indent=2)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # 2) Save UI-ready copies under exports/ for the Streamlit app
    joblib.dump(best_model, EXP_DIR / "model.pkl")
    with open(EXP_DIR / "model_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    # ----------------------------------------------------------

    # Plots for the chosen model
    plot_roc(y_test, best_probs, EXP_DIR / "model_roc.png")
    plot_pr(y_test,  best_probs, EXP_DIR / "model_pr.png")
    plot_confmat(y_test, best_pred, EXP_DIR / "model_confusion.png")
    if best_name == "random_forest":
        plot_feature_importance(best_model, FEATURE_COLUMNS, EXP_DIR / "model_feature_importance.png")

    # Log outputs
    print("Saved:")
    print(f" - chosen model -> {MODELS_DIR / (best_name + '.joblib')}")
    print(f" - metrics table -> {metrics_csv}")
    print(f" - UI model      -> {EXP_DIR / 'model.pkl'}")
    print(f" - UI columns    -> {EXP_DIR / 'model_columns.json'}")
    print(" - ROC           -> exports/model_roc.png")
    print(" - PR            -> exports/model_pr.png")
    print(" - Confusion     -> exports/model_confusion.png")
    if best_name == "random_forest":
        print(" - Feature importance -> exports/model_feature_importance.png")

if __name__ == "__main__":
    main()
