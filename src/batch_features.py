import glob
from pathlib import Path
import pandas as pd
from .features import build_features
from .preprocess import add_sentiment
from .viz import save_volume_plot, save_sentiment_hist

RAW_DIR = Path("data/raw")
FEAT_DIR = Path("data/features")
EXP_DIR = Path("exports")
FEAT_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

def process_one(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    # add sentiment for plotting
    df = add_sentiment(df)
    tag = csv_path.stem.split("_")[0]
    # figures
    save_volume_plot(df, out_path=str(EXP_DIR / f"{tag}_volume.png"))
    save_sentiment_hist(df, out_path=str(EXP_DIR / f"{tag}_sentiment.png"))
    # features
    feats = build_features(df, assumed_hashtag=tag)
    out_csv = FEAT_DIR / f"{tag}_features.csv"
    feats.to_csv(out_csv, index=False)
    print(f"[ok] {tag}: features -> {out_csv}")
    return out_csv

def main():
    paths = sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        print("No raw CSVs in data/raw/. Run src.twitter_client first.")
        return
    for p in paths:
        process_one(p)

if __name__ == "__main__":
    main()
