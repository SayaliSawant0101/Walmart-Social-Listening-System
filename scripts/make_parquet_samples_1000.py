from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
N = 1000
RANDOM_STATE = 42

FILES = [
    "tweets_stage0_raw.parquet",
    "tweets_stage0_raw_comp.parquet",
    "tweets_stage1_sentiment.parquet",
    "tweets_stage1_sentiment_comp.parquet",
    "tweets_stage2_aspects.parquet",
    "tweets_stage2_aspects_comp.parquet",
    "tweets_stage3_aspect_sentiment.parquet",
    "tweets_stage3_aspect_sentiment_comp.parquet",
    "tweets_stage3_themes.parquet",
]

for name in FILES:
    p = DATA_DIR / name
    if not p.exists():
        print(f"SKIP (missing): {p}")
        continue

    df = pd.read_parquet(p)
    before = len(df)

    if before > N:
        df_small = df.sample(n=N, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_small = df.copy()

    df_small.to_parquet(p, index=False)
    print(f"WROTE {name}: {len(df_small)} rows (was {before})")
