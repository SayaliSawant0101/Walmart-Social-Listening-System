import os
from pathlib import Path
import boto3

# ---- Config via env vars ----
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "data").strip().strip("/")
LOCAL_DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", "data"))

# Only download these files (keeps it deterministic)
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
    "theme_names.json",
    "theme_summaries.json",
]

def main():
    if not S3_BUCKET:
        print("S3_BUCKET not set; skipping S3 download (using local data/).")
        return

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    for fname in FILES:
        key = f"{S3_PREFIX}/{fname}"
        dest = LOCAL_DATA_DIR / fname
        print(f"Downloading s3://{S3_BUCKET}/{key} -> {dest}")
        s3.download_file(S3_BUCKET, key, str(dest))

    print("S3 data download complete.")

if __name__ == "__main__":
    main()
