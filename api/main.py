# api/main.py
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from datetime import date
from typing import Optional, Tuple
import pandas as pd
import traceback
import os
import time
import hashlib
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from io import StringIO, BytesIO

# --- Load the repo-root .env no matter where Uvicorn is started from ---
ROOT_DOTENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_DOTENV)

# S3 Configuration - Default to S3 bucket
S3_BUCKET = os.getenv("S3_BUCKET", "sayali-walmart-social-listener").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "data").strip().strip("/")
# Explicitly enable S3 if bucket is specified (default: enabled)
# Force S3 usage - ignore local data if S3 is configured
USE_S3 = os.getenv("USE_S3", "true").lower() in ("true", "1", "yes") and bool(S3_BUCKET)
FORCE_S3 = os.getenv("FORCE_S3", "true").lower() in ("true", "1", "yes")  # Force S3, don't fallback to local

# Local fallback
DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data")

# Initialize S3 filesystem if using S3
s3fs_filesystem = None
if USE_S3:
    try:
        import s3fs
        # Configure S3 with increased timeout and connection settings
        try:
            from botocore.config import Config
        except ImportError:
            Config = None
        
        # Get AWS region from env or default to us-east-1
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Initialize S3 filesystem - use simple approach without config to avoid conflicts
        # s3fs will handle timeouts internally
        s3fs_filesystem = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            token=os.getenv("AWS_SESSION_TOKEN") or None,
            client_kwargs={'region_name': aws_region}
        )
        print(f"[API] ✅ S3 mode enabled: s3://{S3_BUCKET}/{S3_PREFIX}/")
        print(f"[API] ✅ S3 region: {aws_region}")
        print(f"[API] ✅ S3 timeout configured: 60s connect, 300s read")
        print(f"[API] ℹ️  S3 will be tested on first data read")
    except ImportError:
        print("[API] ❌ WARNING: s3fs not installed. Install with: pip install s3fs")
        USE_S3 = False
    except Exception as e:
        print(f"[API] ❌ CRITICAL ERROR: Failed to initialize S3 filesystem: {e}")
        print(f"[API] ❌ Full error details:")
        import traceback
        traceback.print_exc()
        print(f"[API] ⚠️  S3 init failed - keeping USE_S3=True to attempt S3 reads anyway")
        print(f"[API] ⚠️  This will cause errors on data read, but will show the actual S3 error")
        # DON'T set USE_S3 = False - keep it True so paths are still S3
        # The error will be caught when trying to read, giving better error messages
        # USE_S3 = False  # REMOVED - prevents silent fallback to local
else:
    print(f"[API] ⚠️  S3 mode DISABLED - using local data directory: {DATA_DIR}/")
    print(f"[API] DEBUG: USE_S3={USE_S3}, S3_BUCKET={S3_BUCKET}, S3_PREFIX={S3_PREFIX}")

# Debug: Print final S3 configuration
print(f"[API] 🔍 FINAL CONFIG: USE_S3={USE_S3}, S3_BUCKET={S3_BUCKET}, s3fs_available={s3fs_filesystem is not None}")

def _get_file_path(filename: str) -> str:
    """Get file path (S3 or local) based on configuration."""
    if USE_S3:
        return f"s3://{S3_BUCKET}/{S3_PREFIX}/{filename}"
    else:
        return f"{DATA_DIR}/{filename}"

def _file_exists(path: str) -> bool:
    """Check if file exists in S3 or local filesystem."""
    if USE_S3 and path.startswith("s3://"):
        try:
            return s3fs_filesystem.exists(path)
        except Exception:
            return False
    else:
        return os.path.exists(path)

# Global cache for loaded dataframes (in-memory)
# Clear cache on startup to force fresh S3 load
_DATA_CACHE = {}

# Local disk cache directory (uses .cache/ which is gitignored)
CACHE_DIR = Path(".cache/s3_parquet_cache")
CACHE_MAX_AGE = 3600  # Cache valid for 1 hour

def _read_parquet(path: str, use_cache: bool = True) -> pd.DataFrame:
    """Read parquet file from S3 or local filesystem with caching.
    
    Implements:
    1. In-memory caching (fastest)
    2. Local disk caching (fast, persists across restarts)
    3. S3 fallback (slowest, but always fresh)
    """
    cache_key = path
    
    # 1. Check in-memory cache first (fastest)
    if use_cache and cache_key in _DATA_CACHE:
        print(f"[API] ⚡ Using in-memory cache for {Path(path).name}")
        return _DATA_CACHE[cache_key].copy()  # Return copy to prevent mutations
    
    # 2. Check local disk cache (fast, persists)
    if use_cache:
        cache_file = CACHE_DIR / f"{hashlib.md5(path.encode()).hexdigest()}.parquet"
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < CACHE_MAX_AGE:
                print(f"[API] 💾 Using local cache: {cache_file.name} (age: {cache_age:.0f}s)")
                df = pd.read_parquet(cache_file)
                # Also store in memory cache
                _DATA_CACHE[cache_key] = df
                return df.copy()
            else:
                # Cache expired, remove it
                cache_file.unlink()
                print(f"[API] 🗑️  Removed expired cache: {cache_file.name}")
    
    # 3. Read from S3 or local (slowest)
    try:
        if USE_S3 and path.startswith("s3://"):
            if s3fs_filesystem is None:
                raise Exception(f"S3 is enabled but s3fs_filesystem is None. S3 initialization failed during startup. Check server logs for S3 initialization errors.")
            print(f"[API] 📥 Reading from S3: {Path(path).name} (this may take a moment...)")
            start_time = time.time()
            df = pd.read_parquet(path, filesystem=s3fs_filesystem)
            elapsed = time.time() - start_time
            print(f"[API] ✅ Loaded {len(df):,} rows from S3 in {elapsed:.2f}s")
            
            # Save to both caches
            if use_cache:
                # Save to memory cache
                _DATA_CACHE[cache_key] = df
                
                # Save to disk cache
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_file, index=False)
                print(f"[API] 💾 Cached to disk: {cache_file.name}")
            
            return df
        elif USE_S3:
            # S3 is enabled but path is not S3 - this should not happen
            raise Exception(f"ERROR: S3 is enabled (USE_S3=True) but got local path: {path}. Path should be s3://{S3_BUCKET}/{S3_PREFIX}/{Path(path).name}")
        else:
            # Local file reading (only if S3 is disabled)
            print(f"[API] 📂 Reading from local: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            df = pd.read_parquet(path)
            
            # Cache local files too
            if use_cache:
                _DATA_CACHE[cache_key] = df
            
            return df
    except Exception as e:
        print(f"[API] ❌ Error reading {path}: {e}")
        import traceback
        traceback.print_exc()
        raise

# LLM summaries (exec summary + structured brief)
from src.llm.summary import build_executive_summary, summarize_tweets

def _read_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "") or ""
    return key.strip().strip('"').strip("'")

# ---- Theme computation (Stage 3 dynamic) ----
# requires src/features/themes.py with compute_themes_payload()
from src.features.themes import compute_themes_payload

# Load raw tweets data for theme generation
RAW_TWEETS_PATH = _get_file_path("tweets_stage0_raw.parquet")
RAW_TWEETS_PATH_COMP = _get_file_path("tweets_stage0_raw_comp.parquet")

# ------------ Paths ------------
SENTI_PATH = _get_file_path("tweets_stage1_sentiment.parquet")
SENTI_PATH_COMP = _get_file_path("tweets_stage1_sentiment_comp.parquet")  # Competitor (Costco) data
ASPECT_PATH = _get_file_path("tweets_stage2_aspects.parquet")
ASPECT_PATH_COMP = _get_file_path("tweets_stage2_aspects_comp.parquet")  # Competitor aspects
STAGE3_PATH = _get_file_path("tweets_stage3_aspect_sentiment.parquet")  # optional cache (no dates)
STAGE3_THEMES_PARQUET = _get_file_path("tweets_stage3_themes.parquet")  # written by /themes

app = FastAPI(title="Walmart Social Listener API")

# Global exception handler to catch any unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    tb = traceback.format_exc()
    error_msg = str(exc)
    print(f"[GLOBAL ERROR HANDLER] {error_msg}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "error": error_msg,
            "hint": "An unexpected error occurred. Check server logs for details.",
            "trace_tail": tb.splitlines()[-10:] if tb else [],
        },
    )

# Allow calls from Vite dev server (add prod origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://socialmedialistener.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Helpers ------------
def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["createdat", "created_dt", "created_at", "date"]:
        if c in df.columns:
            return c
    raise KeyError("No createdat/date column found in parquet.")

def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out["date_only"] = out[date_col].dt.date
    return out

def _sentiment_summary(sub: pd.DataFrame) -> dict:
    counts = sub["sentiment_label"].value_counts().to_dict()
    total = int(sum(counts.values()) or 1)
    pct = {k: round(v / total * 100, 2) for k, v in counts.items()}
    for k in ["positive", "neutral", "negative"]:
        counts.setdefault(k, 0)
        pct.setdefault(k, 0.0)
    return {"total": total, "counts": counts, "percent": pct}

def _aspect_split_from_subset(sub: pd.DataFrame, aspects: list[str], include_others: bool = False) -> dict:
    if sub.empty:
        zero = [0 for _ in aspects]
        zero_f = [0.0 for _ in aspects]
        labels = aspects.copy()
        if include_others:
            labels.append("others")
            zero.append(0)
            zero_f.append(0.0)
        return {
            "labels": labels,
            "counts": {"positive": zero, "neutral": zero, "negative": zero},
            "percent": {"positive": zero_f, "neutral": zero_f, "negative": zero_f},
        }

    g = (
        sub.groupby(["aspect_dominant", "sentiment_label"])
          .size().reset_index(name="count")
    )
    pivot = (
        g.pivot(index="aspect_dominant", columns="sentiment_label", values="count")
         .fillna(0)
    )
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # Handle predefined aspects
    predefined_pivot = pivot.reindex(aspects, fill_value=0)
    
    labels = aspects.copy()
    counts = {
        "positive": [int(x) for x in predefined_pivot["positive"].tolist()],
        "neutral":  [int(x) for x in predefined_pivot["neutral"].tolist()],
        "negative": [int(x) for x in predefined_pivot["negative"].tolist()],
    }
    percent = {
        "positive": [float(x) for x in predefined_pivot["positive"].tolist()],
        "neutral":  [float(x) for x in predefined_pivot["neutral"].tolist()],
        "negative": [float(x) for x in predefined_pivot["negative"].tolist()],
    }

    # Add "Others" category if requested
    if include_others:
        # Find aspects not in predefined list
        all_aspects = sub["aspect_dominant"].unique()
        other_aspects = [a for a in all_aspects if a not in aspects]
        
        if len(other_aspects) > 0:
            others_data = sub[sub["aspect_dominant"].isin(other_aspects)]
            others_counts = others_data.groupby("sentiment_label").size()
            
            labels.append("others")
            counts["positive"].append(int(others_counts.get("positive", 0)))
            counts["neutral"].append(int(others_counts.get("neutral", 0)))
            counts["negative"].append(int(others_counts.get("negative", 0)))
            
            # Calculate percentages for others
            others_total = others_counts.sum()
            if others_total > 0:
                percent["positive"].append(float((others_counts.get("positive", 0) / others_total * 100).round(2)))
                percent["neutral"].append(float((others_counts.get("neutral", 0) / others_total * 100).round(2)))
                percent["negative"].append(float((others_counts.get("negative", 0) / others_total * 100).round(2)))
            else:
                percent["positive"].append(0.0)
                percent["neutral"].append(0.0)
                percent["negative"].append(0.0)
        else:
            labels.append("others")
            counts["positive"].append(0)
            counts["neutral"].append(0)
            counts["negative"].append(0)
            percent["positive"].append(0.0)
            percent["neutral"].append(0.0)
            percent["negative"].append(0.0)

    return {
        "labels": labels,
        "counts": counts,
        "percent": percent,
    }

def _pick_any_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["createdat", "created_dt", "created_at", "tweet_date", "date", "dt"]:
        if c in df.columns:
            return c
    return None

# ------------ Data Loading (Background Thread) ------------

# Global data variables (initialized as None, loaded in background)
df = None
df_comp = None
adf = None
adf_comp = None
stage3_df = None
_sent_date_col = None
_sent_comp_date_col = None
_asp_date_col = None
_asp_comp_date_col = None
SENT_MIN_DATE = None
SENT_MAX_DATE = None
SENT_COMP_MIN_DATE = None
SENT_COMP_MAX_DATE = None
ASPECT_MIN_DATE = None
ASPECT_MAX_DATE = None
ASPECT_COMP_MIN_DATE = None
ASPECT_COMP_MAX_DATE = None
ASPECTS = ["pricing", "delivery", "returns", "staff", "app/ux"]

# Data loading status
_data_loading = threading.Event()
_data_loading_error = None

def load_all_data():
    """Load all data files in parallel (runs in background thread)."""
    global df, df_comp, adf, adf_comp, stage3_df
    global _sent_date_col, _sent_comp_date_col, _asp_date_col, _asp_comp_date_col
    global SENT_MIN_DATE, SENT_MAX_DATE, SENT_COMP_MIN_DATE, SENT_COMP_MAX_DATE
    global ASPECT_MIN_DATE, ASPECT_MAX_DATE, ASPECT_COMP_MIN_DATE, ASPECT_COMP_MAX_DATE
    global _data_loading_error
    
    try:
        print(f"[API] ⏳ Starting parallel data load from {'S3' if USE_S3 else 'local'}...")
        print(f"[API] DEBUG: USE_S3={USE_S3}, S3_BUCKET={S3_BUCKET}, S3_PREFIX={S3_PREFIX}")
        print(f"[API] DEBUG: s3fs_filesystem={s3fs_filesystem is not None}")
        
        def load_single_file(file_info):
            """Load a single file and return processed data."""
            name, path, process_func = file_info
            try:
                print(f"[API] Loading {name} from {path}...")
                print(f"[API] DEBUG: Path type - S3: {path.startswith('s3://')}, Local: {not path.startswith('s3://')}")
                # Force fresh load from S3 (disable cache on initial load)
                df = _read_parquet(path, use_cache=False)
                if process_func and df is not None:
                    df = process_func(df)
                if df is not None:
                    print(f"[API] ✅ {name} loaded: {len(df):,} rows")
                return name, df, None
            except FileNotFoundError as e:
                print(f"[API] ⚠️  {name} not found: {path}")
                return name, None, "FileNotFoundError"
            except Exception as e:
                print(f"[API] ❌ Error loading {name}: {e}")
                import traceback
                traceback.print_exc()
                return name, None, str(e)

        # Define all files to load with their processing functions
        files_to_load = [
            ("df", SENTI_PATH, lambda df: _normalize_dates(df, _detect_date_col(df))),
            ("df_comp", SENTI_PATH_COMP, lambda df: _normalize_dates(df, _detect_date_col(df))),
            ("adf", ASPECT_PATH, lambda df: _normalize_dates(df, _detect_date_col(df))),
            ("adf_comp", ASPECT_PATH_COMP, lambda df: _normalize_dates(df, _detect_date_col(df))),
            ("stage3_df", STAGE3_PATH, None),  # No processing needed
        ]

        # Load all files in parallel
        results = {}
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(load_single_file, file_info): file_info[0] for file_info in files_to_load}
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result_name, df_result, error = future.result()
                    results[result_name] = (df_result, error)
                except Exception as e:
                    print(f"[API] ❌ Unexpected error loading {name}: {e}")
                    results[name] = (None, str(e))

        elapsed = time.time() - start_time
        print(f"[API] ⚡ Parallel load completed in {elapsed:.2f}s")

        # Process results and set global variables
        # Main sentiment data
        df_result, error = results.get("df", (None, None))
        if df_result is not None:
            df = df_result
            _sent_date_col = _detect_date_col(df)
            SENT_MIN_DATE = df["date_only"].min()
            SENT_MAX_DATE = df["date_only"].max()
            print(f"[API] ✅ Main sentiment: {len(df):,} rows, date range: {SENT_MIN_DATE} to {SENT_MAX_DATE}")
        else:
            if error == "FileNotFoundError":
                raise FileNotFoundError(f"Missing parquet: {SENTI_PATH}. Ensure file exists in S3 or local data directory.")
            else:
                raise Exception(f"Failed to load main sentiment data: {error}")

        # Competitor sentiment
        df_comp_result, error = results.get("df_comp", (None, None))
        if df_comp_result is not None:
            df_comp = df_comp_result
            _sent_comp_date_col = _detect_date_col(df_comp)
            SENT_COMP_MIN_DATE = df_comp["date_only"].min()
            SENT_COMP_MAX_DATE = df_comp["date_only"].max()
            print(f"[API] ✅ Competitor sentiment: {len(df_comp):,} rows, date range: {SENT_COMP_MIN_DATE} to {SENT_COMP_MAX_DATE}")
        else:
            df_comp = None
            SENT_COMP_MIN_DATE = None
            SENT_COMP_MAX_DATE = None
            print(f"[API] ⚠️  Competitor sentiment data not available")

        # Aspects
        adf_result, error = results.get("adf", (None, None))
        if adf_result is not None:
            adf = adf_result
            _asp_date_col = _detect_date_col(adf)
            ASPECT_MIN_DATE = adf["date_only"].min()
            ASPECT_MAX_DATE = adf["date_only"].max()
            print(f"[API] ✅ Aspects: {len(adf):,} rows, date range: {ASPECT_MIN_DATE} to {ASPECT_MAX_DATE}")
        else:
            adf = pd.DataFrame(columns=["date_only", "aspect_dominant", "sentiment_label"])
            ASPECT_MIN_DATE = SENT_MIN_DATE if SENT_MIN_DATE else None
            ASPECT_MAX_DATE = SENT_MAX_DATE if SENT_MAX_DATE else None
            print(f"[API] ⚠️  Aspect data not available, using empty DataFrame")

        # Competitor aspects
        adf_comp_result, error = results.get("adf_comp", (None, None))
        if adf_comp_result is not None:
            adf_comp = adf_comp_result
            _asp_comp_date_col = _detect_date_col(adf_comp)
            ASPECT_COMP_MIN_DATE = adf_comp["date_only"].min()
            ASPECT_COMP_MAX_DATE = adf_comp["date_only"].max()
            print(f"[API] ✅ Competitor aspects: {len(adf_comp):,} rows, date range: {ASPECT_COMP_MIN_DATE} to {ASPECT_COMP_MAX_DATE}")
        else:
            adf_comp = None
            ASPECT_COMP_MIN_DATE = None
            ASPECT_COMP_MAX_DATE = None
            print(f"[API] ⚠️  Competitor aspect data not available")

        # Stage 3 cache
        stage3_df_result, error = results.get("stage3_df", (None, None))
        if stage3_df_result is not None:
            try:
                for c in ["aspect_dominant", "positive", "neutral", "negative"]:
                    assert c in stage3_df_result.columns
                stage3_df = stage3_df_result
                print(f"[API] ✅ Stage 3 cache loaded: {len(stage3_df)} aspects")
            except (AssertionError, Exception):
                stage3_df = None
                print(f"[API] ⚠️  Stage 3 cache invalid, ignoring")
        else:
            stage3_df = None
        
        print(f"[API] ✅ All data loaded successfully!")
        _data_loading.set()  # Signal that data is ready
        
    except Exception as e:
        _data_loading_error = str(e)
        print(f"[API] ❌ CRITICAL: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        _data_loading.set()  # Set event even on error so server can respond

# Start data loading in background thread
_data_loader_thread = threading.Thread(target=load_all_data, daemon=True)
_data_loader_thread.start()
print(f"[API] 🚀 Server starting... Data loading in background thread")

# ---------- Simple in-process cache for /themes ----------
_THEMES_CACHE: dict[Tuple[Optional[str], Optional[str], int, str], dict] = {}

# ------------ Helper Functions ------------
def _check_data_ready():
    """Check if data is loaded and ready."""
    if not _data_loading.is_set():
        return False, "Data is still loading from S3. Please wait..."
    if df is None:
        if _data_loading_error:
            return False, f"Data loading failed: {_data_loading_error}"
        return False, "Data is still loading from S3. Please wait..."
    return True, None

# ------------ Routes ------------
@app.get("/")
def health():
    data_ready = _data_loading.is_set() and df is not None
    
    # Get the earliest and latest dates across all data sources
    all_dates = []
    if SENT_MIN_DATE:
        all_dates.append(SENT_MIN_DATE)
    if SENT_MAX_DATE:
        all_dates.append(SENT_MAX_DATE)
    if ASPECT_MIN_DATE:
        all_dates.append(ASPECT_MIN_DATE)
    if ASPECT_MAX_DATE:
        all_dates.append(ASPECT_MAX_DATE)
    if SENT_COMP_MIN_DATE:
        all_dates.append(SENT_COMP_MIN_DATE)
    if SENT_COMP_MAX_DATE:
        all_dates.append(SENT_COMP_MAX_DATE)
    if ASPECT_COMP_MIN_DATE:
        all_dates.append(ASPECT_COMP_MIN_DATE)
    if ASPECT_COMP_MAX_DATE:
        all_dates.append(ASPECT_COMP_MAX_DATE)
    
    # Calculate overall min and max
    overall_min = min(all_dates) if all_dates else None
    overall_max = max(all_dates) if all_dates else None
    
    return {
        "message": "✅ Walmart Sentiment API is running!",
        "data_loading": not data_ready,
        "data_loading_error": _data_loading_error if _data_loading_error else None,
        "data_source": f"S3: s3://{S3_BUCKET}/{S3_PREFIX}/" if USE_S3 else f"Local: {DATA_DIR}/",
        "date_range": {"min": str(overall_min) if overall_min else None, "max": str(overall_max) if overall_max else None} if data_ready else None,
        "min": str(overall_min) if overall_min else None,  # For backward compatibility
        "max": str(overall_max) if overall_max else None,  # For backward compatibility
        "sentiment_date_range": {"min": str(SENT_MIN_DATE) if SENT_MIN_DATE else None, "max": str(SENT_MAX_DATE) if SENT_MAX_DATE else None} if data_ready else None,
        "aspect_date_range": {"min": str(ASPECT_MIN_DATE) if ASPECT_MIN_DATE else None, "max": str(ASPECT_MAX_DATE) if ASPECT_MAX_DATE else None} if data_ready else None,
        "has_aspects": bool(len(adf) > 0) if data_ready and adf is not None else False,
        "has_stage3_cache": bool(stage3_df is not None) if data_ready else False,
        "has_competitor_data": bool(df_comp is not None) if data_ready else False,
        "competitor_date_range": {"min": str(SENT_COMP_MIN_DATE) if SENT_COMP_MIN_DATE else None, "max": str(SENT_COMP_MAX_DATE) if SENT_COMP_MAX_DATE else None} if data_ready else None,
        "env_loaded": os.path.exists(ROOT_DOTENV),
        "has_openai_key": bool(_read_openai_key()),
    }

# --- Sentiment ---
@app.get("/sentiment/summary")
def sentiment_summary(
    start: date = Query(default=None),
    end:   date = Query(default=None),
):
    ready, error_msg = _check_data_ready()
    if not ready:
        return JSONResponse(
            status_code=503,
            content={"error": error_msg, "retry_after": 5}
        )
    
    # Use defaults if not provided
    if start is None:
        start = SENT_MIN_DATE
    if end is None:
        end = SENT_MAX_DATE
    
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {
            "start": str(start), "end": str(end),
            "total": 0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "percent": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }
    return {"start": str(start), "end": str(end), **_sentiment_summary(sub)}

@app.get("/sentiment/trend")
def sentiment_trend(
    start: date = Query(default=SENT_MIN_DATE),
    end:   date = Query(default=SENT_MAX_DATE),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {"start": str(start), "end": str(end), "trend": []}

    # ---- counts per day per sentiment ----
    daily_counts = (
        sub.groupby(["date_only", "sentiment_label"])
           .size().reset_index(name="count")
           .pivot(index="date_only", columns="sentiment_label", values="count")
           .fillna(0)
    )

    for c in ["positive", "neutral", "negative"]:
        if c not in daily_counts.columns:
            daily_counts[c] = 0

    daily_counts = daily_counts[["positive", "neutral", "negative"]].copy()
    totals = daily_counts.sum(axis=1).replace(0, 1)

    # ---- percent per day ----
    daily_pct = (daily_counts.div(totals, axis=0) * 100).round(2)

    trend = []
    for d in daily_counts.index:
        trend.append({
            "date": str(d),
            # keep existing keys (PERCENT) for chart lines
            "positive": float(daily_pct.loc[d, "positive"]),
            "neutral":  float(daily_pct.loc[d, "neutral"]),
            "negative": float(daily_pct.loc[d, "negative"]),
            # add counts for tooltip
            "counts": {
                "positive": int(daily_counts.loc[d, "positive"]),
                "neutral":  int(daily_counts.loc[d, "neutral"]),
                "negative": int(daily_counts.loc[d, "negative"]),
            },
            "total": int(daily_counts.loc[d].sum()),
        })

    return {"start": str(start), "end": str(end), "trend": trend}


# --- Competitor Sentiment (Costco) ---
@app.get("/sentiment/competitor/summary")
def sentiment_competitor_summary(
    start: date = Query(default=None),
    end:   date = Query(default=None),
):
    if df_comp is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Competitor data not available. Missing tweets_stage1_sentiment_comp.parquet"}
        )
    
    s = start or SENT_COMP_MIN_DATE
    e = end or SENT_COMP_MAX_DATE
    
    mask = (df_comp["date_only"] >= s) & (df_comp["date_only"] <= e)
    sub = df_comp.loc[mask]
    if sub.empty:
        return {
            "start": str(s), "end": str(e),
            "total": 0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "percent": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }
    return {"start": str(s), "end": str(e), **_sentiment_summary(sub)}

@app.get("/sentiment/competitor/trend")
@app.get("/sentiment/competitor/trend")
def sentiment_competitor_trend(
    start: date = Query(default=None),
    end:   date = Query(default=None),
):
    if df_comp is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Competitor data not available. Missing tweets_stage1_sentiment_comp.parquet"}
        )

    s = start or SENT_COMP_MIN_DATE
    e = end or SENT_COMP_MAX_DATE

    mask = (df_comp["date_only"] >= s) & (df_comp["date_only"] <= e)
    sub = df_comp.loc[mask]
    if sub.empty:
        return {"start": str(s), "end": str(e), "trend": []}

    daily_counts = (
        sub.groupby(["date_only", "sentiment_label"])
           .size().reset_index(name="count")
           .pivot(index="date_only", columns="sentiment_label", values="count")
           .fillna(0)
    )

    for c in ["positive", "neutral", "negative"]:
        if c not in daily_counts.columns:
            daily_counts[c] = 0

    daily_counts = daily_counts[["positive", "neutral", "negative"]].copy()
    totals = daily_counts.sum(axis=1).replace(0, 1)
    daily_pct = (daily_counts.div(totals, axis=0) * 100).round(2)

    trend = []
    for d in daily_counts.index:
        trend.append({
            "date": str(d),
            "positive": float(daily_pct.loc[d, "positive"]),
            "neutral":  float(daily_pct.loc[d, "neutral"]),
            "negative": float(daily_pct.loc[d, "negative"]),
            "counts": {
                "positive": int(daily_counts.loc[d, "positive"]),
                "neutral":  int(daily_counts.loc[d, "neutral"]),
                "negative": int(daily_counts.loc[d, "negative"]),
            },
            "total": int(daily_counts.loc[d].sum()),
        })

    return {"start": str(s), "end": str(e), "trend": trend}

# --- Aspects (simple distribution) ---
@app.get("/aspects/summary")
def aspects_summary(
    start: date = Query(default=None),
    end:   date = Query(default=None),
    as_percent: bool = Query(default=False),
):
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE

    if adf.empty:
        counts = {a: 0 for a in ASPECTS}
        pct = {a: 0.0 for a in ASPECTS}
        return {
            "start": str(s), "end": str(e),
            "counts": counts, "percent": pct, "total": 0,
            "labels": ASPECTS, "series": (pct if as_percent else counts)
        }

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]

    if sub.empty:
        counts = {a: 0 for a in ASPECTS}
        pct = {a: 0.0 for a in ASPECTS}
        return {
            "start": str(s), "end": str(e),
            "counts": counts, "percent": pct, "total": 0,
            "labels": ASPECTS, "series": (pct if as_percent else counts)
        }

    dom_counts = sub["aspect_dominant"].value_counts().to_dict()
    counts = {a: int(dom_counts.get(a, 0)) for a in ASPECTS}
    total = int(sum(counts.values()))
    percent = {a: round((counts[a] / total * 100), 2) if total else 0.0 for a in ASPECTS}

    return {
        "start": str(s),
        "end": str(e),
        "counts": counts,
        "percent": percent,
        "total": total,
        "labels": ASPECTS,
        "series": (percent if as_percent else counts),
    }

@app.get("/aspects/avg-scores")
def aspects_avg_scores(
    start: date = Query(default=None),
    end:   date = Query(default=None),
):
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE

    if adf.empty:
        return {"start": str(s), "end": str(e), "avg_scores": {}}

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]

    score_cols = [c for c in adf.columns if c.startswith("aspect_") and c != "aspect_dominant"]
    if sub.empty or not score_cols:
        return {"start": str(s), "end": str(e), "avg_scores": {c: 0.0 for c in score_cols}}

    avg = sub[score_cols].mean().to_dict()
    avg = {k: round(float(v), 4) for k, v in avg.items()}
    return {"start": str(s), "end": str(e), "avg_scores": avg}

# --- Aspects (stacked bar: sentiment split per aspect) ---
@app.get("/aspects/sentiment-split")
def aspects_sentiment_split(
    start: date = Query(default=None),
    end:   date = Query(default=None),
    as_percent: bool = Query(default=False),
    include_others: bool = Query(default=False),
):
    if start is None and end is None and stage3_df is not None:
        s3 = stage3_df.set_index("aspect_dominant").reindex(ASPECTS, fill_value=0)
        for c in ["positive", "neutral", "negative"]:
            if c not in s3.columns:
                s3[c] = 0
        totals = s3[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1)
        pct = (s3[["positive", "neutral", "negative"]].div(totals, axis=0) * 100).round(2)

        labels = ASPECTS.copy()
        counts = {
                "positive": [int(x) for x in s3["positive"].tolist()],
                "neutral":  [int(x) for x in s3["neutral"].tolist()],
                "negative": [int(x) for x in s3["negative"].tolist()],
        }
        percent = {
                "positive": [float(x) for x in pct["positive"].tolist()],
                "neutral":  [float(x) for x in pct["neutral"].tolist()],
                "negative": [float(x) for x in pct["negative"].tolist()],
        }

        # Add "Others" category if requested
        if include_others:
            # Calculate others by finding aspects not in predefined list
            all_aspects = stage3_df["aspect_dominant"].unique()
            other_aspects = [a for a in all_aspects if a not in ASPECTS]
            
            if len(other_aspects) > 0:
                others_data = stage3_df[stage3_df["aspect_dominant"].isin(other_aspects)]
                others_counts = others_data[["positive", "neutral", "negative"]].sum()
                
                labels.append("others")
                counts["positive"].append(int(others_counts["positive"]))
                counts["neutral"].append(int(others_counts["neutral"]))
                counts["negative"].append(int(others_counts["negative"]))
                
                # Calculate percentages for others
                others_total = others_counts.sum()
                if others_total > 0:
                    percent["positive"].append(float((others_counts["positive"] / others_total * 100).round(2)))
                    percent["neutral"].append(float((others_counts["neutral"] / others_total * 100).round(2)))
                    percent["negative"].append(float((others_counts["negative"] / others_total * 100).round(2)))
                else:
                    percent["positive"].append(0.0)
                    percent["neutral"].append(0.0)
                    percent["negative"].append(0.0)

        payload = {
            "labels": labels,
            "counts": counts,
            "percent": percent,
        }
        return payload

    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    if adf.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS, include_others)

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS, include_others)

# --- Competitor Aspects (stacked bar: sentiment split per aspect) ---
@app.get("/aspects/competitor/sentiment-split")
def aspects_competitor_sentiment_split(
    start: date = Query(default=None),
    end:   date = Query(default=None),
    as_percent: bool = Query(default=False),
    include_others: bool = Query(default=False),
):
    if adf_comp is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Competitor aspect data not available. Missing tweets_stage2_aspects_comp.parquet"}
        )
    
    s = start or ASPECT_COMP_MIN_DATE
    e = end or ASPECT_COMP_MAX_DATE
    if adf_comp.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS, include_others)

    mask = (adf_comp["date_only"] >= s) & (adf_comp["date_only"] <= e)
    sub = adf_comp.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS, include_others)

# --- Executive summary over a date window (LLM-powered with fallback) ---
@app.get("/executive-summary")
def executive_summary(
    start: str = Query(..., description="YYYY-MM-DD"),
    end:   str = Query(..., description="YYYY-MM-DD"),
    sample_per_sentiment: int = Query(default=250, ge=50, le=500),
):
    """
    Summarizes all tweets in the selected duration.
    Uses OpenAI if OPENAI_API_KEY is configured, otherwise a rule-based fallback.
    Returns: {start, end, used_llm, summary, stats:{sentiment, top_aspects, keywords}}
    """
    try:
        result = build_executive_summary(
            df_senti=df,            # Stage 1 DF (has date_only, sentiment_label, text cols)
            df_aspects=adf,         # Stage 2 DF (for top aspects)
            start=start,
            end=end,
            openai_api_key=_read_openai_key(),
            sample_per_sentiment=sample_per_sentiment,
        )
        return result
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace_tail": tb.splitlines()[-6:]},
        )

# --- Structured brief (bullets/themes/risks/opps) ---
@app.get("/structured-brief")
def structured_brief(
    start: str = Query(..., description="YYYY-MM-DD"),
    end:   str = Query(..., description="YYYY-MM-DD"),
    keyword: Optional[str] = Query(default=None),
    sample_size: int = Query(default=50, ge=20, le=200),
):
    try:
        # Work on a copy and ensure we expose exactly ONE 'date' column
        df_for_llm = df.copy()

        if "date" in df_for_llm.columns:
            pass  # already present
        elif "date_only" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["date_only"]
        elif "created_at" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["created_at"]
        elif "timestamp" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["timestamp"]
        else:
            maybe = [c for c in df_for_llm.columns if "date" in c or "time" in c]
            if maybe:
                df_for_llm["date"] = df_for_llm[maybe[0]]
            else:
                df_for_llm["date"] = pd.NaT

        # If multiple 'date' columns somehow exist, keep the first and drop the rest
        if (df_for_llm.columns == "date").sum() > 1:
            first_idx = [i for i, c in enumerate(df_for_llm.columns) if c == "date"][0]
            keep = list(range(len(df_for_llm.columns)))
            for i, c in enumerate(df_for_llm.columns):
                if c == "date" and i != first_idx:
                    keep.remove(i)
            df_for_llm = df_for_llm.iloc[:, keep]

        res = summarize_tweets(
            df=df_for_llm,
            start_date=start,
            end_date=end,
            keyword=keyword,
            sample_size=sample_size,
        )
        return res
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace_tail": tb.splitlines()[-6:]},
        )

# --- Themes (dynamic clustering + summaries) ---
@app.get("/themes")
def themes(
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    n_clusters: Optional[int] = Query(default=None, ge=1, le=8),  # Auto-detect if None
    emb_model: str       = Query(default="sentence-transformers/all-MiniLM-L6-v2"),
    merge_similar: bool  = Query(default=True, description="Automatically merge similar themes"),
    force_refresh: bool = Query(default=False, description="Force regeneration even if cached"),
):
    """
    Returns:
      { updated_at, used_llm, themes: [{id,name,summary,tweet_count,positive,negative,neutral}] }
    """
    key = (start, end, n_clusters, emb_model)
    
    # Check cache first (unless force_refresh is True)
    if not force_refresh and key in _THEMES_CACHE:
        print(f"[/themes] Returning cached result for key: {key}")
        return _THEMES_CACHE[key]

    try:
        print(f"[/themes] Generating themes for date range: {start} to {end}, clusters: {n_clusters}")
        
        # Load raw tweets data for theme generation
        print(f"[/themes] Loading raw tweets from {RAW_TWEETS_PATH}")
        try:
            df = _read_parquet(RAW_TWEETS_PATH)
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Raw tweets file not found: {RAW_TWEETS_PATH}",
                    "hint": "Ensure the data file exists in S3 or local data directory.",
                },
            )
        print(f"[/themes] Loaded {len(df)} tweets")
        
        # Get OpenAI key
        openai_key = _read_openai_key()
        if not openai_key:
            print("[/themes] ⚠️  WARNING: No OpenAI API key found. Theme names/summaries will use fallback values.")
            print("[/themes] To enable AI-generated themes, set the OPENAI_API_KEY environment variable.")
        else:
            print(f"[/themes] ✅ OpenAI API key found (length: {len(openai_key)}). AI generation enabled.")
        
        print(f"[/themes] Computing themes payload...")
        payload = compute_themes_payload(
            df=df,
            start_date=start,
            end_date=end,
            n_clusters=n_clusters,
            openai_api_key=openai_key,
            merge_similar=merge_similar,
        )
        
        # Cache the result
        _THEMES_CACHE[key] = payload
        ai_status = "✅ AI" if payload.get('used_llm') else "⚠️  Fallback"
        print(f"[/themes] Successfully generated {len(payload.get('themes', []))} themes ({ai_status})")
        return payload
    except FileNotFoundError as e:
        tb = traceback.format_exc()
        print("[/themes ERROR] File not found:", e, "\n", tb)
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Data file not found: {str(e)}",
                "hint": "Ensure the data files exist in the data/ directory.",
            },
        )
    except MemoryError as e:
        tb = traceback.format_exc()
        print("[/themes ERROR] Memory error:", e, "\n", tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Insufficient memory to process themes",
                "hint": "Try reducing the date range or number of clusters. The system is processing too much data at once.",
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e)
        print("[/themes ERROR]", error_msg, "\n", tb)
        
        # Provide more specific error messages
        hint = "Check OpenAI key, sklearn/torch availability, date range, and data paths."
        if "OpenAI" in error_msg or "API" in error_msg:
            hint = "OpenAI API error. Check your API key and network connection. Theme generation will continue with fallback names/summaries."
        elif "timeout" in error_msg.lower():
            hint = "Request timed out. Try reducing the date range or number of clusters."
        elif "sklearn" in error_msg.lower() or "torch" in error_msg.lower():
            hint = "Missing required library. Install sklearn and/or torch: pip install scikit-learn torch"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "hint": hint,
                "trace_tail": tb.splitlines()[-10:],
            },
        )

# --- Themes for Competitor (Costco) ---
@app.get("/themes/competitor")
def themes_competitor(
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    n_clusters: Optional[int] = Query(default=None, ge=1, le=8),  # Auto-detect if None
    emb_model: str       = Query(default="sentence-transformers/all-MiniLM-L6-v2"),
    merge_similar: bool  = Query(default=True, description="Automatically merge similar themes"),
    force_refresh: bool = Query(default=False, description="Force regeneration even if cached"),
):
    """
    Returns:
      { updated_at, used_llm, themes: [{id,name,summary,tweet_count,positive,negative,neutral}] }
    """
    key = (start, end, n_clusters, emb_model, "competitor")
    
    # Check cache first (unless force_refresh is True)
    if not force_refresh and key in _THEMES_CACHE:
        print(f"[/themes/competitor] Returning cached result for key: {key}")
        return _THEMES_CACHE[key]

    try:
        print(f"[/themes/competitor] Generating themes for date range: {start} to {end}, clusters: {n_clusters}")
        
        # Check if raw tweets file exists, fallback to sentiment file if available
        df = None
        
        # First, check if files exist (more efficient than trying to read)
        raw_exists = _file_exists(RAW_TWEETS_PATH_COMP)
        sent_exists = _file_exists(SENTI_PATH_COMP)
        
        if not raw_exists and not sent_exists:
            error_msg = f"Competitor data file not found"
            print(f"[/themes/competitor] ERROR: {error_msg}")
            print(f"[/themes/competitor] Checked: {RAW_TWEETS_PATH_COMP} (exists: {raw_exists})")
            print(f"[/themes/competitor] Checked: {SENTI_PATH_COMP} (exists: {sent_exists})")
            return JSONResponse(
                status_code=404,
                content={
                    "error": error_msg,
                    "hint": "The competitor (Costco) data file is missing. Please ensure either tweets_stage0_raw_comp.parquet or tweets_stage1_sentiment_comp.parquet exists in the data/ directory or S3 bucket.",
                    "file_paths": {
                        "raw": RAW_TWEETS_PATH_COMP,
                        "sentiment": SENTI_PATH_COMP
                    },
                },
            )
        
        # Try to load raw file first if it exists
        if raw_exists:
            try:
                print(f"[/themes/competitor] Loading raw tweets from {RAW_TWEETS_PATH_COMP}")
                df = _read_parquet(RAW_TWEETS_PATH_COMP)
                print(f"[/themes/competitor] Loaded {len(df)} tweets from raw file")
            except Exception as e:
                error_msg = f"Failed to load competitor raw data file: {str(e)}"
                print(f"[/themes/competitor] ERROR: {error_msg}")
                # Fallback to sentiment file if raw file read fails
                if sent_exists:
                    print(f"[/themes/competitor] Falling back to sentiment file: {SENTI_PATH_COMP}")
                    try:
                        df = _read_parquet(SENTI_PATH_COMP)
                        print(f"[/themes/competitor] Loaded {len(df)} tweets from sentiment file")
                    except Exception as e2:
                        error_msg = f"Failed to load competitor sentiment data file: {str(e2)}"
                        print(f"[/themes/competitor] ERROR: {error_msg}")
                        return JSONResponse(
                            status_code=500,
                            content={
                                "error": error_msg,
                                "hint": "Both competitor data files exist but could not be read. Please check if the files are corrupted or if there are S3 access issues.",
                                "file_paths": {
                                    "raw": RAW_TWEETS_PATH_COMP,
                                    "sentiment": SENTI_PATH_COMP
                                },
                            },
                        )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": error_msg,
                            "hint": "The competitor raw data file exists but could not be read. Please check if the file is corrupted.",
                            "file_path": RAW_TWEETS_PATH_COMP,
                        },
                    )
        elif sent_exists:
            # Only sentiment file exists, use it
            try:
                print(f"[/themes/competitor] Loading sentiment file: {SENTI_PATH_COMP}")
                df = _read_parquet(SENTI_PATH_COMP)
                print(f"[/themes/competitor] Loaded {len(df)} tweets from sentiment file")
            except Exception as e:
                error_msg = f"Failed to load competitor sentiment data file: {str(e)}"
                print(f"[/themes/competitor] ERROR: {error_msg}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": error_msg,
                        "hint": "The competitor sentiment data file exists but could not be read. Please check if the file is corrupted.",
                        "file_path": SENTI_PATH_COMP,
                    },
                )
        
        # Check if df is still None (shouldn't happen, but safety check)
        if df is None:
            error_msg = f"Competitor data file not found"
            print(f"[/themes/competitor] ERROR: {error_msg}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": error_msg,
                    "hint": "The competitor (Costco) data file is missing. Please ensure either tweets_stage0_raw_comp.parquet or tweets_stage1_sentiment_comp.parquet exists.",
                    "file_paths": {
                        "raw": RAW_TWEETS_PATH_COMP,
                        "sentiment": SENTI_PATH_COMP
                    },
                },
            )
        
        # Get OpenAI key
        openai_key = _read_openai_key()
        if not openai_key:
            print("[/themes/competitor] ⚠️  WARNING: No OpenAI API key found. Theme names/summaries will use fallback values.")
            print("[/themes/competitor] To enable AI-generated themes, set the OPENAI_API_KEY environment variable.")
        else:
            print(f"[/themes/competitor] ✅ OpenAI API key found (length: {len(openai_key)}). AI generation enabled.")
        
        print(f"[/themes/competitor] Computing themes payload...")
        payload = compute_themes_payload(
            df=df,
            start_date=start,
            end_date=end,
            n_clusters=n_clusters,
            openai_api_key=openai_key,
            merge_similar=merge_similar,
        )
        
        # Cache the result
        _THEMES_CACHE[key] = payload
        ai_status = "✅ AI" if payload.get('used_llm') else "⚠️  Fallback"
        print(f"[/themes/competitor] Successfully generated {len(payload.get('themes', []))} themes ({ai_status})")
        return payload
    except FileNotFoundError as e:
        tb = traceback.format_exc()
        print("[/themes/competitor ERROR] File not found:", e, "\n", tb)
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Data file not found: {str(e)}",
                "hint": "Ensure the competitor data files exist in the data/ directory.",
            },
        )
    except MemoryError as e:
        tb = traceback.format_exc()
        print("[/themes/competitor ERROR] Memory error:", e, "\n", tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Insufficient memory to process themes",
                "hint": "Try reducing the date range or number of clusters. The system is processing too much data at once.",
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e)
        print("[/themes/competitor ERROR]", error_msg, "\n", tb)
        
        # Provide more specific error messages
        hint = "Check OpenAI key, sklearn/torch availability, date range, and data paths."
        if "OpenAI" in error_msg or "API" in error_msg:
            hint = "OpenAI API error. Check your API key and network connection. Theme generation will continue with fallback names/summaries."
        elif "timeout" in error_msg.lower():
            hint = "Request timed out. Try reducing the date range or number of clusters."
        elif "sklearn" in error_msg.lower() or "torch" in error_msg.lower():
            hint = "Missing required library. Install sklearn and/or torch: pip install scikit-learn torch"
        
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "hint": hint,
                "trace_tail": tb.splitlines()[-10:],
            },
        )

# --- Tweets drill-down for a theme (reads STAGE3_THEMES_PARQUET) ---
@app.get("/themes/{theme_id}/tweets")
def theme_tweets(
    theme_id: int,
    limit: int = Query(default=10, ge=1, le=200),
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    try:
        df3 = _read_parquet(STAGE3_THEMES_PARQUET)
    except FileNotFoundError:
        return {"items": [], "note": "Stage-3 parquet not found. Call /themes first to generate."}
    if "theme" not in df3.columns:
        return {"items": [], "note": "'theme' column not present in stage-3 parquet."}

    date_col = _pick_any_date_col(df3)
    if date_col:
        df3[date_col] = pd.to_datetime(df3[date_col], errors="coerce")
        if start:
            df3 = df3[df3[date_col] >= pd.to_datetime(start)]
        if end:
            df3 = df3[df3[date_col] <= pd.to_datetime(end)]

    sub = df3[df3["theme"] == int(theme_id)].copy()
    if sub.empty:
        return {"items": []}

    if date_col:
        sub = sub.sort_values(date_col, ascending=False)

    cols_keep = [c for c in [
        date_col, "sentiment_label", "sentiment_score",
        "aspect_dominant", "twitter_url", "tweet_url",
        "text_used", "clean_tweet", "text", "fulltext"
    ] if c and c in sub.columns]

    def _pick_text(row: dict) -> str:
        for c in ["text_used", "clean_tweet", "text", "fulltext"]:
            if c in row and c in row and row[c]:
                return str(row[c])
        return ""

    items = []
    for _, r in sub[cols_keep].head(int(limit)).iterrows():
        d = r.to_dict()
        created = str(d.get(date_col)) if date_col else ""
        url_val = d.get("twitter_url") or d.get("tweet_url") or ""
        items.append({
            "date": created,                 # new
            "createdat": created,            # legacy-friendly
            "sentiment_label": d.get("sentiment_label"),
            "sentiment_score": d.get("sentiment_score"),
            "aspect_dominant": d.get("aspect_dominant"),
            "url": url_val,                  # new
            "twitterurl": url_val,           # legacy-friendly
            "text": _pick_text(d),           # new
            "text_clean": _pick_text(d),     # legacy-friendly
        })

    return {"items": items}

# --- Sample tweets for specific aspect and sentiment ---
@app.get("/tweets/sample")
def sample_tweets(
    start: date = Query(default=None),
    end: date = Query(default=None),
    aspect: str = Query(..., description="Aspect name (e.g., pricing, delivery)"),
    sentiment: str = Query(..., description="Sentiment (positive, neutral, negative)"),
    limit: int = Query(default=10, ge=1, le=1000),
):
    """
    Returns sample tweets for a specific aspect and sentiment combination.
    """
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    
    if adf.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}
    
    # Filter by date range
    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]
    
    if sub.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}
    
    # Filter by aspect and sentiment
    aspect_mask = sub["aspect_dominant"].str.lower() == aspect.lower()
    sentiment_mask = sub["sentiment_label"].str.lower() == sentiment.lower()
    filtered = sub[aspect_mask & sentiment_mask]
    
    if filtered.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}
    
    # Get text column (try different possible column names)
    text_col = None
    for col in ["text", "clean_tweet", "text_used", "fulltext", "tweet_text"]:
        if col in filtered.columns:
            text_col = col
            break
    
    if text_col is None:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment, "error": "No text column found"}
    
    # Sample tweets
    sample_tweets = filtered[text_col].dropna().head(limit).tolist()
    
    return {
        "tweets": sample_tweets,
        "count": len(sample_tweets),
        "aspect": aspect,
        "sentiment": sentiment,
        "total_available": len(filtered)
    }

# --- Raw Data Downloads ---
@app.get("/tweets/raw")
def download_raw_tweets(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="csv", regex="^(csv|xlsx)$")
):
    """Download raw tweets data in CSV or Excel format"""
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    
    if sub.empty:
        return {"error": "No data found for the specified date range"}
    
    # Select relevant columns
    columns = ["date", "createdat", "text", "text_clean", "sentiment_label", "aspect_dominant", "user_id"]
    available_columns = [col for col in columns if col in sub.columns]
    export_data = sub[available_columns].copy()
    
    if format == "csv":
        csv_buffer = StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=raw_tweets_{start}_to_{end}.csv"}
        )
    else:  # xlsx
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_data.to_excel(writer, sheet_name='Raw Tweets', index=False)
        excel_content = excel_buffer.getvalue()
        
        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=raw_tweets_{start}_to_{end}.xlsx"}
        )

@app.get("/reports/sentiment")
def download_sentiment_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$")
):
    """Download comprehensive sentiment analysis report"""
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    
    if sub.empty:
        return {"error": "No data found for the specified date range"}
    
    # Generate sentiment summary
    sentiment_data = _sentiment_summary(sub)
    
    if format == "pdf":
        # Create PDF content (simplified version)
        html_content = f"""
        <html>
        <head><title>Sentiment Analysis Report</title></head>
        <body>
            <h1>Sentiment Analysis Report</h1>
            <p>Date Range: {start} to {end}</p>
            <h2>Summary</h2>
            <p>Total Tweets: {sentiment_data['total']}</p>
            <p>Positive: {sentiment_data['counts']['positive']} ({sentiment_data['percent']['positive']}%)</p>
            <p>Neutral: {sentiment_data['counts']['neutral']} ({sentiment_data['percent']['neutral']}%)</p>
            <p>Negative: {sentiment_data['counts']['negative']} ({sentiment_data['percent']['negative']}%)</p>
        </body>
        </html>
        """
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=sentiment_report_{start}_to_{end}.html"}
        )
    else:  # xlsx
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([
                {"Metric": "Total Tweets", "Count": sentiment_data['total']},
                {"Metric": "Positive", "Count": sentiment_data['counts']['positive'], "Percentage": sentiment_data['percent']['positive']},
                {"Metric": "Neutral", "Count": sentiment_data['counts']['neutral'], "Percentage": sentiment_data['percent']['neutral']},
                {"Metric": "Negative", "Count": sentiment_data['counts']['negative'], "Percentage": sentiment_data['percent']['negative']}
            ])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Raw data sheet
            columns = ["date", "createdat", "text", "text_clean", "sentiment_label"]
            available_columns = [col for col in columns if col in sub.columns]
            sub[available_columns].to_excel(writer, sheet_name='Raw Data', index=False)
        
        excel_content = excel_buffer.getvalue()
        
        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=sentiment_report_{start}_to_{end}.xlsx"}
        )

@app.get("/reports/aspects")
def download_aspect_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$")
):
    """Download aspect analysis report"""
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    
    if sub.empty:
        return {"error": "No data found for the specified date range"}
    
    # Get aspect data
    aspects = ["pricing", "delivery", "customer_service", "product_quality", "store_experience"]
    aspect_data = _aspect_split_from_subset(sub, aspects, include_others=True)
    
    if format == "pdf":
        html_content = f"""
        <html>
        <head><title>Aspect Analysis Report</title></head>
        <body>
            <h1>Aspect Analysis Report</h1>
            <p>Date Range: {start} to {end}</p>
            <h2>Aspect Breakdown</h2>
        """
        for i, aspect in enumerate(aspect_data['labels']):
            html_content += f"<p>{aspect}: {aspect_data['counts'][i]} tweets ({aspect_data['percentages'][i]}%)</p>"
        
        html_content += "</body></html>"
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=aspect_report_{start}_to_{end}.html"}
        )
    else:  # xlsx
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Aspect summary
            aspect_df = pd.DataFrame({
                "Aspect": aspect_data['labels'],
                "Count": aspect_data['counts'],
                "Percentage": aspect_data['percentages']
            })
            aspect_df.to_excel(writer, sheet_name='Aspect Summary', index=False)
            
            # Raw data with aspects
            columns = ["date", "createdat", "text", "text_clean", "sentiment_label", "aspect_dominant"]
            available_columns = [col for col in columns if col in sub.columns]
            sub[available_columns].to_excel(writer, sheet_name='Raw Data', index=False)
        
        excel_content = excel_buffer.getvalue()
        
        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=aspect_report_{start}_to_{end}.xlsx"}
        )

@app.get("/reports/themes")
def download_theme_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$")
):
    """Download theme analysis report"""
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    
    if sub.empty:
        return {"error": "No data found for the specified date range"}
    
    if format == "pdf":
        html_content = f"""
        <html>
        <head><title>Theme Analysis Report</title></head>
        <body>
            <h1>Theme Analysis Report</h1>
            <p>Date Range: {start} to {end}</p>
            <p>Total Tweets Analyzed: {len(sub)}</p>
            <h2>Note</h2>
            <p>Theme analysis requires AI processing. Please use the Theme Analysis page to generate themes first.</p>
        </body>
        </html>
        """
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=theme_report_{start}_to_{end}.html"}
        )

@app.get("/reports/theme/{theme_id}")
def download_theme_tweets_report(
    theme_id: int,
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    limit: int = Query(default=200, ge=1, le=1000)
):
    """Download PDF report for specific theme tweets"""
    try:
        # Load theme data
        themes_df = _read_parquet(STAGE3_THEMES_PARQUET)
        
        # Filter by theme ID
        theme_tweets = themes_df[themes_df['theme'] == theme_id]
        
        if theme_tweets.empty:
            return {"error": f"No tweets found for theme {theme_id}"}
        
        # Apply date filter if provided
        if start:
            theme_tweets = theme_tweets[theme_tweets['createdat'] >= pd.to_datetime(start)]
        if end:
            theme_tweets = theme_tweets[theme_tweets['createdat'] <= pd.to_datetime(end)]
        
        if theme_tweets.empty:
            return {"error": f"No tweets found for theme {theme_id} in the specified date range"}
        
        # Calculate sentiment breakdown on ALL tweets (before limiting for display)
        sentiment_counts = theme_tweets['sentiment_label'].value_counts()
        total_tweets = len(theme_tweets)
        
        # Get theme info from themes cache
        theme_info = None
        for cache_file in os.listdir("data"):
            if cache_file.startswith("themes_cache_"):
                try:
                    with open(f"{DATA_DIR}/{cache_file}", "r") as f:
                        cache_data = json.load(f)
                        for theme in cache_data.get('themes', []):
                            if theme.get('id') == theme_id:
                                theme_info = theme
                                break
                        if theme_info:
                            break
                except Exception as e:
                    print(f"Error reading cache file {cache_file}: {e}")
                    continue
        
        if not theme_info:
            # Try to get theme info from the themes parquet file
            try:
                themes_df = _read_parquet(STAGE3_THEMES_PARQUET)
                theme_tweets_sample = themes_df[themes_df['theme'] == theme_id]
                if not theme_tweets_sample.empty:
                    # Get the most common aspect for this theme
                    most_common_aspect = theme_tweets_sample['aspect_dominant'].mode().iloc[0] if not theme_tweets_sample['aspect_dominant'].mode().empty else 'unknown'
                    theme_info = {
                        "name": f"{most_common_aspect.replace('_', ' ').title()} Analysis",
                        "summary": f"Analysis of tweets related to {most_common_aspect.replace('_', ' ')}",
                        "tweet_count": total_tweets
                    }
                else:
                    theme_info = {
                        "name": f"Theme {theme_id}",
                        "summary": "Theme analysis report",
                        "tweet_count": total_tweets
                    }
            except Exception as e:
                print(f"Error reading themes parquet: {e}")
                theme_info = {
                    "name": f"Theme {theme_id}",
                    "summary": "Theme analysis report",
                    "tweet_count": total_tweets
                }
        
        # Limit results for display (after calculating stats)
        theme_tweets_display = theme_tweets.head(limit)
        
        positive_count = sentiment_counts.get('positive', 0)
        negative_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        positive_pct = round((positive_count / total_tweets) * 100, 1) if total_tweets > 0 else 0
        negative_pct = round((negative_count / total_tweets) * 100, 1) if total_tweets > 0 else 0
        neutral_pct = round((neutral_count / total_tweets) * 100, 1) if total_tweets > 0 else 0
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Theme Report: {theme_info.get('name', f'Themes {theme_id}')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .tweet {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; }}
                .tweet.positive {{ border-left: 4px solid #28a745; }}
                .tweet.negative {{ border-left: 4px solid #dc3545; }}
                .tweet.neutral {{ border-left: 4px solid #ffc107; }}
                .sentiment {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; }}
                .positive {{ background: #d4edda; color: #155724; }}
                .negative {{ background: #f8d7da; color: #721c24; }}
                .neutral {{ background: #fff3cd; color: #856404; }}
                .date {{ color: #666; font-size: 0.9em; }}
                .aspect {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
                .download-btn {{ 
                    background: #007bff; 
                    color: white; 
                    padding: 10px 20px; 
                    border: none; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                .download-btn:hover {{ background: #0056b3; }}
                .sentiment-breakdown {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #007bff;
                }}
                .sentiment-stats {{
                    display: flex;
                    justify-content: space-around;
                    margin-top: 10px;
                }}
                .sentiment-stat {{
                    text-align: center;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .sentiment-stat.positive {{ background: #d4edda; }}
                .sentiment-stat.negative {{ background: #f8d7da; }}
                .sentiment-stat.neutral {{ background: #fff3cd; }}
                @media print {{
                    .download-btn {{ display: none; }}
                }}
            </style>
        </head>
        <body>
            <button class="download-btn" onclick="window.print()">📄 Print/Download PDF</button>
            
            <div class="header">
                <h1>Theme Report: {theme_info.get('name', f'Themes {theme_id}')}</h1>
                <p><strong>Summary:</strong> {theme_info.get('summary', 'No summary available')}</p>
                <p><strong>Total Tweets:</strong> {total_tweets}</p>
                <p><strong>Date Range:</strong> {start or 'All time'} to {end or 'All time'}</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="sentiment-breakdown">
                <h3>Sentiment Analysis</h3>
                <div class="sentiment-stats">
                    <div class="sentiment-stat positive">
                        <strong>Positive</strong><br>
                        {positive_count} tweets<br>
                        <strong>{positive_pct}%</strong>
                    </div>
                    <div class="sentiment-stat negative">
                        <strong>Negative</strong><br>
                        {negative_count} tweets<br>
                        <strong>{negative_pct}%</strong>
                    </div>
                    <div class="sentiment-stat neutral">
                        <strong>Neutral</strong><br>
                        {neutral_count} tweets<br>
                        <strong>{neutral_pct}%</strong>
                    </div>
                </div>
            </div>
            
            <h2>Tweets Analysis</h2>
            <p><em>Showing first {min(limit, len(theme_tweets_display))} tweets out of {total_tweets} total tweets</em></p>
        """
        
        # Add tweets (using limited display dataset)
        for idx, tweet in theme_tweets_display.iterrows():
            sentiment = tweet.get('sentiment_label', 'neutral')
            aspect = tweet.get('aspect_dominant', 'unknown')
            tweet_text = tweet.get('text', tweet.get('text_clean', tweet.get('clean_tweet', 'No text available')))
            tweet_date = tweet.get('createdat', tweet.get('date', 'Unknown date'))
            
            html_content += f"""
            <div class="tweet {sentiment}">
                <div style="margin-bottom: 10px;">
                    <span class="sentiment {sentiment}">{sentiment.title()}</span>
                    <span class="aspect">{aspect}</span>
                    <span class="date">{tweet_date}</span>
                </div>
                <p>{tweet_text}</p>
            </div>
            """
        
        html_content += """
            </body>
        </html>
        """
        
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=theme_{theme_id}_report_{start or 'all'}_to_{end or 'all'}.html"}
        )
        
    except Exception as e:
        return {"error": f"Failed to generate report: {str(e)}"}
    else:  # xlsx
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Basic data without themes
            columns = ["date", "createdat", "text", "text_clean", "sentiment_label", "aspect_dominant"]
            available_columns = [col for col in columns if col in sub.columns]
            sub[available_columns].to_excel(writer, sheet_name='Raw Data', index=False)
        
        excel_content = excel_buffer.getvalue()
        
        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=theme_report_{start}_to_{end}.xlsx"}
        )

