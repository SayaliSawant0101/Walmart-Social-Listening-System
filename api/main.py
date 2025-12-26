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
from io import StringIO, BytesIO

DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data")

# LLM summaries (exec summary + structured brief)
from src.llm.summary import build_executive_summary, summarize_tweets

# --- Load the repo-root .env no matter where Uvicorn is started from ---
ROOT_DOTENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_DOTENV)

def _read_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "") or ""
    return key.strip().strip('"').strip("'")

# ---- Theme computation (Stage 3 dynamic) ----
# requires src/features/themes.py with compute_themes_payload()
from src.features.themes import compute_themes_payload

# Load raw tweets data for theme generation
RAW_TWEETS_PATH = f"{DATA_DIR}/tweets_stage0_raw.parquet"
RAW_TWEETS_PATH_COMP = f"{DATA_DIR}/tweets_stage0_raw_comp.parquet"

# ------------ Paths ------------
SENTI_PATH = f"{DATA_DIR}/tweets_stage1_sentiment.parquet"
SENTI_PATH_COMP = f"{DATA_DIR}/tweets_stage1_sentiment_comp.parquet"  # Competitor (Costco) data
ASPECT_PATH = f"{DATA_DIR}/tweets_stage2_aspects.parquet"
STAGE3_PATH = f"{DATA_DIR}/tweets_stage3_aspect_sentiment.parquet"  # optional cache (no dates)
STAGE3_THEMES_PARQUET = f"{DATA_DIR}/tweets_stage3_themes.parquet"  # written by /themes

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

# ------------ Load Sentiment (Stage 1) ------------
if not os.path.exists(SENTI_PATH):
    raise FileNotFoundError(f"Missing parquet: {SENTI_PATH}. Run Stage 1 first.")

df = pd.read_parquet(SENTI_PATH)
_sent_date_col = _detect_date_col(df)
df = _normalize_dates(df, _sent_date_col)
SENT_MIN_DATE = df["date_only"].min()
SENT_MAX_DATE = df["date_only"].max()

# ------------ Load Competitor Sentiment (Stage 1) ------------
df_comp = None
SENT_COMP_MIN_DATE = None
SENT_COMP_MAX_DATE = None
if os.path.exists(SENTI_PATH_COMP):
    try:
        print(f"[API] Loading competitor data from {SENTI_PATH_COMP}")
        df_comp = pd.read_parquet(SENTI_PATH_COMP)
        print(f"[API] Loaded {len(df_comp)} competitor rows")
        _sent_comp_date_col = _detect_date_col(df_comp)
        df_comp = _normalize_dates(df_comp, _sent_comp_date_col)
        SENT_COMP_MIN_DATE = df_comp["date_only"].min()
        SENT_COMP_MAX_DATE = df_comp["date_only"].max()
        print(f"[API] Competitor date range: {SENT_COMP_MIN_DATE} to {SENT_COMP_MAX_DATE}")
    except Exception as e:
        print(f"[API] Error loading competitor data: {e}")
        import traceback
        traceback.print_exc()
        df_comp = None
else:
    print(f"[API] Competitor data file not found: {SENTI_PATH_COMP}")

# ------------ Load Aspects (Stage 2) ------------
ASPECTS = ["pricing", "delivery", "returns", "staff", "app/ux"]
ASPECT_PATH_COMP = f"{DATA_DIR}/tweets_stage2_aspects_comp.parquet"  # Competitor aspects

if os.path.exists(ASPECT_PATH):
    adf = pd.read_parquet(ASPECT_PATH)
    _asp_date_col = _detect_date_col(adf)
    adf = _normalize_dates(adf, _asp_date_col)
    ASPECT_MIN_DATE = adf["date_only"].min()
    ASPECT_MAX_DATE = adf["date_only"].max()
else:
    adf = pd.DataFrame(columns=["date_only", "aspect_dominant", "sentiment_label"])
    ASPECT_MIN_DATE = SENT_MIN_DATE
    ASPECT_MAX_DATE = SENT_MAX_DATE

# ------------ Load Competitor Aspects (Stage 2) ------------
adf_comp = None
ASPECT_COMP_MIN_DATE = None
ASPECT_COMP_MAX_DATE = None
if os.path.exists(ASPECT_PATH_COMP):
    try:
        print(f"[API] Loading competitor aspects from {ASPECT_PATH_COMP}")
        adf_comp = pd.read_parquet(ASPECT_PATH_COMP)
        _asp_comp_date_col = _detect_date_col(adf_comp)
        adf_comp = _normalize_dates(adf_comp, _asp_comp_date_col)
        ASPECT_COMP_MIN_DATE = adf_comp["date_only"].min()
        ASPECT_COMP_MAX_DATE = adf_comp["date_only"].max()
        print(f"[API] Loaded {len(adf_comp)} competitor aspect rows")
    except Exception as e:
        print(f"[API] Error loading competitor aspects: {e}")
        adf_comp = None
else:
    print(f"[API] Competitor aspect file not found: {ASPECT_PATH_COMP}")

# Optional cache without dates (Stage 3)
stage3_df = None
if os.path.exists(STAGE3_PATH):
    try:
        stage3_df = pd.read_parquet(STAGE3_PATH)
        for c in ["aspect_dominant", "positive", "neutral", "negative"]:
            assert c in stage3_df.columns
    except Exception:
        stage3_df = None

# ---------- Simple in-process cache for /themes ----------
_THEMES_CACHE: dict[Tuple[Optional[str], Optional[str], int, str], dict] = {}

# ------------ Routes ------------
@app.get("/")
def health():
    return {
        "message": "✅ Walmart Sentiment API is running!",
        "date_range": {"min": str(SENT_MIN_DATE), "max": str(SENT_MAX_DATE)},
        "aspect_date_range": {"min": str(ASPECT_MIN_DATE), "max": str(ASPECT_MAX_DATE)},
        "has_aspects": bool(len(adf) > 0),
        "has_stage3_cache": bool(stage3_df is not None),
        "has_competitor_data": bool(df_comp is not None),
        "competitor_date_range": {"min": str(SENT_COMP_MIN_DATE) if SENT_COMP_MIN_DATE else None, "max": str(SENT_COMP_MAX_DATE) if SENT_COMP_MAX_DATE else None},
        "env_loaded": os.path.exists(ROOT_DOTENV),
        "has_openai_key": bool(_read_openai_key()),
    }

# --- Sentiment ---
@app.get("/sentiment/summary")
def sentiment_summary(
    start: date = Query(default=SENT_MIN_DATE),
    end:   date = Query(default=SENT_MAX_DATE),
):
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

    daily = (
        sub.groupby(["date_only", "sentiment_label"])
           .size().reset_index(name="count")
           .pivot(index="date_only", columns="sentiment_label", values="count")
           .fillna(0)
    )
    daily = (daily.div(daily.sum(axis=1), axis=0) * 100).reset_index()

    for c in ["positive", "neutral", "negative"]:
        if c not in daily.columns:
            daily[c] = 0.0

    trend = [
        {"date": str(r["date_only"]),
         "positive": float(r["positive"]),
         "neutral":  float(r["neutral"]),
         "negative": float(r["negative"])}
        for _, r in daily.sort_values("date_only").iterrows()
    ]
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

    daily = (
        sub.groupby(["date_only", "sentiment_label"])
           .size().reset_index(name="count")
           .pivot(index="date_only", columns="sentiment_label", values="count")
           .fillna(0)
    )
    daily = (daily.div(daily.sum(axis=1), axis=0) * 100).reset_index()

    for c in ["positive", "neutral", "negative"]:
        if c not in daily.columns:
            daily[c] = 0.0

    trend = [
        {"date": str(r["date_only"]),
         "positive": float(r["positive"]),
         "neutral":  float(r["neutral"]),
         "negative": float(r["negative"])}
        for _, r in daily.sort_values("date_only").iterrows()
    ]
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
        
        # Check if raw tweets file exists
        if not os.path.exists(RAW_TWEETS_PATH):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Raw tweets file not found: {RAW_TWEETS_PATH}",
                    "hint": "Ensure the data file exists and the path is correct.",
                },
            )
        
        # Load raw tweets data for theme generation
        print(f"[/themes] Loading raw tweets from {RAW_TWEETS_PATH}")
        df = pd.read_parquet(RAW_TWEETS_PATH)
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
        if os.path.exists(RAW_TWEETS_PATH_COMP):
            print(f"[/themes/competitor] Loading raw tweets from {RAW_TWEETS_PATH_COMP}")
            try:
                df = pd.read_parquet(RAW_TWEETS_PATH_COMP)
                print(f"[/themes/competitor] Loaded {len(df)} tweets from raw file")
            except Exception as e:
                error_msg = f"Failed to load competitor raw data file: {str(e)}"
                print(f"[/themes/competitor] ERROR: {error_msg}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": error_msg,
                        "hint": "The competitor data file exists but could not be read. Please check if the file is corrupted.",
                        "file_path": RAW_TWEETS_PATH_COMP,
                    },
                )
        elif os.path.exists(SENTI_PATH_COMP):
            # Fallback: use sentiment file if raw file doesn't exist
            print(f"[/themes/competitor] Raw file not found, using sentiment file: {SENTI_PATH_COMP}")
            try:
                df = pd.read_parquet(SENTI_PATH_COMP)
                print(f"[/themes/competitor] Loaded {len(df)} tweets from sentiment file")
            except Exception as e:
                error_msg = f"Failed to load competitor sentiment data file: {str(e)}"
                print(f"[/themes/competitor] ERROR: {error_msg}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": error_msg,
                        "hint": "The competitor sentiment data file exists but could not be read.",
                        "file_path": SENTI_PATH_COMP,
                    },
                )
        else:
            error_msg = f"Competitor data file not found"
            print(f"[/themes/competitor] ERROR: {error_msg}")
            print(f"[/themes/competitor] Checked: {RAW_TWEETS_PATH_COMP}")
            print(f"[/themes/competitor] Checked: {SENTI_PATH_COMP}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": error_msg,
                    "hint": "The competitor (Costco) data file is missing. Please ensure either tweets_stage0_raw_comp.parquet or tweets_stage1_sentiment_comp.parquet exists in the data/ directory.",
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
    if not os.path.exists(STAGE3_THEMES_PARQUET):
        return {"items": [], "note": "Stage-3 parquet not found. Call /themes first to generate."}

    df3 = pd.read_parquet(STAGE3_THEMES_PARQUET)
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
        themes_df = pd.read_parquet(STAGE3_THEMES_PARQUET)
        
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
                themes_df = pd.read_parquet("data/tweets_stage3_themes.parquet")
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

