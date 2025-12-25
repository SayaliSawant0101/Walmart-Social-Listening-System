# src/features/themes.py 
from __future__ import annotations
import os, json, re, traceback, time
from typing import Dict, List, Optional
import pandas as pd

FORCE_TFIDF = os.getenv("THEMES_EMB_BACKEND", "").lower() == "tfidf"

_STOP = {
    "walmart","rt","amp","https","http","co","www","com","org","net",
    "user","users","you","your","yours","u","ur","me","we","us","they","them",
    "im","ive","dont","didnt","cant","couldnt","wont","wouldnt","shouldnt",
    "like","just","get","got","one","two","three","also","still","even",
    "going","go","gotta","gonna","really","please","thanks","thank","help",
    "hey","hi","hello","ok","okay","any","every","everyone","someone","anyone",
    "today","yesterday","tomorrow","now","time","back","make","made","see",
    "store","stores","shop","shopping","customer","customers","people",
    "good","bad","great","best","worst","better","worse",
    "buy","bought","purchase","purchased","sale","sales",
    "app","apps","site","website","httpst","httpsco","tco"
}
_URL_MENTION_HASHTAG = re.compile(r"https?://\S+|[@#]\w+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s']")
_MULTI_SP = re.compile(r"\s+")

def _normalize(text: str) -> str:
    t = text.lower()
    t = _URL_MENTION_HASHTAG.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTI_SP.sub(" ", t).strip()
    return t

def _top_keywords(texts: List[str], top_k: int = 8) -> List[str]:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        for tok in _normalize(str(t)).split():
            if len(tok) < 3 or tok in _STOP or tok.isdigit():
                continue
            cnt[tok] += 1
    return [w for w,_ in cnt.most_common(top_k)]

def _merge_similar_themes(themes: List[dict], similarity_threshold: float = 0.7) -> List[dict]:
    """Merge themes that are semantically similar based on name and summary similarity."""
    if len(themes) <= 1:
        return themes
    
    # Enhanced similarity check based on common words and semantic meaning
    def _calculate_similarity(theme1: dict, theme2: dict) -> float:
        name1 = _normalize(theme1["name"]).split()
        name2 = _normalize(theme2["name"]).split()
        
        # Remove common stop words
        name1 = [w for w in name1 if w not in _STOP and len(w) > 2]
        name2 = [w for w in name2 if w not in _STOP and len(w) > 2]
        
        if not name1 or not name2:
            return 0.0
        
        # Calculate Jaccard similarity
        set1, set2 = set(name1), set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Additional semantic similarity checks
        semantic_similarity = 0.0
        
        # Check for semantic word pairs (synonyms/related terms)
        semantic_pairs = [
            ("availability", "stock"), ("stock", "inventory"), ("product", "item"),
            ("order", "purchase"), ("fulfillment", "delivery"), ("processing", "handling"),
            ("customer", "client"), ("service", "support"), ("issue", "problem"),
            ("concern", "issue"), ("problem", "issue"), ("complaint", "issue"),
            ("experience", "interaction"), ("humor", "funny"), ("joke", "humor"),
            ("availability", "concern"), ("stock", "concern"), ("product", "availability"),
            ("delivery", "fulfillment"), ("pricing", "tariff")
        ]
        
        for word1, word2 in semantic_pairs:
            if (word1 in set1 and word2 in set2) or (word2 in set1 and word1 in set2):
                semantic_similarity += 0.3
                break
        
        # Check for substring matches (e.g., "Product Availability" vs "Product Availability Concerns")
        name1_str = " ".join(name1)
        name2_str = " ".join(name2)
        
        if name1_str in name2_str or name2_str in name1_str:
            semantic_similarity += 0.4
        
        # Check for high word overlap (e.g., "Product Availability and Stock Issues" vs "Product Availability Concerns")
        if len(set1.intersection(set2)) >= 2:  # At least 2 common words
            semantic_similarity += 0.3
        
        # Return the maximum of Jaccard similarity and semantic similarity
        return max(jaccard_sim, semantic_similarity)
    
    merged_themes = []
    used_indices = set()
    
    for i, theme1 in enumerate(themes):
        if i in used_indices:
            continue
            
        # Find similar themes to merge
        similar_themes = [theme1]
        for j, theme2 in enumerate(themes[i+1:], i+1):
            if j in used_indices:
                continue
                
            similarity = _calculate_similarity(theme1, theme2)
            if similarity >= similarity_threshold:
                similar_themes.append(theme2)
                used_indices.add(j)
        
        # Merge similar themes
        if len(similar_themes) > 1:
            # Combine tweet counts and sentiment
            total_tweets = sum(t["tweet_count"] for t in similar_themes)
            total_positive = sum(t["positive"] for t in similar_themes)
            total_negative = sum(t["negative"] for t in similar_themes)
            total_neutral = sum(t["neutral"] for t in similar_themes)
            
            # Use the theme name with highest tweet count
            main_theme = max(similar_themes, key=lambda x: x["tweet_count"])
            
            # Combine summaries - use the best summary without merge indicators
            summaries = [t["summary"] for t in similar_themes if t["summary"]]
            combined_summary = main_theme["summary"]
            # Don't add merge indicators - just use the best summary
            
            merged_theme = {
                "id": main_theme["id"],
                "name": main_theme["name"],
                "summary": combined_summary,
                "tweet_count": total_tweets,
                "positive": total_positive,
                "negative": total_negative,
                "neutral": total_neutral,
            }
            merged_themes.append(merged_theme)
        else:
            merged_themes.append(theme1)
        
        used_indices.add(i)
    
    # Sort by tweet count again after merging
    return sorted(merged_themes, key=lambda x: x["tweet_count"], reverse=True)

# =========================
# A) Diversity-aware selection (MMR)
# =========================
def _mmr_select(themes_all: List[dict], texts_by_cluster: Dict[int, List[str]], n: int, lam: float = 0.65) -> List[dict]:
    """
    Maximal Marginal Relevance selection of N themes.
    lam balances size/coverage vs diversity: lam in [0..1]. Higher -> favors size more.
    """
    # Build a TF-IDF over cluster "documents" (join texts per cluster) for cosine similarity between themes
    joined = []
    ids = []
    for t in themes_all:
        cid = int(t["id"])
        ids.append(cid)
        joined.append(" ".join(texts_by_cluster.get(cid, [])[:400]))  # limit per-cluster for speed

    if not joined:
        return themes_all[:n]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as _sk_normalize
    import numpy as np

    vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(joined).astype("float32")
    X = _sk_normalize(X)
    # Convert sparse matrix to dense array for similarity computation
    sims_result = X @ X.T  # cosine sim (clusters x clusters)
    if hasattr(sims_result, 'toarray'):
        sims = sims_result.toarray()
    else:
        sims = sims_result  # Already dense

    # Normalize tweet_count to 0..1 for comparability
    sizes = np.array([t["tweet_count"] for t in themes_all], dtype=float)
    if sizes.max() > 0:
        sizes = sizes / sizes.max()
    else:
        sizes = np.zeros_like(sizes)

    selected = []
    selected_idx = []
    # Greedy MMR
    for _ in range(min(n, len(themes_all))):
        best_j = None
        best_score = -1e9
        for j in range(len(themes_all)):
            if j in selected_idx:
                continue
            # diversity penalty = max similarity to anything already selected
            if selected_idx:
                div_pen = max(sims[j, k] for k in selected_idx)
            else:
                div_pen = 0.0
            score = lam * sizes[j] - (1 - lam) * div_pen
            if score > best_score:
                best_score = score
                best_j = j
        selected_idx.append(best_j)
        selected.append(themes_all[best_j])
    return selected

def compute_themes_payload(
    df: Optional[pd.DataFrame] = None,
    parquet_stage2: str = "data/tweets_stage2_aspects.parquet",
    n_clusters: int = 6,  # Limited to 6 themes max
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    merge_similar: bool = True,
) -> dict:
    """Return {"updated_at": ts, "themes": [{id,name,summary,tweet_count,positive,negative,neutral}], "used_llm": bool}."""
    if df is None:
        assert os.path.exists(parquet_stage2), f"Missing {parquet_stage2}"
        df = pd.read_parquet(parquet_stage2)

    text_col = next((c for c in ["text_used","clean_tweet","text","fulltext"] if c in df.columns), None)
    if not text_col:
        raise KeyError("No text column among ['text_used','clean_tweet','text','fulltext'].")

    date_col = next((c for c in ["createdat","created_dt","created_at","tweet_date","date","dt"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

    if df.empty:
        return {"updated_at": pd.Timestamp.utcnow().isoformat(), "themes": [], "used_llm": False}

    texts = df[text_col].astype(str).tolist()

    # ---------- Optimized Processing: Sample data for faster processing ----------
    max_samples = 5000  # Limit to 3000 tweets for faster processing
    if len(df) > max_samples:
        df_sample = df.sample(n=max_samples, random_state=42)
        texts_to_process = df_sample[text_col].astype(str).tolist()
    else:
        df_sample = df
        texts_to_process = texts

    # ---------- Embeddings: ST if available, else TF-IDF ----------
    emb = None
    if not FORCE_TFIDF:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(emb_model)
            emb = model.encode(
                texts_to_process, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            emb = None

    if emb is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,2))  # Reduced features
        emb = vec.fit_transform(texts_to_process).astype("float32")
        emb = normalize(emb)

    # ---------- Clustering (Optimized) ----------
    from sklearn.cluster import KMeans
    if n_clusters is None:
        n_clusters = 6  # Default to 6 themes
    k = max(2, int(n_clusters))
    km = KMeans(n_clusters=k, random_state=42, n_init=3)  # Reduced n_init for speed
    labels = km.fit_predict(emb)
    
    # Map labels back to full dataset
    if len(df) > max_samples:
        # For sampled data, assign clusters to nearest centroids for remaining data
        remaining_texts = df[~df.index.isin(df_sample.index)][text_col].astype(str).tolist()
        if remaining_texts:
            if not FORCE_TFIDF and 'model' in locals():
                remaining_embeddings = model.encode(remaining_texts, batch_size=32, show_progress_bar=False)
            else:
                remaining_embeddings = vec.transform(remaining_texts).astype("float32")
                remaining_embeddings = normalize(remaining_embeddings)
            
            remaining_labels = km.predict(remaining_embeddings)
            
            # Combine labels
            df.loc[df_sample.index, "theme"] = labels
            df.loc[~df.index.isin(df_sample.index), "theme"] = remaining_labels
        else:
            df["theme"] = labels
    else:
        df["theme"] = labels

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/tweets_stage3_themes.parquet", index=False)

    # ---------- TF-IDF keywords per theme (Optimized) ----------
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    tfidf_kws: Dict[int, List[str]] = {}
    for tid, sub in df.groupby("theme"):
        sub_texts = sub[text_col].astype(str).tolist()
        if not sub_texts:
            tfidf_kws[int(tid)] = []
            continue
        # Sample texts if too many for faster processing
        if len(sub_texts) > 500:
            sub_texts = sub_texts[:500]
        vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))  # Reduced features
        X = vec.fit_transform(sub_texts)
        # Convert sparse matrix mean to 1D array
        # X.mean(axis=0) returns a sparse matrix, convert to dense array
        mean_result = X.mean(axis=0)
        if hasattr(mean_result, 'toarray'):
            scores = mean_result.toarray().flatten()
        elif hasattr(mean_result, 'A1'):
            # Fallback for older scipy versions
            scores = mean_result.A1
        else:
            # Already a numpy array
            scores = np.asarray(mean_result).flatten()
        feats = vec.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:6]
        toks = [f for f in feats[top_idx] if f.split()[0] not in _STOP]
        tfidf_kws[int(tid)] = toks[:6]

    # ---------- Sentiment counts ----------
    pos_counts, neg_counts, neu_counts = {}, {}, {}
    if "sentiment_label" in df.columns:
        for tid, sub in df.groupby("theme"):
            vc = sub["sentiment_label"].value_counts().to_dict()
            pos_counts[int(tid)] = int(vc.get("positive", 0))
            neg_counts[int(tid)] = int(vc.get("negative", 0))
            neu_counts[int(tid)] = int(vc.get("neutral", 0))
    else:
        for tid in df["theme"].unique():
            pos_counts[int(tid)] = 0
            neg_counts[int(tid)] = 0
            neu_counts[int(tid)] = 0

    # ---------- Name & summarize with OpenAI if key provided ----------
    client = None
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key.strip())
        except Exception as e:
            print("[OpenAI client init error]", e)
            client = None

    theme_names: Dict[int, str] = {}
    summaries: Dict[int, str] = {}

    TITLE_SYSTEM = "You name customer feedback themes for Walmart Twitter data."
    TITLE_USER_TMPL = (
        "Name this theme in 3–6 words. Be specific and professional; no filler. "
        "Avoid generic tokens like user, http, rt, amp, you, me, we. "
        "Keywords (cleaned): {kws}\n"
        "Return ONLY the title."
    )

    SUMMARY_SYSTEM = (
        "You are a retail insights analyst. Write 3-4 lines describing the theme for a Walmart stakeholder. "
        "Focus on what customers are discussing, not sentiment counts or action items. Be descriptive and informative."
    )
    SUMMARY_USER_TMPL = (
        "Theme name: {title}\n"
        "Top keywords: {kws}\n"
        "Examples (up to 2): {examples}\n"
        "Output: 3-4 lines describing what customers are discussing in this theme. Do not mention sentiment counts or suggest actions."
    )

    for tid in sorted(df["theme"].unique().astype(int)):
        sub = df[df["theme"] == tid]
        kws_prompt = tfidf_kws.get(tid) or _top_keywords(sub[text_col].tolist(), 6)
        fallback_title = ", ".join([k for k in kws_prompt if k not in _STOP][:3]) or f"Theme {tid}"

        title = fallback_title
        if client:
            # Retry logic for OpenAI API calls
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":TITLE_SYSTEM},
                            {"role":"user","content":TITLE_USER_TMPL.format(kws=kws_prompt)}
                        ],
                        temperature=0.2,
                        max_tokens=24,
                        timeout=15  # Increased timeout
                    )
                    t = (resp.choices[0].message.content or "").strip().strip('"')
                    if t:
                        title = t
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1  # Exponential backoff: 1s, 2s
                        print(f"[OpenAI title error for theme {tid}, attempt {attempt + 1}/{max_retries}]: {str(e)}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[OpenAI title error for theme {tid} after {max_retries} attempts]: {str(e)}. Using fallback title.")
                        # Use fallback title

        theme_names[tid] = title

        p = pos_counts.get(tid, 0)
        n = neg_counts.get(tid, 0)
        neu = neu_counts.get(tid, 0)

        base_summary = (
            f"{title}: This theme focuses on {', '.join(kws_prompt[:4])}. "
            f"Customers are discussing various aspects related to this topic. "
            f"The discussions cover different perspectives and experiences. "
            f"This represents an important area of customer feedback and engagement."
        )

        if client:
            ex = [str(x) for x in sub[text_col].astype(str).head(2).tolist()]  # Reduced examples
            # Retry logic for OpenAI API calls
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":SUMMARY_SYSTEM},
                            {"role":"user","content":SUMMARY_USER_TMPL.format(
                                title=title, kws=kws_prompt, examples=ex
                            )}
                        ],
                        temperature=0.3,
                        max_tokens=120,  # Increased for 3-4 lines
                        timeout=15  # Increased timeout
                    )
                    s = (resp.choices[0].message.content or "").strip()
                    summaries[tid] = s or base_summary
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1  # Exponential backoff: 1s, 2s
                        print(f"[OpenAI summary error for theme {tid}, attempt {attempt + 1}/{max_retries}]: {str(e)}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[OpenAI summary error for theme {tid} after {max_retries} attempts]: {str(e)}. Using fallback summary.")
                        summaries[tid] = base_summary
        else:
            summaries[tid] = base_summary

    with open("data/theme_names.json", "w") as f:
        json.dump({int(k): v for k,v in theme_names.items()}, f, ensure_ascii=False, indent=2)
    with open("data/theme_summaries.json", "w") as f:
        json.dump({int(k): v for k,v in summaries.items()}, f, ensure_ascii=False, indent=2)

    # ---------- Build ranked list & diversity-aware seed (A) ----------
    counts = df["theme"].value_counts().to_dict()
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # If we don't have enough clusters, generate more (keep your existing fallback)
    if len(ranked) < n_clusters:
        additional_clusters_needed = n_clusters - len(ranked)
        temp_n_clusters = n_clusters + additional_clusters_needed + 2  # Add buffer
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(emb_model)
        embeddings = model.encode(df[text_col].astype(str).tolist())
        kmeans = KMeans(n_clusters=temp_n_clusters, random_state=42, n_init=10)
        df["theme"] = kmeans.fit_predict(embeddings)
        counts = df["theme"].value_counts().to_dict()
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Build small corpus per cluster for similarity (A)
    texts_by_cluster: Dict[int, List[str]] = {}
    for tid, sub in df.groupby("theme"):
        texts_by_cluster[int(tid)] = sub[text_col].astype(str).head(300).tolist()

    # Build themes_all from ranked list (A)
    themes_all: List[dict] = []
    for k_, count in ranked:
        k = int(k_)
        themes_all.append({
            "id": k,
            "name": theme_names.get(k, f"Theme {k}"),
            "summary": summaries.get(k, ""),
            "tweet_count": int(count),
            "positive": pos_counts.get(k, 0),
            "negative": neg_counts.get(k, 0),
            "neutral": neu_counts.get(k, 0),
        })

    # Diversity-aware selection instead of raw top-N (A)
    themes = _mmr_select(themes_all, texts_by_cluster, n=n_clusters, lam=0.65)

    # Merge similar themes to avoid duplicates (existing behavior)
    if merge_similar:
        final_themes = _merge_similar_themes(themes, similarity_threshold=0.5)
        # Only return themes with actual content (tweet_count > 0)
        final_themes = [t for t in final_themes if t["tweet_count"] > 0]
        # Limit to requested number of themes, but don't add empty ones
        final_themes = final_themes[:n_clusters]
    else:
        # Only return themes with actual content
        final_themes = [t for t in themes if t["tweet_count"] > 0][:n_clusters]

    used_llm = bool(client)
    return {"updated_at": pd.Timestamp.utcnow().isoformat(), "themes": final_themes, "used_llm": used_llm}