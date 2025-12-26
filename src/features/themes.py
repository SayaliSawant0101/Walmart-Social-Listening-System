# src/features/themes.py 
from __future__ import annotations
import os, json, re, traceback, time, sys
from typing import Dict, List, Optional
import pandas as pd

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # If stdout/stderr don't have a buffer, skip the fix
        pass

FORCE_TFIDF = os.getenv("THEMES_EMB_BACKEND", "").lower() == "tfidf"

_STOP = {
    "walmart","costco","rt","amp","https","http","co","www","com","org","net",
    "user","users","you","your","yours","u","ur","me","we","us","they","them",
    "im","ive","dont","didnt","cant","couldnt","wont","wouldnt","shouldnt",
    "like","just","get","got","one","two","three","also","still","even",
    "going","go","gotta","gonna","really","please","thanks","thank","help",
    "hey","hi","hello","ok","okay","any","every","everyone","someone","anyone",
    "today","yesterday","tomorrow","now","time","back","make","made","see",
    "store","stores","shop","shopping","customer","customers","people",
    "good","bad","great","best","worst","better","worse",
    "buy","bought","purchase","purchased","sale","sales",
    "app","apps","site","website","httpst","httpsco","tco",
    "the","and","for","are","but","not","this","that","with","from","have",
    "has","had","was","were","been","being","will","would","could","should",
    "may","might","must","can","cannot","don","does","did","do","doesnt",
    "isnt","wasnt","werent","havent","hasnt","hadnt","wont","wouldnt","couldnt",
    "shouldnt","is","am","are","was","were","be","been","being","a","an","as",
    "at","by","in","on","to","of","it","its","or","if","so","than","then","there",
    "their","they","these","those","what","when","where","which","who","why"
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
        top_idx = scores.argsort()[::-1][:15]  # Get more candidates
        # Filter out stop words and very short terms
        toks = []
        for f in feats[top_idx]:
            # Check if any word in the feature is a stop word
            words = f.split()
            if words and words[0] not in _STOP and len(words[0]) > 2:
                # For multi-word phrases, check all words
                if all(w not in _STOP and len(w) > 2 for w in words):
                    toks.append(f)
        tfidf_kws[int(tid)] = toks[:8]  # Return top 8 meaningful keywords

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
        openai_api_key = openai_api_key.strip()
        if openai_api_key and len(openai_api_key) > 10:  # Basic validation
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                print(f"[Themes] ✅ OpenAI client initialized successfully (key length: {len(openai_api_key)})")
            except Exception as e:
                print(f"[Themes] ❌ OpenAI client init error: {e}")
                client = None
        else:
            print(f"[Themes] ⚠️  OpenAI API key provided but appears invalid (length: {len(openai_api_key) if openai_api_key else 0})")
    else:
        print("[Themes] ⚠️  No OpenAI API key provided. Using fallback mode for theme names and summaries.")

    theme_names: Dict[int, str] = {}
    summaries: Dict[int, str] = {}
    ai_used_for_any_theme = False  # Track if AI was actually used for any theme

    TITLE_SYSTEM = "You are an expert at analyzing customer feedback and creating concise, meaningful theme names for retail customer feedback. Generate professional, specific theme names that accurately represent what customers are discussing. The theme name should be descriptive and capture the main topic or concern."
    TITLE_USER_TMPL = (
        "Based on the following keywords extracted from customer tweets, generate a concise and meaningful theme name (2-5 words) that captures what customers are discussing. "
        "The theme name should be:\n"
        "- Specific and descriptive (e.g., 'Product Availability Issues', 'Delivery Experience', 'Customer Service Quality')\n"
        "- Professional and clear\n"
        "- Avoid generic words, brand names, or common stop words\n"
        "- Focus on the actual topic or concern being discussed\n\n"
        "Keywords from customer tweets: {kws}\n\n"
        "Return ONLY the theme name, nothing else. Do not include quotes, colons, or additional text."
    )

    SUMMARY_SYSTEM = (
        "You are a retail insights analyst specializing in customer feedback analysis. Write clear, informative summaries that describe what customers are discussing in each theme. "
        "Your summaries should be 2-3 sentences that provide specific context and insight into the customer conversations. Be descriptive and focus on what customers are actually talking about."
    )
    SUMMARY_USER_TMPL = (
        "Theme name: {title}\n\n"
        "Top keywords from customer tweets: {kws}\n\n"
        "Sample customer tweets from this theme:\n{examples}\n\n"
        "Write a 2-3 sentence summary that describes what customers are discussing in this theme. "
        "Be specific about the topics, concerns, or experiences customers are sharing. "
        "Describe the actual content of the conversations, not generic statements. "
        "Do not mention sentiment counts, percentages, or suggest actions. "
        "Focus on what customers are actually saying based on the keywords and sample tweets provided."
    )

    for tid in sorted(df["theme"].unique().astype(int)):
        sub = df[df["theme"] == tid]
        kws_prompt = tfidf_kws.get(tid) or _top_keywords(sub[text_col].tolist(), 10)
        # Filter out stop words and very short keywords
        meaningful_kws = [k for k in kws_prompt if k not in _STOP and len(k) > 2]
        # Use top 5 meaningful keywords for better context
        kws_prompt = meaningful_kws[:5] if meaningful_kws else kws_prompt[:3]
        # Create a better fallback title from meaningful keywords
        if meaningful_kws:
            # Capitalize first letter of each keyword and join
            fallback_title = " ".join([kw.capitalize() for kw in meaningful_kws[:3]])
        else:
            # If no meaningful keywords, try to extract from actual tweet text
            sample_texts = sub[text_col].astype(str).head(5).tolist()
            # Extract meaningful words from sample texts
            all_words = []
            for text in sample_texts:
                words = _normalize(text).split()
                meaningful = [w for w in words if w not in _STOP and len(w) > 3]
                all_words.extend(meaningful[:3])
            from collections import Counter
            if all_words:
                top_words = [w.capitalize() for w, _ in Counter(all_words).most_common(2)]
                fallback_title = " ".join(top_words) if top_words else f"Theme {tid}"
            else:
                fallback_title = f"Theme {tid}"

        title = fallback_title
        if client and kws_prompt:
            # Only call AI if we have meaningful keywords
            if not meaningful_kws or len(meaningful_kws) < 2:
                print(f"[Theme {tid}] Skipping AI - insufficient meaningful keywords (have {len(meaningful_kws) if meaningful_kws else 0}, need 2+)")
                print(f"[Theme {tid}] Keywords: {kws_prompt[:5]}")
            else:
                # Retry logic for OpenAI API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        keywords_str = ", ".join(meaningful_kws[:8])
                        print(f"[Theme {tid}] Calling OpenAI for title with keywords: {keywords_str[:100]}...")
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role":"system","content":TITLE_SYSTEM},
                                {"role":"user","content":TITLE_USER_TMPL.format(kws=keywords_str)}
                            ],
                            temperature=0.3,
                            max_tokens=40,
                            timeout=25
                        )
                        t = (resp.choices[0].message.content or "").strip().strip('"').strip("'").strip(".")
                        # Remove common prefixes/suffixes that AI might add
                        t = t.replace("Theme:", "").replace("Title:", "").strip()
                        if t and len(t) > 3 and len(t) < 60:  # Ensure we got a meaningful title
                            title = t
                            ai_used_for_any_theme = True  # Mark that AI was successfully used
                            # Safely print Unicode title
                            safe_title = title.encode('ascii', 'replace').decode('ascii')
                            print(f"[Theme {tid}] AI generated title: {safe_title}")
                            break  # Success, exit retry loop
                        elif attempt == max_retries - 1:
                            print(f"[Theme {tid}] Generated title invalid (len={len(t) if t else 0}), using fallback: {fallback_title}")
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 1.5
                            print(f"[Theme {tid}] OpenAI error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"[Theme {tid}] OpenAI failed after {max_retries} attempts: {str(e)[:100]}. Using fallback: {fallback_title}")
                            # Use fallback title

        theme_names[tid] = title

        p = pos_counts.get(tid, 0)
        n = neg_counts.get(tid, 0)
        neu = neu_counts.get(tid, 0)

        # Create a better fallback summary using actual tweet content
        sample_tweets = sub[text_col].astype(str).head(3).tolist()
        if sample_tweets and meaningful_kws:
            # Create a more descriptive fallback summary
            kw_str = ", ".join(meaningful_kws[:4])
            base_summary = (
                f"Customers are discussing topics related to {kw_str}. "
                f"This theme captures conversations about these key areas based on customer feedback and social media discussions."
            )
        else:
            base_summary = (
                f"This theme represents a collection of customer discussions and feedback. "
                f"Analysis of the tweets in this theme reveals patterns in customer conversations and experiences."
            )

        if client and meaningful_kws and len(meaningful_kws) >= 2:
            ex = [str(x) for x in sub[text_col].astype(str).head(3).tolist()]  # Get 3 examples for better context
            examples_text = "\n".join([f"- {e[:200]}" for e in ex if e and len(e.strip()) > 10])  # Limit each example to 200 chars, filter empty
            if not examples_text:
                examples_text = "Sample tweets from this theme are available."
            
            keywords_str = ", ".join(meaningful_kws[:8])
            # Retry logic for OpenAI API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"[Theme {tid}] Calling OpenAI for summary...")
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role":"system","content":SUMMARY_SYSTEM},
                            {"role":"user","content":SUMMARY_USER_TMPL.format(
                                title=title, kws=keywords_str, examples=examples_text
                            )}
                        ],
                        temperature=0.4,
                        max_tokens=180,  # Increased for better summaries
                        timeout=25
                    )
                    s = (resp.choices[0].message.content or "").strip()
                    # Remove common boilerplate phrases
                    s = s.replace("This theme focuses on", "").replace("Customers are discussing", "").strip()
                    if s and len(s) > 30:  # Ensure we got a meaningful summary
                        summaries[tid] = s
                        ai_used_for_any_theme = True  # Mark that AI was successfully used
                        print(f"[Theme {tid}] AI generated summary (len={len(s)})")
                        break  # Success, exit retry loop
                    elif attempt == max_retries - 1:
                        print(f"[Theme {tid}] Generated summary too short (len={len(s) if s else 0}), using fallback.")
                        summaries[tid] = base_summary
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1.5
                        print(f"[Theme {tid}] OpenAI summary error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[Theme {tid}] OpenAI summary failed after {max_retries} attempts: {str(e)[:100]}. Using fallback.")
                        summaries[tid] = base_summary
        else:
            if not client:
                print(f"[Theme {tid}] No OpenAI client - using fallback summary")
            elif not meaningful_kws:
                print(f"[Theme {tid}] No meaningful keywords - using fallback summary")
            summaries[tid] = base_summary

    with open("data/theme_names.json", "w", encoding="utf-8") as f:
        json.dump({int(k): v for k,v in theme_names.items()}, f, ensure_ascii=False, indent=2)
    with open("data/theme_summaries.json", "w", encoding="utf-8") as f:
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

    # Only mark as using LLM if AI was actually called and succeeded for at least one theme
    used_llm = ai_used_for_any_theme
    if client and not ai_used_for_any_theme:
        print(f"[Themes] ⚠️  OpenAI client available but AI was not used (likely insufficient keywords or API errors). Using fallback mode.")
    return {"updated_at": pd.Timestamp.utcnow().isoformat(), "themes": final_themes, "used_llm": used_llm}