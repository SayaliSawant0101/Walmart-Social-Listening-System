// frontend/src/api.js
import axios from "axios";

// Netlify/Vite env (build-time)
const RAW =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE ||
  "";

// Hard fallback for PROD so it never hits localhost
const RAILWAY_FALLBACK =
  "https://walmart-social-listening-system-production.up.railway.app";

const BASE_URL = (
  RAW
    ? RAW
    : import.meta.env.DEV
    ? "http://localhost:8000"
    : RAILWAY_FALLBACK
).replace(/\/+$/, "");

// DEBUG (keep until it’s fixed)
console.log("[api] MODE:", import.meta.env.MODE);
console.log("[api] VITE_API_BASE_URL:", import.meta.env.VITE_API_BASE_URL);
console.log("[api] BASE_URL:", BASE_URL);

const API = axios.create({ baseURL: BASE_URL, timeout: 120000 }); // 2 minutes for S3 reads
const LONG_API = axios.create({ baseURL: BASE_URL, timeout: 600000 });

// --- Meta ---
export async function getMeta() {
  const { data } = await API.get("/");
  return data?.date_range || null;
}

// --- AI / Insights ---
export async function getStructuredBrief(start, end, keyword = null, sample_size = 80) {
  const params = { start, end, sample_size };
  if (keyword) params.keyword = keyword;
  const { data } = await API.get("/structured-brief", { params });
  return data;
}

export async function getExecutiveSummary(start, end, sample_per_sentiment = 250) {
  const { data } = await API.get("/executive-summary", {
    params: { start, end, sample_per_sentiment },
  });
  return data;
}

// --- Sentiment ---
export async function getSummary(start, end) {
  const { data } = await API.get("/sentiment/summary", { params: { start, end } });
  return data;
}

export async function getTrend(start, end, period = "daily", offset = 0, limit = 50) {
  const { data } = await API.get("/sentiment/trend", {
    params: { start, end, period, offset, limit },
  });
  return data;
}

// --- Competitor ---
export async function getCompetitorSummary(start, end) {
  const { data } = await API.get("/sentiment/competitor/summary", { params: { start, end } });
  return data;
}

export async function getCompetitorTrend(start, end, period = "daily", offset = 0, limit = 50) {
  const { data } = await API.get("/sentiment/competitor/trend", {
    params: { start, end, period, offset, limit },
  });
  return data;
}

// --- Aspects ---
export async function getAspectSummary(start, end, asPercent = false) {
  const { data } = await API.get("/aspects/summary", {
    params: { start, end, as_percent: asPercent },
  });
  return data;
}

export async function getAspectAvgScores(start, end) {
  const { data } = await API.get("/aspects/avg-scores", { params: { start, end } });
  return data;
}

export async function getAspectSentimentSplit(start, end, asPercent = false, includeOthers = false) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

export async function getCompetitorAspectSentimentSplit(start, end, asPercent = false, includeOthers = false) {
  const { data } = await API.get("/aspects/competitor/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

export async function getSampleTweets(start, end, aspect, sentiment, limit = 10) {
  const { data } = await API.get("/tweets/sample", {
    params: { start, end, aspect, sentiment, limit },
  });
  return data.tweets || [];
}

// --- Themes ---
export async function fetchThemes({
  start = null,
  end = null,
  n_clusters = null,
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
  merge_similar = true,
  force_refresh = false,
} = {}) {
  const params = { merge_similar, force_refresh };
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes", { params });
  return data;
}

export async function fetchThemesCompetitor({
  start = null,
  end = null,
  n_clusters = null,
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
  merge_similar = true,
  force_refresh = false,
} = {}) {
  const params = { merge_similar, force_refresh };
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes/competitor", { params });
  return data;
}
