// frontend/src/api.js
import axios from "axios";

/**
 * Base URL resolution:
 * 1) Use Netlify/Vite env var if provided: VITE_API_BASE_URL
 * 2) If running locally (Vite dev), use localhost backend
 * 3) If deployed and env var missing, fall back to Railway backend (prevents 127.0.0.1 calls)
 *
 * IMPORTANT:
 * - In Netlify, set: VITE_API_BASE_URL = https://walmart-social-listening-system-production.up.railway.app
 */
const RAW_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE || // support older naming if you used it earlier
  "";

// Detect if the frontend is being served on localhost (dev)
const IS_BROWSER = typeof window !== "undefined";
const IS_LOCALHOST =
  IS_BROWSER &&
  (window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1");

// Final default if prod env var is missing
const FALLBACK_RAILWAY =
  "https://walmart-social-listening-system-production.up.railway.app";

const BASE_URL = (RAW_BASE_URL
  ? RAW_BASE_URL
  : import.meta.env.DEV || IS_LOCALHOST
  ? "http://localhost:8000"
  : FALLBACK_RAILWAY
).replace(/\/+$/, "");

// Optional: log once (helps debugging)
if (import.meta.env.DEV) {
  // eslint-disable-next-line no-console
  console.log("[api.js] BASE_URL =", BASE_URL);
}

// Main API instance
const API = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
});

// Special API instance for long-running operations like theme generation
const LONG_API = axios.create({
  baseURL: BASE_URL,
  timeout: 600000, // 10 minutes
});

// --- Sentiment ---
export async function getMeta() {
  const { data } = await API.get("/");
  return data?.date_range || null;
}

export async function getSummary(start, end) {
  const { data } = await API.get("/sentiment/summary", {
    params: { start, end },
  });
  return data;
}

export async function getTrend(start, end, period = "daily", offset = 0, limit = 50) {
  const { data } = await API.get("/sentiment/trend", {
    params: { start, end, period, offset, limit },
  });
  return data;
}

// --- Competitor Sentiment (Costco) ---
export async function getCompetitorSummary(start, end) {
  const { data } = await API.get("/sentiment/competitor/summary", {
    params: { start, end },
  });
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
  const { data } = await API.get("/aspects/avg-scores", {
    params: { start, end },
  });
  return data;
}

// --- Aspect × Sentiment (Stacked Bar) ---
export async function getAspectSentimentSplit(
  start,
  end,
  asPercent = false,
  includeOthers = false
) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

// --- Competitor Aspect × Sentiment (Stacked Bar) ---
export async function getCompetitorAspectSentimentSplit(
  start,
  end,
  asPercent = false,
  includeOthers = false
) {
  const { data } = await API.get("/aspects/competitor/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

// Get raw aspect data for calculating "Others" category
export async function getRawAspectData(start, end) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: false },
  });
  return data;
}

// Get sample tweets for specific aspect and sentiment
export async function getSampleTweets(start, end, aspect, sentiment, limit = 10) {
  const { data } = await API.get("/tweets/sample", {
    params: { start, end, aspect, sentiment, limit },
  });
  return data.tweets || [];
}

// --- AI Insights ---
export async function getExecutiveSummary(start, end, sample_per_sentiment = 250) {
  const { data } = await LONG_API.get("/executive-summary", {
    params: { start, end, sample_per_sentiment },
  });
  return data;
}

export async function getStructuredBrief(start, end, keyword = null, sample_size = 80) {
  const { data } = await LONG_API.get("/structured-brief", {
    params: { start, end, keyword, sample_size },
  });
  return data;
}

// --- Themes (dynamic clustering + summaries) ---
export async function fetchThemes({
  start = null,
  end = null,
  n_clusters = null, // Auto-detect if null
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
} = {}) {
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes", { params });
  return data;
}

// --- Themes for Competitor (Costco) ---
export async function fetchThemesCompetitor({
  start = null,
  end = null,
  n_clusters = null,
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
} = {}) {
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes/competitor", { params });
  return data;
}

// --- Raw Data Downloads ---
export async function downloadRawTweets(start, end, format = "csv") {
  const response = await API.get("/tweets/raw", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `raw_tweets_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadSentimentReport(start, end, format = "pdf") {
  const response = await API.get("/reports/sentiment", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sentiment_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadAspectReport(start, end, format = "pdf") {
  const response = await API.get("/reports/aspects", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `aspect_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadThemeReport(start, end, format = "pdf") {
  const response = await API.get("/reports/themes", {
    params: { start, end, format },
    responseType: "blob",
  });

  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `theme_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadThemeTweetsReport(themeId, start, end) {
  const response = await API.get(`/reports/theme/${themeId}`, {
    params: { start, end, limit: 200 },
    responseType: "blob",
  });

  const blob = new Blob([response.data], { type: "text/html" });
  const url = window.URL.createObjectURL(blob);

  // Open in new tab instead of downloading
  window.open(url, "_blank");

  setTimeout(() => {
    window.URL.revokeObjectURL(url);
  }, 10000);
}
