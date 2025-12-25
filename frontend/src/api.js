// frontend/src/api.js
import axios from "axios";

// Prefer env if provided; otherwise fall back to localhost
const BASE_URL = import.meta?.env?.VITE_API_BASE || "http://localhost:8000";

// If you set up a Vite proxy, change baseURL to "/api"
const API = axios.create({
  baseURL: BASE_URL, // or "http://localhost:8000"
  timeout: 15000,
});

// Special API instance for long-running operations like theme generation
const LONG_API = axios.create({
  baseURL: BASE_URL,
  timeout: 600000, // 10 minutes for theme generation
});

// --- Sentiment ---
export async function getMeta() {
  const { data } = await API.get("/");
  // prefer aspect range if needed, but keep existing behavior
  return data?.date_range || null;
}

export async function getSummary(start, end) {
  const { data } = await API.get("/sentiment/summary", { params: { start, end } });
  return data;
}

export async function getTrend(start, end, period = "daily", offset = 0, limit = 50) {
  const { data } = await API.get("/sentiment/trend", { 
    params: { start, end, period, offset, limit } 
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

// --- Aspect × Sentiment (Stacked Bar) ---
export async function getAspectSentimentSplit(start, end, asPercent = false, includeOthers = false) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: asPercent, include_others: includeOthers },
  });
  return data;
}

// Get raw aspect data for calculating "Others" category
export async function getRawAspectData(start, end) {
  const { data } = await API.get("/aspects/sentiment-split", {
    params: { start, end, as_percent: false }
  });
  return data;
}

// Get sample tweets for specific aspect and sentiment
export async function getSampleTweets(start, end, aspect, sentiment, limit = 10) {
  const { data } = await API.get("/tweets/sample", {
    params: { start, end, aspect, sentiment, limit }
  });
  return data.tweets || [];
}

// --- Themes (dynamic clustering + summaries) ---
export async function fetchThemes({
  start = null,
  end = null,
  n_clusters = null, // Auto-detect if null
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
} = {}) {
  // Build params without null/empty values
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes", { params });
  return data; // { updated_at, themes: [{id, name, summary, tweet_count}] }
}

// --- Themes for Competitor (Costco) ---
export async function fetchThemesCompetitor({
  start = null,
  end = null,
  n_clusters = null, // Auto-detect if null
  emb_model = "sentence-transformers/all-MiniLM-L6-v2",
} = {}) {
  // Build params without null/empty values
  const params = {};
  if (start) params.start = start;
  if (end) params.end = end;
  if (n_clusters !== null) params.n_clusters = n_clusters;
  if (emb_model) params.emb_model = emb_model;

  const { data } = await LONG_API.get("/themes/competitor", { params });
  return data; // { updated_at, themes: [{id, name, summary, tweet_count}] }
}

// --- Raw Data Downloads ---
export async function downloadRawTweets(start, end, format = 'csv') {
  const response = await API.get("/tweets/raw", {
    params: { start, end, format },
    responseType: 'blob'
  });
  
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `raw_tweets_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadSentimentReport(start, end, format = 'pdf') {
  const response = await API.get("/reports/sentiment", {
    params: { start, end, format },
    responseType: 'blob'
  });
  
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `sentiment_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadAspectReport(start, end, format = 'pdf') {
  const response = await API.get("/reports/aspects", {
    params: { start, end, format },
    responseType: 'blob'
  });
  
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `aspect_report_${start}_to_${end}.${format}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

export async function downloadThemeReport(start, end, format = 'pdf') {
  const response = await API.get("/reports/themes", {
    params: { start, end, format },
    responseType: 'blob'
  });
  
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
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
    responseType: 'blob'
  });
  
  const blob = new Blob([response.data], { type: 'text/html' });
  const url = window.URL.createObjectURL(blob);
  
  // Open in new tab instead of downloading
  const newWindow = window.open(url, '_blank');
  
  // Clean up the URL after a delay to free memory
  setTimeout(() => {
    window.URL.revokeObjectURL(url);
  }, 10000); // 10 seconds delay
}
