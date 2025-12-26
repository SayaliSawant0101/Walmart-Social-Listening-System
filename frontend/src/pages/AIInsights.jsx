// frontend/src/pages/AIInsights.jsx
import React, { useState } from "react";
import { useDate } from "../contexts/DateContext";

// Use the same env var pattern as api.js
const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL;
const BASE_URL = (RAW_BASE_URL || (import.meta.env.DEV ? "http://localhost:8000" : "")).replace(
  /\/+$/,
  ""
);

export default function AIInsights() {
  const { start, end, meta, setStart, setEnd, loading: metaLoading } = useDate();

  const [briefKeyword, setBriefKeyword] = useState("");

  // Exec Summary
  const [execData, setExecData] = useState(null);
  const [execLoading, setExecLoading] = useState(false);
  const [execErr, setExecErr] = useState("");

  // Structured Brief
  const [briefData, setBriefData] = useState(null);
  const [briefLoading, setBriefLoading] = useState(false);
  const [briefErr, setBriefErr] = useState("");

  async function runExecutiveSummary() {
    if (!start || !end) return;
    try {
      setExecLoading(true);
      setExecErr("");

      const q = new URLSearchParams({
        start,
        end,
        sample_per_sentiment: String(250),
      }).toString();

      const r = await fetch(`${BASE_URL}/executive-summary?${q}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setExecData(j);
    } catch (e) {
      setExecErr(String(e));
      setExecData(null);
    } finally {
      setExecLoading(false);
    }
  }

  async function runStructuredBrief() {
    if (!start || !end) return;
    try {
      setBriefLoading(true);
      setBriefErr("");

      const paramsObj = {
        start,
        end,
        sample_size: String(80),
      };
      if (briefKeyword && briefKeyword.trim()) paramsObj.keyword = briefKeyword.trim();

      const q = new URLSearchParams(paramsObj).toString();

      const r = await fetch(`${BASE_URL}/structured-brief?${q}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setBriefData(j);
    } catch (e) {
      setBriefErr(String(e));
      setBriefData(null);
    } finally {
      setBriefLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* AI Insights Header */}
      <div className="flex items-center gap-2">
        <div className="h-2 w-2 rounded-full bg-emerald-400" />
        <h1 className="text-xl font-semibold text-white">AI Insights</h1>
      </div>
      <p className="text-slate-300 text-sm">Executive summaries and structured analysis</p>

      {/* Filter Options Card */}
      <div className="rounded-2xl bg-slate-900/40 border border-slate-700 p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Keyword */}
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full bg-emerald-400" />
            <div className="text-slate-200 font-medium">Filter Options</div>

            <input
              value={briefKeyword}
              onChange={(e) => setBriefKeyword(e.target.value)}
              placeholder="Filter by keyword..."
              className="px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-lg
                         focus:ring-2 focus:ring-emerald-400 focus:border-emerald-400
                         transition-all w-64 text-white placeholder-slate-400 text-sm"
            />
          </div>

          {/* Date */}
          <div className="flex items-center gap-2 rounded-xl bg-slate-800/40 border border-slate-700 px-3 py-2">
            <span className="text-slate-200 text-sm">
              <span className="text-emerald-400">•</span> Date:
            </span>

            <input
              type="date"
              value={start || ""}
              onChange={(e) => setStart(e.target.value)}
              className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white
                         focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all ml-0.5"
              disabled={metaLoading}
            />
            <span className="text-slate-400 text-xs">to</span>
            <input
              type="date"
              value={end || ""}
              onChange={(e) => setEnd(e.target.value)}
              className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white
                         focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all ml-0.5"
              disabled={metaLoading}
            />
          </div>
        </div>

        {/* Filter help */}
        <div className="mt-4 rounded-xl bg-slate-900/50 border border-slate-700 p-3 text-sm text-slate-300">
          <div className="font-medium text-slate-200 mb-2">How to use filter options:</div>
          <ul className="space-y-1">
            <li>• <b>Date Range:</b> Select the time period for analysis</li>
            <li>• <b>Keyword Filter:</b> Enter terms (e.g., “pricing”, “delivery”) to focus AI analysis</li>
            <li>• <b>Leave keyword empty:</b> AI analyzes all tweets for broader insights</li>
          </ul>
          <div className="mt-2 text-xs text-slate-400">
            API target: <span className="text-slate-200">{BASE_URL || "(missing VITE_API_BASE_URL)"}</span>
          </div>
        </div>
      </div>

      {/* Executive Summary */}
      <div className="rounded-2xl bg-slate-900/40 border border-slate-700 p-4">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-semibold text-white">AI Executive Summary</h2>
          <button
            onClick={runExecutiveSummary}
            disabled={execLoading || !BASE_URL}
            className="px-4 py-2 rounded-lg bg-emerald-500/90 hover:bg-emerald-500 text-slate-950 font-semibold disabled:opacity-50"
          >
            {execLoading ? "Generating..." : "Generate Summary"}
          </button>
        </div>

        {execErr && (
          <div className="mt-3 text-sm text-red-300">
            <b>Error:</b> {execErr}
          </div>
        )}

        {execData && (
          <div className="mt-4 space-y-3">
            <div className="text-xs text-slate-400">
              Mode:{" "}
              <span className="text-slate-200">
                {execData.used_llm ? "✅ AI Model Active" : "⚠️ Fallback Mode"}
              </span>
            </div>
            <div className="whitespace-pre-wrap text-slate-200">{execData.summary}</div>

            {execData.stats?.sentiment?.counts && (
              <div className="text-sm text-slate-300">
                <b>Stats:</b>{" "}
                Negative: {execData.stats.sentiment.counts.negative ?? 0} | Neutral:{" "}
                {execData.stats.sentiment.counts.neutral ?? 0} | Positive:{" "}
                {execData.stats.sentiment.counts.positive ?? 0}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Structured Brief */}
      <div className="rounded-2xl bg-slate-900/40 border border-slate-700 p-4">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-semibold text-white">Structured Brief</h2>
          <button
            onClick={runStructuredBrief}
            disabled={briefLoading || !BASE_URL}
            className="px-4 py-2 rounded-lg bg-sky-400/90 hover:bg-sky-400 text-slate-950 font-semibold disabled:opacity-50"
          >
            {briefLoading ? "Generating..." : "Generate Brief"}
          </button>
        </div>

        {briefErr && (
          <div className="mt-3 text-sm text-red-300">
            <b>Error:</b> {briefErr}
          </div>
        )}

        {briefData && (
          <div className="mt-4 space-y-4 text-slate-200">
            {briefData.structured?.executive_bullets?.length ? (
              <>
                <div>
                  <h3 className="text-slate-100 font-semibold mb-2">Executive Bullets</h3>
                  <ol className="list-decimal ml-5 space-y-1">
                    {briefData.structured.executive_bullets.map((b, i) => (
                      <li key={i}>{b}</li>
                    ))}
                  </ol>
                </div>

                {briefData.structured.themes?.length ? (
                  <div>
                    <h3 className="text-slate-100 font-semibold mb-2">Key Themes</h3>
                    <div className="flex flex-wrap gap-2">
                      {briefData.structured.themes.map((t, i) => (
                        <span
                          key={i}
                          className="px-2 py-1 rounded-lg bg-slate-800/60 border border-slate-700 text-sm"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}

                {briefData.structured.risks?.length ? (
                  <div>
                    <h3 className="text-slate-100 font-semibold mb-2">Risks</h3>
                    <ul className="list-disc ml-5 space-y-1">
                      {briefData.structured.risks.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {briefData.structured.opportunities?.length ? (
                  <div>
                    <h3 className="text-slate-100 font-semibold mb-2">Opportunities</h3>
                    <ul className="list-disc ml-5 space-y-1">
                      {briefData.structured.opportunities.map((o, i) => (
                        <li key={i}>{o}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="whitespace-pre-wrap">{briefData.executive_text || "No content available"}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
