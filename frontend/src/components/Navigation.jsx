import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useDate } from "../contexts/DateContext";

// --- helpers ---
function iso(x) {
  if (!x) return "";
  if (typeof x === "string" && /^\d{4}-\d{2}-\d{2}$/.test(x)) {
    return x;
  }
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function Navigation() {
  const location = useLocation();
  const { start, end, meta, setStart, setEnd } = useDate();

  const navItems = [
    { path: "/", label: "Dashboard", description: "Overview & Sentiment Trends" },
    { path: "/theme-analysis", label: "Theme Analysis", description: "AI-Generated Theme Clusters" },
    { path: "/ai-insights", label: "AI Insights", description: "Executive Summary & Briefs" },
  ];

  return (
    <>
      {/* Top Navigation - Dark Theme */}
      <nav className="bg-slate-800/95 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 py-3">
          <div className="flex items-center space-x-4">
            <div className="w-11 h-11 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-2xl flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-lg">W</span>
            </div>

            {/* TITLE ROW */}
            <div className="flex-1">
              <div className="flex items-center justify-between gap-4">
                {/* Left: Product title */}
                <h1 className="text-2xl font-semibold tracking-tight text-white whitespace-nowrap">
                  AI-Powered Social Intelligence
                </h1>

                {/* Right: Ownership + links */}
                <div className="flex items-center gap-2 text-sm font-medium tracking-tight text-slate-400 whitespace-nowrap">
                  <span className="text-slate-600">•</span>
                  <span>Built by Sayali Sawant</span>
                  <span className="text-slate-600">•</span>
                  <a
                    href="https://github.com/SayaliSawant0101/Walmart-Social-Listening-System"
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-4 text-slate-300 hover:text-emerald-400 transition-colors"
                  >
                    GitHub
                  </a>
                  <span className="text-slate-600">•</span>
                  <a
                    href="https://sayalis.org"
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-4 text-slate-300 hover:text-emerald-400 transition-colors"
                  >
                    Portfolio
                  </a>
                </div>
              </div>

              {/* Updated Subtitle */}
              <p className="text-slate-400">
                Sentiment, aspect, and theme analysis with AI-generated executive insights
              </p>
            </div>
          </div>
        </div>
      </nav>

      {/* Sidebar - Dark Theme */}
      <div className="fixed left-0 top-20 h-screen w-48 bg-slate-800/95 backdrop-blur-md border-r border-slate-700 z-40">
        <div className="p-3 space-y-2">
          {navItems.map((item) => (
              <Link
              key={item.path}
              to={item.path}
              className={`flex items-center space-x-2 px-3 py-2 rounded-xl transition-all duration-300 ${
                location.pathname === item.path
                  ? "bg-emerald-900/30 text-emerald-400 border border-emerald-700 shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-700"
              }`}
            >
              <div className="w-6 h-6 bg-slate-700 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-xs text-slate-400">•</span>
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-semibold text-xs truncate">{item.label}</div>
                <div className="text-[10px] text-slate-500 line-clamp-2">{item.description}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </>
  );
}
