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
    { path: "/aspect-analysis", label: "Aspect Analysis", description: "Aspect × Sentiment Breakdown" },
    { path: "/theme-analysis", label: "Theme Analysis", description: "AI-Generated Theme Clusters" },
    { path: "/ai-insights", label: "AI Insights", description: "Executive Summary & Briefs" },
  ];

  return (
    <>
      {/* Top Navigation */}
      <nav className="bg-slate-800/95 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 py-4">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-2xl flex items-center justify-center shadow-lg">
              <span className="text-slate-900 font-bold text-xl">W</span>
            </div>

            {/* TITLE ROW */}
            <div className="flex-1">
              <div className="flex items-center justify-between gap-4">
                {/* Left: Product title */}
                <h1 className="text-2xl font-bold tracking-tight text-white whitespace-nowrap">
                  AI-Powered Social Intelligence
                </h1>

                {/* Right: Ownership + links */}
                <div className="flex items-center gap-2 text-2xl font-bold tracking-tight text-white whitespace-nowrap">
                  <span className="text-white/50">•</span>
                  <span>Built by Sayali Sawant</span>
                  <span className="text-white/50">•</span>
                  <a
                    href="https://github.com/SayaliSawant0101/Walmart-Social-Listening-System"
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-4 hover:text-emerald-300 transition-colors"
                  >
                    GitHub
                  </a>
                  <span className="text-white/50">•</span>
                  <a
                    href="https://sayalis.org"
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-4 hover:text-emerald-300 transition-colors"
                  >
                    Portfolio
                  </a>
                </div>
              </div>

              <p className="text-slate-400">
                Advanced Analytics Platform
              </p>
            </div>
          </div>
        </div>
      </nav>

      {/* Sidebar */}
      <div className="fixed left-0 top-20 h-screen w-64 bg-slate-800/95 backdrop-blur-md border-r border-slate-700 z-40">
        <div className="p-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                location.pathname === item.path
                  ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-400 border border-emerald-400/30 shadow-lg shadow-emerald-400/10"
                  : "text-slate-400 hover:text-white hover:bg-slate-700/30"
              }`}
            >
              <div className="w-8 h-8 bg-slate-700/50 rounded-lg flex items-center justify-center">
                <span className="text-sm">•</span>
              </div>
              <div>
                <div className="font-semibold text-sm">{item.label}</div>
                <div className="text-xs text-slate-500">{item.description}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </>
  );
}
