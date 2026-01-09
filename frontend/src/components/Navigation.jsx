import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useDate } from "../contexts/DateContext";

// --- helpers ---
function iso(x) {
  if (!x) return "";
  // HTML date inputs already return YYYY-MM-DD format, so use it directly
  // This prevents timezone issues where Date() constructor can shift dates
  if (typeof x === "string" && /^\d{4}-\d{2}-\d{2}$/.test(x)) {
    return x;
  }
  // Fallback for other date formats
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
    {
      path: "/",
      label: "Dashboard",
      description: "Overview & Sentiment Trends",
    },
    {
      path: "/aspect-analysis",
      label: "Aspect Analysis",
      description: "Aspect × Sentiment Breakdown",
    },
    {
      path: "/theme-analysis",
      label: "Theme Analysis",
      description: "AI-Generated Theme Clusters",
    },
    {
      path: "/ai-insights",
      label: "AI Insights",
      description: "Executive Summary & Briefs",
    },
  ];

  return (
    <>
      {/* Top Navigation - Branding Only */}
      <nav className="bg-slate-800/95 backdrop-blur-md border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-2xl flex items-center justify-center shadow-lg">
                <span className="text-slate-900 font-bold text-xl">W</span>
              </div>

              {/* Title + Ownership */}
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Walmart Social Intelligence
                </h1>

                <p className="text-slate-400">Advanced Analytics Platform</p>

                {/* Ownership watermark */}
                <div className="mt-1 text-xs text-slate-400 flex flex-wrap items-center gap-2">
                  <span>
                    Built by{" "}
                    <span className="font-semibold text-slate-200">
                      Sayali Sawant
                    </span>
                  </span>
                  <span className="opacity-50">•</span>
                  <a
                    href="https://github.com/SayaliSawant0101/Walmart-Social-Listening-System"
                    target="_blank"
                    rel="noreferrer"
                    className="underline hover:text-slate-200"
                  >
                    GitHub
                  </a>
                  <span className="opacity-50">•</span>
                  <a
                    href="https://sayalis.org"
                    target="_blank"
                    rel="noreferrer"
                    className="underline hover:text-slate-200"
                  >
                    Portfolio
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Sidebar Navigation */}
      <div className="fixed left-0 top-20 h-screen w-64 bg-slate-800/95 backdrop-blur-md border-r border-slate-700 z-40">
        <div className="p-4">
          <div className="space-y-2">
            {navItems.map((item) => {
              const getIcon = (path) => {
                switch (path) {
                  case "/":
                    return (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z"
                        />
                      </svg>
                    );
                  case "/aspect-analysis":
                    return (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                        />
                      </svg>
                    );
                  case "/theme-analysis":
                    return (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                        />
                      </svg>
                    );
                  case "/ai-insights":
                    return (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                        />
                      </svg>
                    );
                  default:
                    return null;
                }
              };

              return (
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
                    {getIcon(item.path)}
                  </div>
                  <div>
                    <div className="font-semibold text-sm">{item.label}</div>
                    <div className="text-xs text-slate-500">
                      {item.description}
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
