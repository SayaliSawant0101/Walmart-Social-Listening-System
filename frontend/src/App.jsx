// frontend/src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { DateProvider } from "./contexts/DateContext";
import Navigation from "./components/Navigation";
import Dashboard from "./pages/Dashboard";
import AspectAnalysis from "./pages/AspectAnalysis";
import ThemeAnalysis from "./pages/ThemeAnalysis";
import AIInsights from "./pages/AIInsights";

/* ---------- Build Stamp Header ---------- */

function BuildStampHeader() {
  const commit = import.meta.env.VITE_GIT_COMMIT || "local";
  const context = import.meta.env.VITE_NETLIFY_CONTEXT || "dev";
  const buildTime = import.meta.env.VITE_BUILD_TIME || "";

  const shortCommit = commit ? commit.slice(0, 7) : "local";

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-slate-950/90 backdrop-blur border-b border-white/10">
      <div className="ml-64 px-6 py-2 text-xs text-white/80 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span>
            Built by <span className="font-semibold">Sayali Sawant</span>
          </span>
          <span className="opacity-50">•</span>
          <a
            href="https://github.com/SayaliSawant0101/Walmart-Social-Listening-System"
            target="_blank"
            rel="noreferrer"
            className="underline hover:opacity-90"
          >
            GitHub
          </a>
          <span className="opacity-50">•</span>
          <a
            href="https://sayalis.org"
            target="_blank"
            rel="noreferrer"
            className="underline hover:opacity-90"
          >
            Portfolio
          </a>
        </div>

        <div className="opacity-60 whitespace-nowrap">
          Build: {buildTime || "N/A"} • {context} • {shortCommit}
        </div>
      </div>
    </div>
  );
}

/* ---------- Main App ---------- */

export default function App() {
  React.useEffect(() => {
    document.title = "Sayali Sawant | Walmart Social Listening System";
  }, []);

  return (
    <DateProvider>
      <Router>
        <div className="min-h-screen bg-slate-900 text-white">
          <Navigation />

          {/* Ownership header */}
          <BuildStampHeader />

          {/* Push content down so it doesn't hide under header */}
          <main className="ml-64 pt-10">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/aspect-analysis" element={<AspectAnalysis />} />
              <Route path="/theme-analysis" element={<ThemeAnalysis />} />
              <Route path="/ai-insights" element={<AIInsights />} />
            </Routes>
          </main>
        </div>
      </Router>
    </DateProvider>
  );
}
