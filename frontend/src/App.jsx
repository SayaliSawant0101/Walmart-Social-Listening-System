// frontend/src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { DateProvider } from "./contexts/DateContext";
import Navigation from "./components/Navigation";
import Dashboard from "./pages/Dashboard";
import AspectAnalysis from "./pages/AspectAnalysis";
import ThemeAnalysis from "./pages/ThemeAnalysis";
import AIInsights from "./pages/AIInsights";

/* ---------------- Footer + Build Stamp ---------------- */

function BuildStamp() {
  const commit = import.meta.env.VITE_GIT_COMMIT || "local";
  const context = import.meta.env.VITE_NETLIFY_CONTEXT || "dev";
  const buildTime = import.meta.env.VITE_BUILD_TIME || "";

  const shortCommit = commit ? commit.slice(0, 7) : "local";

  return (
    <div className="text-xs opacity-60 whitespace-nowrap">
      Build: {buildTime || "N/A"} • {context} • {shortCommit}
    </div>
  );
}

function AppFooter() {
  return (
    <footer className="mt-10 px-6 py-4 border-t border-white/10 text-xs text-white/80">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span>
            © {new Date().getFullYear()} Built by{" "}
            <span className="font-semibold">Sayali Sawant</span>
          </span>
          <span className="opacity-50">•</span>
          <a
            href="https://github.com/SayaliSawant0101/Walmart-Social-Listening-System"
            target="_blank"
            rel="noreferrer"
            className="underline hover:opacity-90"
          >
            Source (GitHub)
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
        <BuildStamp />
      </div>
    </footer>
  );
}

/* ---------------- Main App ---------------- */

export default function App() {
  // Browser title watermark
  React.useEffect(() => {
    document.title = "Sayali Sawant | Walmart Social Listening System";
  }, []);

  return (
    <DateProvider>
      <Router>
        <div className="min-h-screen bg-slate-900 text-white flex">
          <Navigation />

          <div className="flex-1 ml-64">
            <main className="min-h-screen">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/aspect-analysis" element={<AspectAnalysis />} />
                <Route path="/theme-analysis" element={<ThemeAnalysis />} />
                <Route path="/ai-insights" element={<AIInsights />} />
              </Routes>
            </main>

            {/* Ownership footer */}
            <AppFooter />
          </div>
        </div>
      </Router>
    </DateProvider>
  );
}
