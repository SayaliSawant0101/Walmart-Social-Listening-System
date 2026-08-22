import React from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import { DateProvider } from "./contexts/DateContext";
import Navigation from "./components/Navigation";
import Dashboard from "./pages/Dashboard";
import ThemeAnalysis from "./pages/ThemeAnalysis";
import AIInsights from "./pages/AIInsights";
import { trackVirtualPageview } from "./analytics";

// react-router-dom handles navigation client-side (no full page load), so
// GTM's built-in triggers never see it happen. This component pushes a
// "virtual_pageview" event to the dataLayer every time the route changes,
// which we'll use in GTM as the trigger for a GA4 page_view-equivalent tag.
function RouteTracker() {
  const location = useLocation();

  React.useEffect(() => {
    trackVirtualPageview(location.pathname + location.search, document.title);
  }, [location]);

  return null;
}

export default function App() {
  React.useEffect(() => {
    document.title = "Sayali Sawant | AI-Powered Social Listening System";
  }, []);

  return (
    <DateProvider>
      <Router>
        <div className="min-h-screen bg-slate-900 text-white">
          <RouteTracker />
          <Navigation />

          <main className="ml-48">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route
                path="/theme-analysis"
                element={<ThemeAnalysis />}
              />
              <Route
                path="/ai-insights"
                element={<AIInsights />}
              />
            </Routes>
          </main>
        </div>
      </Router>
    </DateProvider>
  );
}
