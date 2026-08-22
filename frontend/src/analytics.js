// frontend/src/analytics.js
//
// Centralized helper for sending events into Google Tag Manager's dataLayer.
// GTM (see index.html) picks these up via Custom Event triggers and forwards
// them to GA4 as events. Keeping every push() call in one file means:
//   - one place to see every event this app can send
//   - one place to fix event naming/params if GA4's schema needs change
//   - components stay free of raw dataLayer plumbing

function pushEvent(eventName, params = {}) {
  if (typeof window === "undefined") return;
  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push({
    event: eventName,
    ...params,
  });
}

/** Fired on every client-side route change (Dashboard / Theme Analysis / AI Insights). */
export function trackVirtualPageview(pagePath, pageTitle) {
  pushEvent("virtual_pageview", {
    page_path: pagePath,
    page_title: pageTitle,
  });
}

/** Fired when a user edits the shared start/end date filter, on any page. */
export function trackDateRangeChange({ page, changedField, start, end }) {
  pushEvent("date_range_change", {
    page,
    changed_field: changedField, // "start" or "end"
    date_start: start,
    date_end: end,
  });
}

/** Fired when the user clicks "Generate Themes" on the Theme Analysis page. */
export function trackGenerateThemes({ themeCount, start, end }) {
  pushEvent("generate_themes", {
    theme_count: themeCount ?? "auto",
    date_start: start,
    date_end: end,
  });
}

/** Fired when the user clicks "Generate Executive Summary" on AI Insights. */
export function trackExecutiveSummary({ start, end }) {
  pushEvent("generate_executive_summary", {
    date_start: start,
    date_end: end,
  });
}

/** Fired when the user clicks "Generate Structured Brief" on AI Insights. */
export function trackStructuredBrief({ keyword, start, end }) {
  pushEvent("generate_structured_brief", {
    keyword: keyword && keyword.trim() ? keyword.trim() : "(none)",
    date_start: start,
    date_end: end,
  });
}

/** Fired when a user clicks into a chart point/bar to open a drilldown modal. */
export function trackDashboardDrilldown({ chartName, label }) {
  pushEvent("dashboard_drilldown", {
    chart_name: chartName,
    chart_label: label,
  });
}

/** Fired when a user toggles a sentiment category on/off via the chart legend. */
export function trackLegendToggle({ chartName, sentiment }) {
  pushEvent("sentiment_legend_toggle", {
    chart_name: chartName,
    sentiment,
  });
}
