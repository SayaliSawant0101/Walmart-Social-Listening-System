// frontend/src/pages/AspectAnalysis.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { Chart } from "chart.js";
import {
  getAspectSentimentSplit,
  getSampleTweets,
  getCompetitorAspectSentimentSplit,
} from "../api";
import { useDate } from "../contexts/DateContext";

// Helper function to format date
function iso(x) {
  if (!x) return "";
  if (typeof x === "string" && /^\d{4}-\d{2}-\d{2}$/.test(x)) return x;
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function AspectAnalysis() {
  const { start, end, meta, setStart, setEnd } = useDate();

  const [split, setSplit] = useState(null); // Walmart
  const [competitorSplit, setCompetitorSplit] = useState(null); // Costco
  const [selectedBrand, setSelectedBrand] = useState("all"); // "walmart", "costco", "all"
  const [splitAsPercent, setSplitAsPercent] = useState(false);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [totalTweets, setTotalTweets] = useState(0);

  const [sampleTweetsModal, setSampleTweetsModal] = useState({
    isOpen: false,
    aspect: null,
    sentiment: null,
    tweets: [],
  });
  const [loadingTweets, setLoadingTweets] = useState(false);

  // ---- load aspects × sentiment split ----
  useEffect(() => {
    if (!start || !end) return;

    (async () => {
      try {
        setLoading(true);
        setErr("");

        const promises = [];

        if (selectedBrand === "walmart" || selectedBrand === "all") {
          promises.push(
            getAspectSentimentSplit(start, end, splitAsPercent, true)
              .then((s) => ({ type: "walmart", data: s }))
              .catch(() => ({ type: "walmart", data: null }))
          );
        }

        if (selectedBrand === "costco" || selectedBrand === "all") {
          promises.push(
            getCompetitorAspectSentimentSplit(start, end, splitAsPercent, true)
              .then((s) => ({ type: "costco", data: s }))
              .catch(() => ({ type: "costco", data: null }))
          );
        }

        const results = await Promise.all(promises);

        results.forEach((r) => {
          if (r.type === "walmart") setSplit(r.data);
          if (r.type === "costco") setCompetitorSplit(r.data);
        });

        const sumCounts = (d) => {
          if (!d?.labels || !d?.counts) return 0;
          const c = d.counts;
          return d.labels.reduce((sum, _, i) => {
            const pos = c.positive?.[i] || 0;
            const neu = c.neutral?.[i] || 0;
            const neg = c.negative?.[i] || 0;
            return sum + pos + neu + neg;
          }, 0);
        };

        let total = 0;
        if (selectedBrand === "all") {
          const w = results.find((x) => x.type === "walmart")?.data;
          const c = results.find((x) => x.type === "costco")?.data;
          total = sumCounts(w) + sumCounts(c);
        } else if (selectedBrand === "walmart") {
          total = sumCounts(results.find((x) => x.type === "walmart")?.data);
        } else {
          total = sumCounts(results.find((x) => x.type === "costco")?.data);
        }

        setTotalTweets(total);
      } catch (e) {
        setErr("Failed to load aspect data");
        console.error("Failed to load aspect split", e);
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end, splitAsPercent, selectedBrand]);

  // Function to fetch sample tweets for specific aspect and sentiment
  const fetchSampleTweets = async (aspect, sentiment) => {
    try {
      setLoadingTweets(true);
      const tweets = await getSampleTweets(start, end, aspect, sentiment, 10);
      setSampleTweetsModal({ isOpen: true, aspect, sentiment, tweets });
    } catch (error) {
      console.error("Failed to fetch sample tweets:", error);
      const sampleTweets = [
        `Sample tweet about ${aspect} with ${sentiment} sentiment`,
        `Another ${sentiment} tweet regarding ${aspect}`,
        `Customer feedback on ${aspect} - ${sentiment} tone`,
        `User experience with ${aspect} was ${sentiment}`,
        `Review of ${aspect} service - ${sentiment} sentiment`,
      ];
      setSampleTweetsModal({
        isOpen: true,
        aspect,
        sentiment,
        tweets: sampleTweets,
      });
    } finally {
      setLoadingTweets(false);
    }
  };

  // Download functions for specific aspect and sentiment (from modal)
  const downloadTweetsAsExcel = async () => {
    try {
      const allTweets = await getSampleTweets(
        start,
        end,
        sampleTweetsModal.aspect,
        sampleTweetsModal.sentiment,
        1000
      );

      if (!allTweets || allTweets.length === 0) {
        alert("No tweets found to download");
        return;
      }

      const excelData = allTweets.map((tweet, index) => ({
        "Tweet #": index + 1,
        "Tweet Text": tweet,
        Aspect: sampleTweetsModal.aspect,
        Sentiment: sampleTweetsModal.sentiment,
        "Date Range": `${start} to ${end}`,
      }));

      const csvContent = [
        Object.keys(excelData[0]).join(","),
        ...excelData.map((row) =>
          Object.values(row)
            .map((val) => `"${String(val).replace(/"/g, '""')}"`)
            .join(",")
        ),
      ].join("\n");

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute(
        "download",
        `tweets_${sampleTweetsModal.aspect}_${sampleTweetsModal.sentiment}_${start}_to_${end}.csv`
      );
      link.style.visibility = "hidden";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download Excel:", error);
      if (error.response?.status === 422) {
        alert(
          `Invalid parameters: ${
            error.response?.data?.detail ||
            "Please check your date range and aspect/sentiment values"
          }`
        );
      } else if (error.response?.status === 404) {
        alert("No data found for the selected criteria");
      } else {
        alert(`Failed to download Excel file: ${error.message}`);
      }
    }
  };

  const downloadTweetsAsPDF = async () => {
    try {
      const allTweets = await getSampleTweets(
        start,
        end,
        sampleTweetsModal.aspect,
        sampleTweetsModal.sentiment,
        1000
      );

      if (!allTweets || allTweets.length === 0) {
        alert("No tweets found to download");
        return;
      }

      const pdfContent = `
        <!DOCTYPE html>
        <html>
          <head>
            <title>Tweets Report - ${sampleTweetsModal.aspect} (${sampleTweetsModal.sentiment})</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; }
              .meta { background: #f5f5f5; padding: 10px; margin-bottom: 20px; }
              .tweet { margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; }
              .tweet-number { font-weight: bold; color: #666; }
            </style>
          </head>
          <body>
            <h1>Tweets Report</h1>
            <div class="meta">
              <p><strong>Aspect:</strong> ${sampleTweetsModal.aspect}</p>
              <p><strong>Sentiment:</strong> ${sampleTweetsModal.sentiment}</p>
              <p><strong>Date Range:</strong> ${start} to ${end}</p>
              <p><strong>Total Tweets:</strong> ${allTweets.length}</p>
            </div>
            ${allTweets
              .map(
                (tweet, index) => `
              <div class="tweet">
                <div class="tweet-number">Tweet ${index + 1}</div>
                <div>${String(tweet).replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
              </div>
            `
              )
              .join("")}
          </body>
        </html>
      `;

      const newWindow = window.open("", "_blank");
      if (!newWindow) {
        alert("Please allow popups for this site to download PDF");
        return;
      }
      newWindow.document.write(pdfContent);
      newWindow.document.close();
      newWindow.focus();
      setTimeout(() => newWindow.print(), 1000);
    } catch (error) {
      console.error("Failed to download PDF:", error);
      if (error.response?.status === 422) {
        alert(
          `Invalid parameters: ${
            error.response?.data?.detail ||
            "Please check your date range and aspect/sentiment values"
          }`
        );
      } else if (error.response?.status === 404) {
        alert("No data found for the selected criteria");
      } else {
        alert(`Failed to download PDF file: ${error.message}`);
      }
    }
  };

  // View all tweets as PDF (view only)
  const viewAllTweetsAsPDF = async () => {
    try {
      const allAspects = stackedBarData.labels || [];
      const allTweets = [];

      for (const aspect of allAspects) {
        const [positiveTweets, neutralTweets, negativeTweets] = await Promise.all([
          getSampleTweets(start, end, aspect, "positive", 1000),
          getSampleTweets(start, end, aspect, "neutral", 1000),
          getSampleTweets(start, end, aspect, "negative", 1000),
        ]);

        allTweets.push(
          ...positiveTweets.map((tweet) => ({ tweet, aspect, sentiment: "positive" })),
          ...neutralTweets.map((tweet) => ({ tweet, aspect, sentiment: "neutral" })),
          ...negativeTweets.map((tweet) => ({ tweet, aspect, sentiment: "negative" }))
        );
      }

      if (allTweets.length === 0) {
        alert("No tweets found to view");
        return;
      }

      const pdfContent = `
        <html>
          <head>
            <title>All Tweets Report</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; text-align: center; }
              .tweet { margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; background: #f8f9fa; }
              .metadata { font-size: 12px; color: #666; margin-top: 5px; }
              .positive { border-left-color: #28a745; }
              .negative { border-left-color: #dc3545; }
              .neutral { border-left-color: #ffc107; }
            </style>
          </head>
          <body>
            <h1>All Tweets Report</h1>
            <p><strong>Date Range:</strong> ${start} to ${end}</p>
            <p><strong>Total Tweets:</strong> ${allTweets.length}</p>
            ${allTweets
              .map(
                (item, index) => `
              <div class="tweet ${item.sentiment}">
                <strong>Tweet #${index + 1}</strong><br>
                ${String(item.tweet).replace(/</g, "&lt;").replace(/>/g, "&gt;")}<br>
                <div class="metadata">
                  <strong>Aspect:</strong> ${item.aspect} |
                  <strong>Sentiment:</strong> ${item.sentiment}
                </div>
              </div>
            `
              )
              .join("")}
          </body>
        </html>
      `;

      const viewWindow = window.open("", "_blank");
      viewWindow.document.write(pdfContent);
      viewWindow.document.close();
    } catch (error) {
      console.error("Failed to view all tweets:", error);
      alert("Failed to view all tweets. Please try again.");
    }
  };

  /**
   * Chart data:
   * - If Brand=All: X-axis has 6 aspects. Two stacks per aspect (Walmart + Costco).
   * - If single brand: one stack per aspect.
   */
  const stackedBarData = useMemo(() => {
    // Single brand view
    if (selectedBrand !== "all") {
      const data = selectedBrand === "costco" ? competitorSplit : split;
      if (!data) return { labels: [], datasets: [] };

      const labels = (data.labels || []).map((t) => String(t).toUpperCase());
      const series = splitAsPercent ? data.percent : data.counts;

      const totals = labels.map((_, i) => {
        const pos = series.positive?.[i] || 0;
        const neu = series.neutral?.[i] || 0;
        const neg = series.negative?.[i] || 0;
        return pos + neu + neg;
      });

      return {
        labels,
        datasets: [
          {
            label: "Positive",
            sentimentKey: "positive",
            brand: selectedBrand,
            data: series.positive || [],
            backgroundColor: "#4ade80",
            stack: "brand",
            totals,
          },
          {
            label: "Neutral",
            sentimentKey: "neutral",
            brand: selectedBrand,
            data: series.neutral || [],
            backgroundColor: "#fbbf24",
            stack: "brand",
            totals,
          },
          {
            label: "Negative",
            sentimentKey: "negative",
            brand: selectedBrand,
            data: series.negative || [],
            backgroundColor: "#f87171",
            stack: "brand",
            totals,
          },
        ],
      };
    }

    // All brands view
    if (!split && !competitorSplit) return { labels: [], datasets: [] };

    const walmartLabels = split?.labels || [];
    const costcoLabels = competitorSplit?.labels || [];
    const aspectOrder = [...walmartLabels, ...costcoLabels.filter((a) => !walmartLabels.includes(a))];

    const labels = aspectOrder.map((t) => String(t).toUpperCase());

    const wSeries = splitAsPercent ? split?.percent || {} : split?.counts || {};
    const cSeries = splitAsPercent ? competitorSplit?.percent || {} : competitorSplit?.counts || {};

    const alignSeries = (brandLabels, seriesArr, aspectList) =>
      aspectList.map((a) => {
        const idx = brandLabels.indexOf(a);
        return idx >= 0 ? seriesArr?.[idx] || 0 : 0;
      });

    const wPos = alignSeries(walmartLabels, wSeries.positive, aspectOrder);
    const wNeu = alignSeries(walmartLabels, wSeries.neutral, aspectOrder);
    const wNeg = alignSeries(walmartLabels, wSeries.negative, aspectOrder);

    const cPos = alignSeries(costcoLabels, cSeries.positive, aspectOrder);
    const cNeu = alignSeries(costcoLabels, cSeries.neutral, aspectOrder);
    const cNeg = alignSeries(costcoLabels, cSeries.negative, aspectOrder);

    const wTotals = aspectOrder.map((_, i) => (wPos[i] || 0) + (wNeu[i] || 0) + (wNeg[i] || 0));
    const cTotals = aspectOrder.map((_, i) => (cPos[i] || 0) + (cNeu[i] || 0) + (cNeg[i] || 0));

    return {
      labels,
      datasets: [
        // Walmart stack
        { label: "Positive", sentimentKey: "positive", brand: "walmart", data: wPos, backgroundColor: "#4ade80", stack: "Walmart", totals: wTotals },
        { label: "Neutral", sentimentKey: "neutral", brand: "walmart", data: wNeu, backgroundColor: "#fbbf24", stack: "Walmart", totals: wTotals },
        { label: "Negative", sentimentKey: "negative", brand: "walmart", data: wNeg, backgroundColor: "#f87171", stack: "Walmart", totals: wTotals },

        // Costco stack
        { label: "Positive", sentimentKey: "positive", brand: "costco", data: cPos, backgroundColor: "#4ade80", stack: "Costco", totals: cTotals },
        { label: "Neutral", sentimentKey: "neutral", brand: "costco", data: cNeu, backgroundColor: "#fbbf24", stack: "Costco", totals: cTotals },
        { label: "Negative", sentimentKey: "negative", brand: "costco", data: cNeg, backgroundColor: "#f87171", stack: "Costco", totals: cTotals },
      ],
    };
  }, [split, competitorSplit, splitAsPercent, selectedBrand]);

  // --- Custom plugin: draw aspect centered + Walmart/Costco under each aspect (Brand=All) ---
  const brandAspectLabelPlugin = useMemo(
    () => ({
      id: "brandAspectLabelPlugin",
      afterDraw: (chart) => {
        if (selectedBrand !== "all") return;

        const { ctx, scales } = chart;
        const xAxis = scales?.x;
        if (!ctx || !xAxis) return;

        const walmartDsIndex = chart.data.datasets.findIndex((d) => d.brand === "walmart");
        const costcoDsIndex = chart.data.datasets.findIndex((d) => d.brand === "costco");
        if (walmartDsIndex < 0 || costcoDsIndex < 0) return;

        const walmartMeta = chart.getDatasetMeta(walmartDsIndex);
        const costcoMeta = chart.getDatasetMeta(costcoDsIndex);
        if (!walmartMeta?.data?.length || !costcoMeta?.data?.length) return;

        const labels = chart.data.labels || [];

        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.imageSmoothingEnabled = true;

        const brandY = xAxis.bottom + 0;
        const aspectY = xAxis.bottom + 28;

        const aspectColor = "#ffffff";
        const brandColor = "#ffffff";

        labels.forEach((label, i) => {
          const walmartBar = walmartMeta.data[i];
          const costcoBar = costcoMeta.data[i];
          if (!walmartBar || !costcoBar) return;

          const wx = walmartBar.x;
          const cx = costcoBar.x;
          if (typeof wx !== "number" || typeof cx !== "number") return;

          const centerX = (wx + cx) / 2;

          // Aspect title centered between bars
          ctx.font = "bold 12px Arial";
          ctx.fillStyle = aspectColor;
          ctx.fillText(String(label).toUpperCase(), centerX, aspectY);

          // Brand labels under each bar
          ctx.font = "11px Arial";
          ctx.fillStyle = brandColor;
          ctx.fillText("Walmart", wx, brandY);
          ctx.fillText("Costco", cx, brandY);
        });

        ctx.restore();
      },
    }),
    [selectedBrand]
  );

  const chartOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          bottom: selectedBrand === "all" ? 50 : 0,
          left: 10,
          right: 10,
        },
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#ffffff",
            font: { size: 12, weight: "bold" },
            generateLabels: function () {
              // Show only 3 legend entries
              const mk = (text, fillStyle) => ({
                text,
                fillStyle,
                strokeStyle: fillStyle,
                lineWidth: 0,
                hidden: false,
                datasetIndex: -1, // synthetic
                fontColor: "#ffffff",
              });
              return [
                mk("Positive", "#4ade80"),
                mk("Neutral", "#fbbf24"),
                mk("Negative", "#f87171"),
              ];
            },
          },
          onClick: (e, legendItem, legend) => {
            // Toggle all datasets with same sentimentKey
            const chart = legend.chart;
            const sentiment = (legendItem.text || "").toLowerCase();

            chart.data.datasets.forEach((ds) => {
              if ((ds.sentimentKey || "").toLowerCase() === sentiment) {
                ds.hidden = !ds.hidden;
              }
            });
            chart.update();
          },
        },

        datalabels: {
          display: (context) => {
            const v = context?.parsed?.y ?? 0;
            return v > 0;
          },
          color: "white",
          font: { weight: "bold", size: 11 },
          formatter: (value, context) => {
            const totals = context.dataset?.totals || [];
            const total = totals[context.dataIndex] || 0;
            const pct = total > 0 ? Math.round((value / total) * 100) : 0;
            return value > 0 ? `${pct}%` : "";
          },
          anchor: "center",
          align: "center",
          offset: 0,
          padding: 0,
        },

        tooltip: {
          callbacks: {
            title: (items) => items?.[0]?.label || "",
            label: (ctx) => {
              const brand = ctx.dataset?.brand ? ctx.dataset.brand.toUpperCase() : "";
              const sentiment = ctx.dataset?.label || "";
              const val = ctx.parsed?.y ?? 0;
              return `${brand} • ${sentiment}: ${val.toLocaleString()}`;
            },
            afterLabel: (ctx) => {
              const totals = ctx.dataset?.totals || [];
              const total = totals[ctx.dataIndex] || 0;
              const value = ctx.parsed?.y ?? 0;
              const pct = total > 0 ? Math.round((value / total) * 100) : 0;
              return `Total (${ctx.dataset?.brand?.toUpperCase() || ""}): ${total.toLocaleString()} • ${pct}%`;
            },
          },
        },
      },

      scales: {
        x: {
          stacked: false,
          grid: { display: false },
          ticks: {
            color: "#ffffff",
            font: { size: 10, weight: "normal" },
            maxRotation: 0,
            minRotation: 0,
            padding: 6,
            autoSkip: false,
            callback: function (value) {
              // In Brand=All mode: hide axis tick text (plugin draws it cleanly)
              if (selectedBrand === "all") return "";
              return this.getLabelForValue(value);
            },
          },
        },
        y: {
          stacked: true,
          suggestedMax: splitAsPercent ? 100 : undefined,
          beginAtZero: true,
          ticks: {
            callback: (v) => (splitAsPercent ? `${v}%` : v),
            color: "#ffffff",
            font: { size: 11 },
            padding: 8,
          },
          grid: {
            color: "rgba(148, 163, 184, 0.15)",
            lineWidth: 1,
          },
        },
      },
    };
  }, [selectedBrand, splitAsPercent]);

  return (
    <div className="space-y-4">
      {/* Aspect Analysis Header */}
      <div className="mb-0.5 pt-2 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center shadow-lg ml-2">
              <svg className="w-3 h-3 text-slate-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <div className="px-2 py-1">
              <h1 className="text-xl font-bold text-white">Aspect Analysis</h1>
              <p className="text-slate-400 text-sm">Aspect × Sentiment Breakdown</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Brand Selector */}
            <div className="flex items-center space-x-1 bg-slate-700/30 backdrop-blur-sm rounded-md border border-slate-600/50 px-2 py-2">
              <span className="text-xs text-slate-200 font-medium">Brand:</span>
              <select
                value={selectedBrand}
                onChange={(e) => setSelectedBrand(e.target.value)}
                className="px-2 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
              >
                <option value="all">All</option>
                <option value="walmart">Walmart</option>
                <option value="costco">Costco</option>
              </select>
            </div>

            {/* Date Card */}
            <div className="flex items-center space-x-1 bg-slate-700/30 backdrop-blur-sm rounded-md border border-slate-600/50 px-3 py-2">
              <div className="w-1 h-1 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
              <span className="text-xs text-slate-200 font-medium">Date:</span>
              <input
                type="date"
                value={start || ""}
                min={meta?.min || ""}
                max={end || meta?.max || ""}
                onChange={(e) => setStart(iso(e.target.value))}
                className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
              />
              <span className="text-slate-400 text-xs">to</span>
              <input
                type="date"
                value={end || ""}
                min={start || meta?.min || ""}
                max={meta?.max || ""}
                onChange={(e) => setEnd(iso(e.target.value))}
                className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Error */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-4">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
            <span className="text-red-300 font-semibold text-sm">{err}</span>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-slate-300 font-semibold">Loading aspect analysis...</span>
          </div>
        </div>
      )}

      {/* Chart */}
      {!loading && split && (
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl px-4 pt-2 pb-0 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-end mb-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => viewAllTweetsAsPDF()}
                  className="flex items-center space-x-1 bg-slate-600 hover:bg-slate-700 text-white px-2 py-1 rounded text-xs transition-colors"
                  title="View All Tweets as PDF"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                    />
                  </svg>
                  <span>View All Tweets</span>
                </button>
              </div>

              <div className="text-sm text-slate-300 bg-slate-700/30 px-3 py-1 rounded-lg">
                Total: {totalTweets.toLocaleString()} tweets
              </div>
            </div>
          </div>

          <div className="h-80" style={{ position: "relative" }}>
            {stackedBarData?.labels?.length > 0 && stackedBarData?.datasets?.length > 0 && (
              <Bar
                key={`aspect-chart-${selectedBrand}-${splitAsPercent}-${stackedBarData.labels.length}-${stackedBarData.datasets.length}`}
                data={stackedBarData}
                options={{
                  ...chartOptions,
                  onClick: (event, elements) => {
                    if (!elements?.length) return;
                    const el = elements[0];
                    const datasetIndex = el.datasetIndex;
                    const dataIndex = el.index;

                    const aspect = (stackedBarData.labels?.[dataIndex] || "").toLowerCase();
                    const dataset = stackedBarData.datasets?.[datasetIndex];
                    const sentiment = (dataset?.sentimentKey || dataset?.label || "").toLowerCase();

                    fetchSampleTweets(aspect, sentiment);
                  },
                }}
                plugins={[ChartDataLabels, brandAspectLabelPlugin]}
              />
            )}
          </div>
        </div>
      )}

      {/* Sample Tweets Modal */}
      {sampleTweetsModal.isOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-800 rounded-2xl p-6 max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-bold text-white">
                  Sample Tweets - {sampleTweetsModal.aspect?.toUpperCase()} ({sampleTweetsModal.sentiment?.toUpperCase()})
                </h3>
                <p className="text-sm text-slate-400">Raw tweets for this aspect and sentiment combination</p>
              </div>
              <button
                onClick={() => setSampleTweetsModal({ isOpen: false, aspect: null, sentiment: null, tweets: [] })}
                className="text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3">
              {loadingTweets ? (
                <div className="flex items-center justify-center py-8">
                  <div className="flex items-center space-x-3">
                    <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="text-slate-300 font-semibold">Loading tweets...</span>
                  </div>
                </div>
              ) : (
                <>
                  {sampleTweetsModal.tweets.map((tweet, index) => (
                    <div key={index} className="bg-slate-700/50 rounded-lg p-4 border border-slate-600/30">
                      <p className="text-slate-200 text-sm leading-relaxed">{tweet}</p>
                    </div>
                  ))}

                  <div className="flex items-center justify-center space-x-3 pt-4 border-t border-slate-600/30">
                    <button
                      onClick={() => downloadTweetsAsExcel()}
                      className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      <span className="text-sm font-medium">Download CSV</span>
                    </button>

                    <button
                      onClick={() => downloadTweetsAsPDF()}
                      className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      <span className="text-sm font-medium">Download PDF</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
