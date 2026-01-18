
// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Line, Bar } from "react-chartjs-2";
import "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { Chart } from "chart.js";
import {
  getSummary,
  getTrend,
  getAspectSentimentSplit,
  getCompetitorSummary,
  getCompetitorTrend,
  getCompetitorAspectSentimentSplit,
} from "../api";
import { useDate } from "../contexts/DateContext";

// --- helpers ---
function iso(x) {
  if (!x) return "";
  if (typeof x === "string" && /^\d{4}-\d{2}-\d{2}$/.test(x)) return x;
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function Dashboard() {
  const { start, end, meta, setStart, setEnd, loading: metaLoading } = useDate();

  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState([]);
  const [competitorSummary, setCompetitorSummary] = useState(null);
  const [competitorTrend, setCompetitorTrend] = useState([]);
  const [selectedBrand, setSelectedBrand] = useState("all"); // "walmart", "costco", "all"

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [timePeriod, setTimePeriod] = useState("daily");
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMoreData, setHasMoreData] = useState(true);
  const [currentOffset, setCurrentOffset] = useState(0);

  const [aspectSplitModal, setAspectSplitModal] = useState({
    isOpen: false,
    sentiment: null,
    data: null,
    showByBrand: false,
  });
  const [loadingAspectSplit, setLoadingAspectSplit] = useState(false);

  const [selectedDateModal, setSelectedDateModal] = useState({
    isOpen: false,
    date: null,
    data: null,
  });
  const [loadingDateAspects, setLoadingDateAspects] = useState(false);

  // We'll keep a cache of daily counts keyed by YYYY-MM-DD
  const [tweetCountsCache, setTweetCountsCache] = useState({});

  // Aspect data for sentiment drivers
  const [walmartAspectData, setWalmartAspectData] = useState(null);
  const [costcoAspectData, setCostcoAspectData] = useState(null);

  // Function to get actual tweet counts for a specific date (day window)
  const getTweetCountsForDate = async (date) => {
    try {
      const dateObj = new Date(date);
      const nextDay = new Date(dateObj);
      nextDay.setDate(dateObj.getDate() + 1);

      const startDate = date;
      const endDate = nextDay.toISOString().split("T")[0];

      // false = counts mode, true = include brands? (your API signature)
      const data = await getAspectSentimentSplit(startDate, endDate, false, true);

      if (data?.labels && data?.counts) {
        const positiveTotal = data.labels.reduce(
          (sum, _, idx) => sum + (data.counts.positive[idx] || 0),
          0
        );
        const neutralTotal = data.labels.reduce(
          (sum, _, idx) => sum + (data.counts.neutral[idx] || 0),
          0
        );
        const negativeTotal = data.labels.reduce(
          (sum, _, idx) => sum + (data.counts.negative[idx] || 0),
          0
        );
        const totalTweets = positiveTotal + neutralTotal + negativeTotal;

        return {
          totalTweets,
          positive: positiveTotal,
          neutral: neutralTotal,
          negative: negativeTotal,
        };
      }
      return null;
    } catch (error) {
      console.error("Failed to get tweet counts:", error);
      return null;
    }
  };

  // Helper function to transform aspect data
  const transformAspectData = (data) => {
    const transformedData = {
      totalTweets: 0,
      aspectBreakdown: [],
      sentimentBreakdown: {
        positive: { total: 0, aspects: [] },
        neutral: { total: 0, aspects: [] },
        negative: { total: 0, aspects: [] },
      },
    };

    if (data?.labels && data?.counts && data?.percent) {
      const grandTotal = data.labels.reduce((sum, _, idx) => {
        return (
          sum +
          (data.counts.positive[idx] || 0) +
          (data.counts.negative[idx] || 0) +
          (data.counts.neutral[idx] || 0)
        );
      }, 0);

      const positiveTotal = data.labels.reduce(
        (sum, _, idx) => sum + (data.counts.positive[idx] || 0),
        0
      );
      const neutralTotal = data.labels.reduce(
        (sum, _, idx) => sum + (data.counts.neutral[idx] || 0),
        0
      );
      const negativeTotal = data.labels.reduce(
        (sum, _, idx) => sum + (data.counts.negative[idx] || 0),
        0
      );

      transformedData.aspectBreakdown = data.labels
        .map((label, index) => {
          const positiveCount = data.counts.positive[index] || 0;
          const negativeCount = data.counts.negative[index] || 0;
          const neutralCount = data.counts.neutral[index] || 0;
          const totalCount = positiveCount + negativeCount + neutralCount;

          return {
            aspect: label,
            count: totalCount,
            percentage: grandTotal > 0 ? (totalCount / grandTotal) * 100 : 0,
          };
        })
        .filter((item) => item.count > 0);

      transformedData.sentimentBreakdown.positive = {
        total: positiveTotal,
        aspects: data.labels
          .map((label, index) => ({
            aspect: label,
            count: data.counts.positive[index] || 0,
            percentage:
              positiveTotal > 0
                ? ((data.counts.positive[index] || 0) / positiveTotal) * 100
                : 0,
          }))
          .filter((item) => item.count > 0),
      };

      transformedData.sentimentBreakdown.neutral = {
        total: neutralTotal,
        aspects: data.labels
          .map((label, index) => ({
            aspect: label,
            count: data.counts.neutral[index] || 0,
            percentage:
              neutralTotal > 0
                ? ((data.counts.neutral[index] || 0) / neutralTotal) * 100
                : 0,
          }))
          .filter((item) => item.count > 0),
      };

      transformedData.sentimentBreakdown.negative = {
        total: negativeTotal,
        aspects: data.labels
          .map((label, index) => ({
            aspect: label,
            count: data.counts.negative[index] || 0,
            percentage:
              negativeTotal > 0
                ? ((data.counts.negative[index] || 0) / negativeTotal) * 100
                : 0,
          }))
          .filter((item) => item.count > 0),
      };

      transformedData.totalTweets = grandTotal;
    } else {
      console.error("Invalid data structure:", data);
    }

    return transformedData;
  };

  // Format date function for chart labels
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    switch (timePeriod) {
      case "daily":
        return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      case "weekly":
        return `Week ${date.toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
      case "monthly":
        return date.toLocaleDateString("en-US", { month: "short", year: "numeric" });
      default:
        return dateStr;
    }
  };

  // ---- load sentiment summary + trend ----
  useEffect(() => {
    if (!start || !end) return;

    (async () => {
      try {
        setLoading(true);
        setErr("");
        setCurrentOffset(0);
        setHasMoreData(true);

        const promises = [];

        // Always load Walmart data
        promises.push(
          getSummary(start, end).then((s) => ({ type: "walmart", summary: s })),
          getTrend(start, end, timePeriod, 0, 50).then((t) => ({
            type: "walmart",
            trend: t?.trend || [],
          }))
        );

        // Load competitor data if needed
        if (selectedBrand === "costco" || selectedBrand === "all") {
          promises.push(
            getCompetitorSummary(start, end)
              .then((s) => ({ type: "costco", summary: s }))
              .catch(() => ({ type: "costco", summary: null })),
            getCompetitorTrend(start, end, timePeriod, 0, 50)
              .then((t) => ({ type: "costco", trend: t?.trend || [] }))
              .catch(() => ({ type: "costco", trend: [] }))
          );
        }

        const results = await Promise.all(promises);

        results.forEach((result) => {
          if (result.type === "walmart") {
            if (result.summary) setSummary(result.summary);
            if (result.trend) {
              setTrend(result.trend);
              setHasMoreData(result.trend.length === 50);
            }
          } else if (result.type === "costco") {
            if (result.summary) setCompetitorSummary(result.summary);
            if (result.trend) setCompetitorTrend(result.trend);
          }
        });

        // Prefetch counts only for DAILY points (weekly/monthly is aggregated later)
        // We still cache by day so tooltips can show counts for daily view reliably.
        const walmartTrend = results.find((r) => r.type === "walmart" && r.trend)?.trend || [];
        if (walmartTrend.length > 0) {
          const countsCache = {};
          const p = walmartTrend.map(async (trendItem) => {
            const counts = await getTweetCountsForDate(trendItem.date);
            if (counts) countsCache[trendItem.date] = counts;
          });

          Promise.all(p).then(() => {
            setTweetCountsCache(countsCache);
          });
        }
      } catch (error) {
        console.error("Failed to load data:", error);
        console.error("Error details:", error.response?.data || error.message);
        setErr(`Failed to load data. Error: ${error.message}`);
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end, timePeriod, selectedBrand]);

  // ---- load aspect data for sentiment drivers ----
  useEffect(() => {
    if (!start || !end) return;

    (async () => {
      try {
        // Load Walmart aspect data
        const walmartData = await getAspectSentimentSplit(start, end, true, true);
        setWalmartAspectData(walmartData);

        // Load Costco aspect data
        try {
          const costcoData = await getCompetitorAspectSentimentSplit(start, end, true, true);
          setCostcoAspectData(costcoData);
        } catch (error) {
          console.error("Failed to load Costco aspect data:", error);
          setCostcoAspectData(null);
        }
      } catch (error) {
        console.error("Failed to load aspect data:", error);
        setWalmartAspectData(null);
      }
    })();
  }, [start, end]);

  // Helper function to get top positive and negative aspects
  const getTopAspects = (aspectData, sentiment, topN = 3) => {
    if (!aspectData?.labels || !aspectData?.counts) return [];

    const labels = aspectData.labels;
    const counts = aspectData.counts[sentiment] || [];
    const aspects = labels
      .map((label, index) => ({
        aspect: label === "others" ? "Others" : label.charAt(0).toUpperCase() + label.slice(1),
        count: counts[index] || 0,
      }))
      .filter((item) => item.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, topN);

    return aspects.map((a) => a.aspect);
  };

  // Build stacked bar chart data for Walmart aspect sentiment breakdown
  const stackedBarData = useMemo(() => {
    if (!walmartAspectData) return { labels: [], datasets: [] };

    const walmartLabels = walmartAspectData?.labels || [];
    const labels = walmartLabels.map((t) => String(t).toUpperCase());

    const wPos = walmartAspectData?.counts?.positive || [];
    const wNeu = walmartAspectData?.counts?.neutral || [];
    const wNeg = walmartAspectData?.counts?.negative || [];

    const wTotals = labels.map((_, i) => (wPos[i] || 0) + (wNeu[i] || 0) + (wNeg[i] || 0));

    return {
      labels,
      datasets: [
        { label: "Positive", sentimentKey: "positive", brand: "walmart", data: wPos, backgroundColor: "#4ade80", stack: "Walmart", totals: wTotals },
        { label: "Neutral", sentimentKey: "neutral", brand: "walmart", data: wNeu, backgroundColor: "#fbbf24", stack: "Walmart", totals: wTotals },
        { label: "Negative", sentimentKey: "negative", brand: "walmart", data: wNeg, backgroundColor: "#f87171", stack: "Walmart", totals: wTotals },
      ],
    };
  }, [walmartAspectData]);

  // Build stacked bar chart data for Costco aspect sentiment breakdown
  const costcoStackedBarData = useMemo(() => {
    if (!costcoAspectData) return { labels: [], datasets: [] };

    const costcoLabels = costcoAspectData?.labels || [];
    const labels = costcoLabels.map((t) => String(t).toUpperCase());

    const cPos = costcoAspectData?.counts?.positive || [];
    const cNeu = costcoAspectData?.counts?.neutral || [];
    const cNeg = costcoAspectData?.counts?.negative || [];

    const cTotals = labels.map((_, i) => (cPos[i] || 0) + (cNeu[i] || 0) + (cNeg[i] || 0));

    return {
      labels,
      datasets: [
        { label: "Positive", sentimentKey: "positive", brand: "costco", data: cPos, backgroundColor: "#4ade80", stack: "Costco", totals: cTotals },
        { label: "Neutral", sentimentKey: "neutral", brand: "costco", data: cNeu, backgroundColor: "#fbbf24", stack: "Costco", totals: cTotals },
        { label: "Negative", sentimentKey: "negative", brand: "costco", data: cNeg, backgroundColor: "#f87171", stack: "Costco", totals: cTotals },
      ],
    };
  }, [costcoAspectData]);

  // Chart options for the bar chart
  const barChartOptions = useMemo(() => {
    return {
      indexAxis: 'y', // Make it a horizontal bar chart
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          bottom: 0,
          left: 10,
          right: 10,
        },
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#64748b",
            font: { size: 11, weight: "bold" },
            generateLabels: function () {
              const mk = (text, fillStyle) => ({
                text,
                fillStyle,
                strokeStyle: fillStyle,
                lineWidth: 2,
                hidden: false,
                datasetIndex: -1,
                fontColor: "#64748b",
                usePointStyle: true,
                pointStyle: "line",
              });
              return [
                mk("Positive", "#4ade80"),
                mk("Neutral", "#fbbf24"),
                mk("Negative", "#f87171"),
              ];
            },
          },
          onClick: (e, legendItem, legend) => {
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
            const v = context?.parsed?.x ?? 0;
            return v > 0;
          },
          color: "white",
          font: { weight: "bold", size: 10 },
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
              const sentiment = ctx.dataset?.label || "";
              const val = ctx.parsed?.x ?? 0;
              return `${sentiment}: ${val.toLocaleString()}`;
            },
            afterLabel: (ctx) => {
              const totals = ctx.dataset?.totals || [];
              const total = totals[ctx.dataIndex] || 0;
              const value = ctx.parsed?.x ?? 0;
              const pct = total > 0 ? Math.round((value / total) * 100) : 0;
              return `Total: ${total.toLocaleString()} • ${pct}%`;
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          beginAtZero: true,
          ticks: {
            callback: (v) => v,
            color: "#64748b",
            font: { size: 10 },
            padding: 8,
          },
          grid: {
            color: "rgba(148, 163, 184, 0.15)",
            lineWidth: 1,
          },
        },
        y: {
          stacked: false,
          grid: { display: false },
          categoryPercentage: 0.9,
          barPercentage: 0.95,
          ticks: {
            color: "#64748b",
            font: { size: 9, weight: "normal" },
            padding: 6,
            autoSkip: false,
          },
        },
      },
    };
  }, []);


  // ---- load more trend data ----
  const loadMoreData = async () => {
    if (loadingMore || !hasMoreData) return;

    try {
      setLoadingMore(true);
      const newOffset = currentOffset + 50;
      const t = await getTrend(start, end, timePeriod, newOffset, 50);
      const newData = t?.trend || [];

      if (newData.length > 0) {
        setTrend((prev) => [...prev, ...newData]);
        setCurrentOffset(newOffset);
        setHasMoreData(newData.length === 50);
      } else {
        setHasMoreData(false);
      }
    } catch (error) {
      console.error("Failed to load more data:", error);
    } finally {
      setLoadingMore(false);
    }
  };

  // ---- load aspect split for selected date (click on chart) ----
  const loadDateAspects = async (actualDateISO) => {
    try {
      setLoadingDateAspects(true);

      const dateObj = new Date(actualDateISO);
      const nextDay = new Date(dateObj);
      nextDay.setDate(dateObj.getDate() + 1);

      const startDate = actualDateISO;
      const endDate = nextDay.toISOString().split("T")[0];

      const data = await getAspectSentimentSplit(startDate, endDate, true, true);
      const transformedData = transformAspectData(data);

      setSelectedDateModal({
        isOpen: true,
        date: actualDateISO,
        formattedDate: formatDate(actualDateISO),
        data: transformedData,
      });
    } catch (error) {
      console.error("Failed to load date aspects:", error);
      console.error("Error details:", error.response?.data || error.message);
      setErr(`Failed to load aspect breakdown for selected date: ${error.message}`);
    } finally {
      setLoadingDateAspects(false);
    }
  };

  // ---- load aspect split modal (positive/neutral/negative button) ----
  const loadAspectSplit = async (sentiment) => {
    try {
      setLoadingAspectSplit(true);

      let walmartData = null;
      let costcoData = null;

      if (selectedBrand === "walmart" || selectedBrand === "all") {
        try {
          walmartData = await getAspectSentimentSplit(start, end, true, true);
        } catch (error) {
          console.error("Failed to load Walmart aspects:", error);
        }
      }

      if (selectedBrand === "costco" || selectedBrand === "all") {
        try {
          costcoData = await getCompetitorAspectSentimentSplit(start, end, true, true);
        } catch (error) {
          console.error("Failed to load Costco aspects:", error);
        }
      }

      if (selectedBrand === "all") {
        const walmartFiltered =
          walmartData?.labels && walmartData?.counts && walmartData?.percent
            ? walmartData.labels.map((aspect, index) => ({
                aspect:
                  aspect === "others"
                    ? "Others"
                    : aspect.charAt(0).toUpperCase() + aspect.slice(1),
                count: walmartData.counts[sentiment.toLowerCase()]?.[index] || 0,
                percentage: walmartData.percent[sentiment.toLowerCase()]?.[index] || 0,
                sentiment: sentiment.toLowerCase(),
                brand: "Walmart",
              }))
            : [];

        const costcoFiltered =
          costcoData?.labels && costcoData?.counts && costcoData?.percent
            ? costcoData.labels.map((aspect, index) => ({
                aspect:
                  aspect === "others"
                    ? "Others"
                    : aspect.charAt(0).toUpperCase() + aspect.slice(1),
                count: costcoData.counts[sentiment.toLowerCase()]?.[index] || 0,
                percentage: costcoData.percent[sentiment.toLowerCase()]?.[index] || 0,
                sentiment: sentiment.toLowerCase(),
                brand: "Costco",
              }))
            : [];

        setAspectSplitModal({
          isOpen: true,
          sentiment: sentiment,
          data: { walmart: walmartFiltered, costco: costcoFiltered },
          showByBrand: true,
        });
      } else {
        const data = selectedBrand === "costco" ? costcoData : walmartData;
        let filteredData = [];

        if (data?.labels && data?.counts && data?.percent) {
          const labels = data.labels;
          const countsArr = data.counts[sentiment.toLowerCase()] || [];
          const percentagesArr = data.percent[sentiment.toLowerCase()] || [];

          filteredData = labels.map((aspect, index) => ({
            aspect:
              aspect === "others"
                ? "Others"
                : aspect.charAt(0).toUpperCase() + aspect.slice(1),
            count: countsArr[index] || 0,
            percentage: percentagesArr[index] || 0,
            sentiment: sentiment.toLowerCase(),
          }));
        }

        setAspectSplitModal({
          isOpen: true,
          sentiment: sentiment,
          data: filteredData,
          showByBrand: false,
        });
      }
    } catch (error) {
      console.error("Failed to load aspect split:", error);
      setErr("Failed to load aspect breakdown");
    } finally {
      setLoadingAspectSplit(false);
    }
  };

  /**
   * ✅ NEW: Build chart datasets with POINT OBJECTS
   * Each point contains:
   *  - x: ISO date
   *  - y: percentage
   *  - count: tweet count for that sentiment on that date (from cache if available)
   *  - total: total tweets on that date
   */
  const trendLineData = useMemo(() => {
    const datasets = [];

    const getCountsForISODate = (isoDate) => tweetCountsCache?.[isoDate] || null;

    // When timePeriod is weekly/monthly, you are averaging percentages.
    // Counts in tooltips for aggregated periods will be 0 unless you also aggregate counts.
    // For now: show correct counts in DAILY view (your main issue).
    const makePointsDaily = (arr, sentimentKey) =>
      (arr || []).map((p) => {
        const c = getCountsForISODate(p.date);
        return {
          x: p.date,
          y: Number(p?.[sentimentKey] ?? 0),
          count: Number(c?.[sentimentKey] ?? 0),
          total: Number(c?.totalTweets ?? 0),
        };
      });

    // Basic aggregation for % values (keeps your behavior)
    const aggregateData = (data, period) => {
      const grouped = {};

      data.forEach((item) => {
        const date = new Date(item.date);
        let key;

        switch (period) {
          case "weekly": {
            const weekStart = new Date(date);
            weekStart.setDate(date.getDate() - date.getDay() + 1);
            key = weekStart.toISOString().split("T")[0];
            break;
          }
          case "monthly":
            key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-01`;
            break;
          default:
            key = item.date;
        }

        if (!grouped[key]) {
          grouped[key] = { date: key, positive: 0, neutral: 0, negative: 0, count: 0 };
        }

        grouped[key].positive += item.positive || 0;
        grouped[key].neutral += item.neutral || 0;
        grouped[key].negative += item.negative || 0;
        grouped[key].count += 1;
      });

      return Object.values(grouped)
        .map((item) => ({
          date: item.date,
          positive: item.count > 0 ? item.positive / item.count : 0,
          neutral: item.count > 0 ? item.neutral / item.count : 0,
          negative: item.count > 0 ? item.negative / item.count : 0,
        }))
        .sort((a, b) => new Date(a.date) - new Date(b.date));
    };

    const walmartAgg =
      trend && trend.length > 0
        ? timePeriod === "daily"
          ? [...trend].sort((a, b) => new Date(a.date) - new Date(b.date))
          : aggregateData(trend, timePeriod)
        : [];

    const costcoAgg =
      competitorTrend && competitorTrend.length > 0
        ? timePeriod === "daily"
          ? [...competitorTrend].sort((a, b) => new Date(a.date) - new Date(b.date))
          : aggregateData(competitorTrend, timePeriod)
        : [];

    // If "all", we don't need labels array for the chart because we use x values (time scale by category)
    // But Chart.js with "chart.js/auto" will handle category from x strings; still fine.

    // Helper function to sort data points by date
    const sortByDate = (points) => {
      return [...points].sort((a, b) => new Date(a.x) - new Date(b.x));
    };

    if ((selectedBrand === "walmart" || selectedBrand === "all") && walmartAgg.length > 0) {
      const wPointsPos = sortByDate(timePeriod === "daily" ? makePointsDaily(walmartAgg, "positive") : walmartAgg.map(p => ({ x: p.date, y: Number(p.positive ?? 0), count: 0, total: 0 })));
      const wPointsNeu = sortByDate(timePeriod === "daily" ? makePointsDaily(walmartAgg, "neutral") : walmartAgg.map(p => ({ x: p.date, y: Number(p.neutral ?? 0), count: 0, total: 0 })));
      const wPointsNeg = sortByDate(timePeriod === "daily" ? makePointsDaily(walmartAgg, "negative") : walmartAgg.map(p => ({ x: p.date, y: Number(p.negative ?? 0), count: 0, total: 0 })));

      datasets.push(
        { label: "Walmart % Positive", data: wPointsPos, borderColor: "#22c55e", fill: false, borderDash: [], tension: 0.35 },
        { label: "Walmart % Neutral", data: wPointsNeu, borderColor: "#facc15", fill: false, borderDash: [], tension: 0.35 },
        { label: "Walmart % Negative", data: wPointsNeg, borderColor: "#ef4444", fill: false, borderDash: [], tension: 0.35 }
      );
    }

    if ((selectedBrand === "costco" || selectedBrand === "all") && costcoAgg.length > 0) {
      const cPointsPos = sortByDate(timePeriod === "daily" ? makePointsDaily(costcoAgg, "positive") : costcoAgg.map(p => ({ x: p.date, y: Number(p.positive ?? 0), count: 0, total: 0 })));
      const cPointsNeu = sortByDate(timePeriod === "daily" ? makePointsDaily(costcoAgg, "neutral") : costcoAgg.map(p => ({ x: p.date, y: Number(p.neutral ?? 0), count: 0, total: 0 })));
      const cPointsNeg = sortByDate(timePeriod === "daily" ? makePointsDaily(costcoAgg, "negative") : costcoAgg.map(p => ({ x: p.date, y: Number(p.negative ?? 0), count: 0, total: 0 })));

      datasets.push(
        { label: "Costco % Positive", data: cPointsPos, borderColor: "#22c55e", fill: false, borderDash: [5, 5], tension: 0.35 },
        { label: "Costco % Neutral", data: cPointsNeu, borderColor: "#fbbf24", fill: false, borderDash: [5, 5], tension: 0.35 },
        { label: "Costco % Negative", data: cPointsNeg, borderColor: "#f87171", fill: false, borderDash: [5, 5], tension: 0.35 }
      );
    }

    // Collect all unique dates from all datasets and sort them
    const allDates = new Set();
    datasets.forEach(dataset => {
      dataset.data.forEach(point => {
        if (point.x) allDates.add(point.x);
      });
    });
    const sortedLabels = Array.from(allDates).sort((a, b) => new Date(a) - new Date(b));

    return { datasets, labels: sortedLabels };
  }, [trend, competitorTrend, timePeriod, selectedBrand, tweetCountsCache]);

  // Determine which summary to use based on selected brand
  const activeSummary = selectedBrand === "costco" ? competitorSummary : summary;
  const total = activeSummary?.total || 0;
  const pct = activeSummary?.percent || { positive: 0, negative: 0, neutral: 0 };
  const counts = activeSummary?.counts || { positive: 0, negative: 0, neutral: 0 };

  // For "all", combine both summaries
  const combinedTotal =
    selectedBrand === "all" ? (summary?.total || 0) + (competitorSummary?.total || 0) : total;

  const combinedCounts =
    selectedBrand === "all"
      ? {
          positive: (summary?.counts?.positive || 0) + (competitorSummary?.counts?.positive || 0),
          neutral: (summary?.counts?.neutral || 0) + (competitorSummary?.counts?.neutral || 0),
          negative: (summary?.counts?.negative || 0) + (competitorSummary?.counts?.negative || 0),
        }
      : counts;

  const combinedPct =
    selectedBrand === "all" && combinedTotal > 0
      ? {
          positive: parseFloat(((combinedCounts.positive / combinedTotal) * 100).toFixed(2)),
          neutral: parseFloat(((combinedCounts.neutral / combinedTotal) * 100).toFixed(2)),
          negative: parseFloat(((combinedCounts.negative / combinedTotal) * 100).toFixed(2)),
        }
      : pct;

  const displayTotal = selectedBrand === "all" ? combinedTotal : total;
  const displayCounts = selectedBrand === "all" ? combinedCounts : counts;
  const displayPct = selectedBrand === "all" ? combinedPct : pct;

  return (
    <div className="h-screen bg-slate-900 overflow-hidden">
      <div className="h-full flex flex-col px-4 py-2 max-w-7xl mx-auto">
        {/* Error & Loading States */}
        {err && (
          <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-xl p-4 mb-4">
            <div className="flex items-center space-x-3">
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                />
              </svg>
              <span className="font-semibold">{err}</span>
            </div>
          </div>
        )}

        {(loading || metaLoading) && (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center space-y-4">
              <div className="w-8 h-8 border-3 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-slate-300 font-semibold text-lg">
                {metaLoading ? "Loading metadata..." : "Loading analytics..."}
              </span>
            </div>
          </div>
        )}

        {!loading && !metaLoading && (
          <>
            {/* Analytics Dashboard Header - Dark Theme */}
            <div className="mb-2 pt-2 pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center shadow-md">
                    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5a2 2 0 012-2h4a2 2 0 012 2v2H8V5z" />
                    </svg>
                  </div>
                  <div className="px-2 py-1">
                    <h1 className="text-xl font-semibold text-white">Analytics Dashboard</h1>
                    <p className="text-slate-400 text-sm">
                      This dashboard analyzes public social media conversations to compare sentiment, aspects, and emerging themes across major retail brands (Walmart vs Costco).
                    </p>
                    {/* Date Range Controls */}
                    <div className="flex items-center space-x-2 mt-3">
                      <div className="flex items-center space-x-1 bg-slate-800 rounded-md border border-slate-700 px-2 py-1 shadow-sm">
                        <span className="text-xs text-slate-300 font-medium">Date:</span>
                        <input
                          type="date"
                          value={start || ""}
                          min={meta?.min || ""}
                          max={end || meta?.max || ""}
                          onChange={(e) => setStart(iso(e.target.value))}
                          className="px-1 py-0.5 bg-slate-800 border border-slate-700 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                        />
                        <span className="text-slate-500 text-xs">to</span>
                        <input
                          type="date"
                          value={end || ""}
                          min={start || meta?.min || ""}
                          max={meta?.max || ""}
                          onChange={(e) => setEnd(iso(e.target.value))}
                          className="px-1 py-0.5 bg-slate-800 border border-slate-700 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                        />
                      </div>
                    </div>
                  </div>
                </div>

            </div>

                {/* Executive Insight Strip */}
                <div className="mt-3 grid grid-cols-1 lg:grid-cols-3 gap-2">
                  <div className="bg-slate-800 rounded-xl border border-slate-700 p-3 shadow-sm">
                    <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide mb-2">
                      Executive Snapshot
                    </p>
                    <div className="space-y-2">
                      {/* Walmart Section */}
                      <div className="border-b border-slate-700 pb-2">
                        <p className="text-xs font-semibold text-slate-200 mb-1">Walmart</p>
                        <p className="text-xs text-slate-100 mb-1">
                          {(summary?.total || 0).toLocaleString()} tweets analyzed
                        </p>
                        <p className="text-[10px] text-slate-400">
                          Positive:{" "}
                          <span className="text-emerald-600 font-semibold">
                            {summary?.percent?.positive ? summary.percent.positive.toFixed(2) : "0.00"}%
                          </span>{" "}
                          · Neutral:{" "}
                          <span className="text-yellow-500 font-semibold">
                            {summary?.percent?.neutral ? summary.percent.neutral.toFixed(2) : "0.00"}%
                          </span>{" "}
                          · Negative:{" "}
                          <span className="text-red-500 font-semibold">
                            {summary?.percent?.negative ? summary.percent.negative.toFixed(2) : "0.00"}%
                          </span>
                        </p>
                      </div>
                      {/* Costco Section */}
                      <div>
                        <p className="text-xs font-semibold text-slate-200 mb-1">Costco</p>
                        <p className="text-xs text-slate-100 mb-1">
                          {(competitorSummary?.total || 0).toLocaleString()} tweets analyzed
                        </p>
                        <p className="text-[10px] text-slate-400">
                          Positive:{" "}
                          <span className="text-emerald-600 font-semibold">
                            {competitorSummary?.percent?.positive ? competitorSummary.percent.positive.toFixed(2) : "0.00"}%
                          </span>{" "}
                          · Neutral:{" "}
                          <span className="text-yellow-500 font-semibold">
                            {competitorSummary?.percent?.neutral ? competitorSummary.percent.neutral.toFixed(2) : "0.00"}%
                          </span>{" "}
                          · Negative:{" "}
                          <span className="text-red-500 font-semibold">
                            {competitorSummary?.percent?.negative ? competitorSummary.percent.negative.toFixed(2) : "0.00"}%
                          </span>
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-slate-800 rounded-xl border border-slate-700 p-3 shadow-sm">
                    <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide mb-2">
                      Walmart Aspect Sentiment Breakdown
                    </p>
                    {walmartAspectData ? (
                      <div className="h-40" style={{ position: "relative" }}>
                        {stackedBarData?.labels?.length > 0 && stackedBarData?.datasets?.length > 0 ? (
                          <Bar
                            data={stackedBarData}
                            options={barChartOptions}
                            plugins={[ChartDataLabels]}
                          />
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="text-xs text-slate-500">No data available</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-40">
                        <div className="flex flex-col items-center space-y-2">
                          <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                          <p className="text-xs text-slate-500">Loading chart data...</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="bg-slate-800 rounded-xl border border-slate-700 p-3 shadow-sm">
                    <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide mb-2">
                      Costco Aspect Sentiment Breakdown
                    </p>
                    {costcoAspectData ? (
                      <div className="h-40" style={{ position: "relative" }}>
                        {costcoStackedBarData?.labels?.length > 0 && costcoStackedBarData?.datasets?.length > 0 ? (
                          <Bar
                            data={costcoStackedBarData}
                            options={barChartOptions}
                            plugins={[ChartDataLabels]}
                          />
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p className="text-xs text-slate-400">No data available</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-40">
                        <div className="flex flex-col items-center space-y-2">
                          <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                          <p className="text-xs text-slate-400">Loading chart data...</p>
                        </div>
                      </div>
                    )}
                  </div>
                    </div>
                </div>

            {/* Main Analytics Row: Aspect Panel + Trend Chart + Date Drill-down */}
            <div className="flex-1 flex flex-row gap-3 min-h-0">
              {/* Middle Side - Aspect Breakdown Card */}
              {aspectSplitModal.isOpen && (
                <div className="w-[280px] flex flex-col min-h-0 flex-shrink-0">
                  <div
                    className="bg-slate-800 rounded-md p-4 border border-slate-700 shadow-sm flex flex-col min-h-0"
                    style={{ height: "320px" }}
                  >
                    <div className="flex items-center justify-between mb-4 flex-shrink-0">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                          <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                            />
                          </svg>
                        </div>
                        <div>
                          <h2 className="text-sm font-bold text-white">
                            {aspectSplitModal.sentiment?.charAt(0).toUpperCase() + aspectSplitModal.sentiment?.slice(1)} Aspects
                          </h2>
                          <p className="text-slate-400 text-xs">Aspect breakdown</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setAspectSplitModal({ isOpen: false, sentiment: null, data: null, showByBrand: false })}
                        className="text-slate-400 hover:text-white transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    <div className="flex-1 overflow-y-auto">
                      {aspectSplitModal.showByBrand ? (
                        <div className="space-y-6">
                          {/* Walmart Section */}
                          {aspectSplitModal.data.walmart && aspectSplitModal.data.walmart.length > 0 && (
                            <div className="space-y-3">
                              <div className="flex items-center space-x-2 mb-3">
                                <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                                <h3 className="text-sm font-bold text-white">Walmart</h3>
                              </div>
                              {aspectSplitModal.data.walmart.map((aspect, index) => {
                                const totalSentimentTweets = aspectSplitModal.data.walmart.reduce((sum, a) => sum + (a.count || 0), 0);
                                const aspectCount = aspect.count || 0;
                                const percentageOfTotal = totalSentimentTweets > 0 ? (aspectCount / totalSentimentTweets) * 100 : 0;

                                return (
                                  <div key={`walmart-${index}`} className="space-y-1">
                                    <div className="flex items-center justify-between">
                                      <h4 className="font-medium text-white text-sm">{aspect.aspect || `Aspect ${index + 1}`}</h4>
                                      <div className="flex items-center space-x-2">
                                        <span className="text-xs text-slate-400">{aspectCount} tweets</span>
                                        <span className="text-sm font-bold text-white">{percentageOfTotal.toFixed(1)}%</span>
                                      </div>
                                    </div>
                                    <div className="w-full bg-slate-600/50 rounded-full h-3 overflow-hidden">
                                      <div
                                        className={`h-3 rounded-full transition-all duration-500 ${
                                          aspectSplitModal.sentiment === "positive"
                                            ? "bg-gradient-to-r from-green-500 to-emerald-500"
                                            : aspectSplitModal.sentiment === "negative"
                                            ? "bg-gradient-to-r from-red-500 to-pink-500"
                                            : "bg-gradient-to-r from-yellow-500 to-amber-500"
                                        }`}
                                        style={{ width: `${Math.min(percentageOfTotal, 100)}%`, maxWidth: "100%" }}
                                      ></div>
                                    </div>
                                  </div>
                                );
                              })}
                              <div className="text-xs text-slate-400 pt-2 border-t border-slate-600/50">
                                Total: {aspectSplitModal.data.walmart.reduce((sum, a) => sum + (a.count || 0), 0)} tweets
                              </div>
                            </div>
                          )}

                          {/* Costco Section */}
                          {aspectSplitModal.data.costco && aspectSplitModal.data.costco.length > 0 && (
                            <div className="space-y-3">
                              <div className="flex items-center space-x-2 mb-3">
                                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                                <h3 className="text-sm font-bold text-white">Costco</h3>
                              </div>
                              {aspectSplitModal.data.costco.map((aspect, index) => {
                                const totalSentimentTweets = aspectSplitModal.data.costco.reduce((sum, a) => sum + (a.count || 0), 0);
                                const aspectCount = aspect.count || 0;
                                const percentageOfTotal = totalSentimentTweets > 0 ? (aspectCount / totalSentimentTweets) * 100 : 0;

                                return (
                                  <div key={`costco-${index}`} className="space-y-1">
                                    <div className="flex items-center justify-between">
                                      <h4 className="font-medium text-white text-sm">{aspect.aspect || `Aspect ${index + 1}`}</h4>
                                      <div className="flex items-center space-x-2">
                                        <span className="text-xs text-slate-400">{aspectCount} tweets</span>
                                        <span className="text-sm font-bold text-white">{percentageOfTotal.toFixed(1)}%</span>
                                      </div>
                                    </div>
                                    <div className="w-full bg-slate-600/50 rounded-full h-3 overflow-hidden">
                                      <div
                                        className={`h-3 rounded-full transition-all duration-500 ${
                                          aspectSplitModal.sentiment === "positive"
                                            ? "bg-gradient-to-r from-green-500 to-emerald-500"
                                            : aspectSplitModal.sentiment === "negative"
                                            ? "bg-gradient-to-r from-red-500 to-pink-500"
                                            : "bg-gradient-to-r from-yellow-500 to-amber-500"
                                        }`}
                                        style={{ width: `${Math.min(percentageOfTotal, 100)}%`, maxWidth: "100%" }}
                                      ></div>
                                    </div>
                                  </div>
                                );
                              })}
                              <div className="text-xs text-slate-400 pt-2 border-t border-slate-600/50">
                                Total: {aspectSplitModal.data.costco.reduce((sum, a) => sum + (a.count || 0), 0)} tweets
                              </div>
                            </div>
                          )}
                        </div>
                      ) : aspectSplitModal.data && Array.isArray(aspectSplitModal.data) && aspectSplitModal.data.length > 0 ? (
                        <div className="space-y-3">
                          {aspectSplitModal.data.map((aspect, index) => {
                            const totalSentimentTweets = aspectSplitModal.data.reduce(
                              (sum, a) => sum + (a.count || a.tweet_count || a.total || 0),
                              0
                            );
                            const aspectCount = aspect.count || aspect.tweet_count || aspect.total || 0;
                            const percentageOfTotal = totalSentimentTweets > 0 ? (aspectCount / totalSentimentTweets) * 100 : 0;

                            return (
                              <div key={index} className="space-y-1">
                                <div className="flex items-center justify-between">
                                  <h4 className="font-medium text-white text-sm">
                                    {aspect.aspect || aspect.name || aspect.label || `Aspect ${index + 1}`}
                                  </h4>
                                  <div className="flex items-center space-x-2">
                                    <span className="text-xs text-slate-400">{aspectCount} tweets</span>
                                    <span className="text-sm font-bold text-white">{percentageOfTotal.toFixed(1)}%</span>
                                  </div>
                                </div>

                                <div className="w-full bg-slate-600/50 rounded-full h-3 overflow-hidden">
                                  <div
                                    className={`h-3 rounded-full transition-all duration-500 ${
                                      aspectSplitModal.sentiment === "positive"
                                        ? "bg-gradient-to-r from-green-500 to-emerald-500"
                                        : aspectSplitModal.sentiment === "negative"
                                        ? "bg-gradient-to-r from-red-500 to-pink-500"
                                        : "bg-gradient-to-r from-yellow-500 to-amber-500"
                                    }`}
                                    style={{ width: `${Math.min(percentageOfTotal, 100)}%`, maxWidth: "100%" }}
                                  ></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-center py-8">
                          <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-3">
                            <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                              />
                            </svg>
                          </div>
                          <h4 className="text-sm font-semibold text-white mb-2">No Aspect Data Found</h4>
                          <p className="text-slate-400 text-xs">
                            No aspect breakdown available for {aspectSplitModal.sentiment} sentiment.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Summary */}
                    {aspectSplitModal.data &&
                      !aspectSplitModal.showByBrand &&
                      Array.isArray(aspectSplitModal.data) &&
                      aspectSplitModal.data.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-slate-600/50 flex-shrink-0">
                          <div className="flex items-center justify-between text-xs text-slate-400">
                            <span>Total {aspectSplitModal.sentiment} tweets analyzed</span>
                            <span>
                              {aspectSplitModal.data.reduce(
                                (sum, a) => sum + (a.count || a.tweet_count || a.total || 0),
                                0
                              )}{" "}
                              tweets
                            </span>
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              )}

              {/* Chart and Date Aspect Sidebar Container */}
              <div className="flex flex-row min-h-0 gap-2 flex-1">
                {/* Chart Section */}
                <div className={`flex flex-col min-h-0 ${selectedDateModal.isOpen ? "flex-1" : "w-full"}`}>
                  <div
                    className="bg-slate-800 rounded-md p-1 border border-slate-700 shadow-sm flex flex-col min-h-0"
                    style={{ height: "480px" }}
                  >
                    <div className="flex items-center justify-between mb-1 flex-shrink-0">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                          <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                            />
                          </svg>
                        </div>
                        <div>
                          <h2 className="text-sm font-bold text-white">Sentiment Trend</h2>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        {/* Brand Selector */}
                        <div className="flex items-center space-x-1 bg-slate-700 rounded-md border border-slate-600 px-2 py-1 shadow-sm">
                          <span className="text-xs text-slate-300 font-medium">Brand:</span>
                          <select
                            value={selectedBrand}
                            onChange={(e) => setSelectedBrand(e.target.value)}
                            className="px-1 py-0.5 bg-slate-700 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                          >
                            <option value="walmart">Walmart</option>
                            <option value="costco">Costco</option>
                            <option value="all">All</option>
                          </select>
                        </div>
                        {/* View Selector */}
                        <div className="flex items-center space-x-1 bg-slate-700 rounded-md border border-slate-600 px-2 py-1 shadow-sm">
                          <span className="text-xs text-slate-300 font-medium">View:</span>
                          <select
                            value={timePeriod}
                            onChange={(e) => setTimePeriod(e.target.value)}
                            className="px-1 py-0.5 bg-slate-700 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all"
                          >
                            <option value="daily">Daily</option>
                            <option value="weekly">Weekly</option>
                            <option value="monthly">Monthly</option>
                          </select>
                        </div>
                        {loadingMore && (
                          <div className="flex items-center space-x-1 text-xs text-emerald-400">
                            <div className="w-2 h-2 border border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                            <span>Loading...</span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div
                      className="flex-1 bg-slate-900/30 rounded p-0.5 border border-slate-600/20 min-h-0 overflow-x-auto overflow-y-hidden"
                      onScroll={(e) => {
                        const { scrollLeft, scrollWidth, clientWidth } = e.target;
                        if (scrollLeft + clientWidth >= scrollWidth * 0.8 && hasMoreData && !loadingMore) {
                          loadMoreData();
                        }
                      }}
                    >
                      <Line
                        data={trendLineData}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          layout: { padding: { right: hasMoreData ? 20 : 0 } },
                          onClick: (event, elements, chart) => {
                            if (!elements?.length) return;
                            const el = elements[0];
                            const ds = chart.data.datasets[el.datasetIndex];
                            const point = ds.data[el.index];
                            const actualISO = point?.x; // ✅ always correct
                            if (actualISO) loadDateAspects(actualISO);
                          },
                          interaction: { intersect: false, mode: "index" },
                          plugins: {
                            legend: {
                              position: "top",
                              labels: {
                                color: "#e2e8f0",
                                font: { size: 9, weight: "bold" },
                                padding: 8,

                                // show line samples in legend
                                usePointStyle: true,
                                pointStyle: "line",

                                // ✅ Make legend reflect dataset.borderDash by mapping it to legend's lineDash
                                generateLabels: (chart) => {
                                  const original =
                                    Chart.defaults.plugins.legend.labels.generateLabels;
                                  const labels = original(chart);

                                  labels.forEach((l) => {
                                    const ds = chart.data.datasets[l.datasetIndex];

                                    // ensure correct colors
                                    l.strokeStyle = ds.borderColor || l.strokeStyle;
                                    l.fillStyle = ds.borderColor || l.fillStyle;

                                    // IMPORTANT: legend uses `lineDash`, not `borderDash`
                                    l.lineDash = ds.borderDash || [];
                                    l.lineWidth = ds.borderWidth ?? 2;

                                    // optional: make the legend line a bit longer/thicker
                                    l.pointStyleWidth = 40;
                                  });

                                  return labels;
                                },
                              },
                            },
                            tooltip: {
                              backgroundColor: "rgba(15, 23, 42, 0.95)",
                              titleColor: "#f1f5f9",
                              bodyColor: "#cbd5e1",
                              borderColor: "rgba(148, 163, 184, 0.2)",
                              borderWidth: 1,
                              cornerRadius: 8,
                              displayColors: true,
                              padding: 12,
                              callbacks: {
                                title: (items) => {
                                  const raw = items?.[0]?.raw;
                                  const d = raw?.x;
                                  const totalTweets = Number(raw?.total ?? 0);
                                  const pretty = d ? formatDate(d) : "";
                                  return totalTweets
                                    ? `Date: ${pretty} - Total: ${totalTweets} tweets`
                                    : `Date: ${pretty}`;
                                },
                                label: (ctx) => {
                                  const label = ctx.dataset?.label || "";
                                  const pct = Number(ctx.raw?.y ?? 0);
                                  const count = Number(ctx.raw?.count ?? 0);
                                  return `${label}: ${count} tweets (${pct.toFixed(1)}%)`;
                                },
                                afterBody: () => "Click for detailed breakdown",
                              },
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              max: 80,
                              grid: { color: "rgba(148, 163, 184, 0.1)", drawBorder: false },
                              ticks: { color: "#94a3b8", font: { size: 8 }, padding: 2 },
                            },
                            x: {
                              type: "category",
                              labels: trendLineData.labels || [],
                              grid: { display: false },
                              ticks: {
                                color: "#94a3b8",
                                font: { size: 8 },
                                padding: 2,
                                maxRotation: 45,
                                minRotation: 45,
                              },
                            },
                          },
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Date Aspect Breakdown Sidebar */}
                {selectedDateModal.isOpen ? (
                  <div className="fixed top-0 right-0 w-80 h-screen bg-slate-800/95 backdrop-blur-sm border-l border-slate-600/50 z-50 flex flex-col shadow-2xl">
                    <div className="flex flex-col h-full p-4">
                      <div className="flex items-center justify-between mb-4 flex-shrink-0">
                        <div className="flex items-center space-x-2">
                          <div className="w-6 h-6 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-md flex items-center justify-center">
                            <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                              />
                            </svg>
                          </div>
                          <div>
                            <h3 className="text-sm font-bold text-white">Date Aspect Breakdown</h3>
                            <p className="text-slate-400 text-xs">
                              Date: {selectedDateModal.formattedDate || selectedDateModal.date}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => setSelectedDateModal({ isOpen: false, date: null, formattedDate: null, data: null })}
                          className="w-6 h-6 bg-slate-700/50 hover:bg-slate-600/50 rounded-md flex items-center justify-center transition-colors"
                        >
                          <svg className="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>

                      <div className="flex-1 bg-slate-900/30 rounded p-2 border border-slate-600/20 min-h-0 overflow-y-auto">
                        {loadingDateAspects ? (
                          <div className="flex items-center justify-center h-full">
                            <div className="flex flex-col items-center space-y-3">
                              <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                              <span className="text-slate-300 text-sm">Loading aspect data...</span>
                            </div>
                          </div>
                        ) : selectedDateModal.data ? (
                          <div className="space-y-4">
                            {/* Total Tweets Summary */}
                            <div className="bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 rounded-lg p-4 border border-emerald-500/30">
                              <div className="flex items-center justify-between">
                                <h4 className="text-lg font-bold text-white">Total Tweets</h4>
                                <span className="text-2xl font-bold text-emerald-400">{selectedDateModal.data.totalTweets || 0}</span>
                              </div>
                              <p className="text-sm text-slate-300 mt-1">
                                for {selectedDateModal.formattedDate || selectedDateModal.date}
                              </p>
                            </div>

                            {/* Overall Aspect Breakdown */}
                            <div className="space-y-3">
                              <h4 className="text-sm font-semibold text-white">Overall Aspect Breakdown</h4>
                              {selectedDateModal.data.aspectBreakdown && selectedDateModal.data.aspectBreakdown.length > 0 && (
                                <div className="space-y-2">
                                  {selectedDateModal.data.aspectBreakdown.map((aspect, index) => (
                                    <div key={index} className="space-y-1">
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm text-slate-300 capitalize">{aspect.aspect}</span>
                                        <div className="flex items-center space-x-2">
                                          <span className="text-xs text-slate-400">{aspect.count} tweets</span>
                                          <span className="text-sm font-semibold text-white">{aspect.percentage?.toFixed(1) || 0}%</span>
                                        </div>
                                      </div>
                                      <div className="w-full bg-slate-700/50 rounded-full h-2">
                                        <div
                                          className="bg-gradient-to-r from-emerald-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                                          style={{ width: `${Math.min(aspect.percentage || 0, 100)}%` }}
                                        ></div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>

                            {/* Sentiment Breakdown */}
                            <div className="space-y-3">
                              <h4 className="text-sm font-semibold text-white">Sentiment-Based Aspect Breakdown</h4>

                              {/* Positive */}
                              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center space-x-2">
                                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                    <div>
                                    <span className="text-sm font-semibold text-green-400">Positive</span>
                                      <p className="text-[10px] text-slate-400 mt-0.5">Drivers of favorable sentiment</p>
                                    </div>
                                  </div>
                                  <span className="text-sm font-bold text-white">
                                    {selectedDateModal.data.sentimentBreakdown?.positive?.total || 0}
                                    {selectedDateModal.data.totalTweets > 0 && (
                                      <span className="text-xs text-green-300 ml-1">
                                        (
                                        {(
                                          ((selectedDateModal.data.sentimentBreakdown?.positive?.total || 0) /
                                            selectedDateModal.data.totalTweets) *
                                          100
                                        ).toFixed(1)}
                                        %)
                                      </span>
                                    )}
                                  </span>
                                </div>
                                {selectedDateModal.data.sentimentBreakdown?.positive?.aspects &&
                                  selectedDateModal.data.sentimentBreakdown.positive.aspects.length > 0 && (
                                    <div className="space-y-2">
                                      {selectedDateModal.data.sentimentBreakdown.positive.aspects.map((aspect, index) => (
                                        <div key={index} className="flex items-center justify-between text-xs">
                                          <span className="text-slate-300 capitalize">{aspect.aspect}</span>
                                          <span className="text-green-300">
                                            {aspect.count} ({aspect.percentage?.toFixed(1)}%)
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                              </div>

                              {/* Neutral */}
                              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center space-x-2">
                                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                                    <div>
                                    <span className="text-sm font-semibold text-yellow-400">Neutral</span>
                                      <p className="text-[10px] text-slate-400 mt-0.5">Informational or mixed mentions</p>
                                    </div>
                                  </div>
                                  <span className="text-sm font-bold text-white">
                                    {selectedDateModal.data.sentimentBreakdown?.neutral?.total || 0}
                                    {selectedDateModal.data.totalTweets > 0 && (
                                      <span className="text-xs text-yellow-300 ml-1">
                                        (
                                        {(
                                          ((selectedDateModal.data.sentimentBreakdown?.neutral?.total || 0) /
                                            selectedDateModal.data.totalTweets) *
                                          100
                                        ).toFixed(1)}
                                        %)
                                      </span>
                                    )}
                                  </span>
                                </div>
                                {selectedDateModal.data.sentimentBreakdown?.neutral?.aspects &&
                                  selectedDateModal.data.sentimentBreakdown.neutral.aspects.length > 0 && (
                                    <div className="space-y-2">
                                      {selectedDateModal.data.sentimentBreakdown.neutral.aspects.map((aspect, index) => (
                                        <div key={index} className="flex items-center justify-between text-xs">
                                          <span className="text-slate-300 capitalize">{aspect.aspect}</span>
                                          <span className="text-yellow-300">
                                            {aspect.count} ({aspect.percentage?.toFixed(1)}%)
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                              </div>

                              {/* Negative */}
                              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center space-x-2">
                                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                    <div>
                                    <span className="text-sm font-semibold text-red-400">Negative</span>
                                      <p className="text-[10px] text-slate-400 mt-0.5">Risk signals and complaints</p>
                                    </div>
                                  </div>
                                  <span className="text-sm font-bold text-white">
                                    {selectedDateModal.data.sentimentBreakdown?.negative?.total || 0}
                                    {selectedDateModal.data.totalTweets > 0 && (
                                      <span className="text-xs text-red-300 ml-1">
                                        (
                                        {(
                                          ((selectedDateModal.data.sentimentBreakdown?.negative?.total || 0) /
                                            selectedDateModal.data.totalTweets) *
                                          100
                                        ).toFixed(1)}
                                        %)
                                      </span>
                                    )}
                                  </span>
                                </div>
                                {selectedDateModal.data.sentimentBreakdown?.negative?.aspects &&
                                  selectedDateModal.data.sentimentBreakdown.negative.aspects.length > 0 && (
                                    <div className="space-y-2">
                                      {selectedDateModal.data.sentimentBreakdown.negative.aspects.map((aspect, index) => (
                                        <div key={index} className="flex items-center justify-between text-xs">
                                          <span className="text-slate-300 capitalize">{aspect.aspect}</span>
                                          <span className="text-red-300">
                                            {aspect.count} ({aspect.percentage?.toFixed(1)}%)
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-8">
                            <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-3">
                              <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                                />
                              </svg>
                            </div>
                            <h4 className="text-sm font-semibold text-white mb-2">No Data Available</h4>
                            <p className="text-slate-400 text-xs">No aspect data found for the selected date.</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}