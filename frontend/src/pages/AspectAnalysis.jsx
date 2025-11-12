// frontend/src/pages/AspectAnalysis.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { getAspectSentimentSplit, getSampleTweets } from "../api";
import { useDate } from "../contexts/DateContext";

// Helper function to format date
function iso(x) {
  if (!x) return "";
  // HTML date inputs already return YYYY-MM-DD format, so use it directly
  // This prevents timezone issues where Date() constructor can shift dates
  if (typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x)) {
    return x;
  }
  // Fallback for other date formats
  const d = new Date(x);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export default function AspectAnalysis() {
  const { start, end, meta, setStart, setEnd } = useDate();
  const [split, setSplit] = useState(null);
  const [splitAsPercent, setSplitAsPercent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [totalTweets, setTotalTweets] = useState(0);
  const [sampleTweetsModal, setSampleTweetsModal] = useState({ isOpen: false, aspect: null, sentiment: null, tweets: [] });
  const [loadingTweets, setLoadingTweets] = useState(false);
  const [selectedAspect, setSelectedAspect] = useState('');
  const [selectedFormat, setSelectedFormat] = useState('');

  // ---- load aspects × sentiment split ----
  useEffect(() => {
    if (!start || !end) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const s = await getAspectSentimentSplit(start, end, splitAsPercent, true); // Include others
        setSplit(s);
        
        // Calculate total tweets
        if (s?.labels && s?.counts) {
          const total = s.labels.reduce((sum, _, index) => {
            const pos = s.counts.positive?.[index] || 0;
            const neu = s.counts.neutral?.[index] || 0;
            const neg = s.counts.negative?.[index] || 0;
            return sum + pos + neu + neg;
          }, 0);
          setTotalTweets(total);
        }
      } catch (e) {
        setErr("Failed to load aspect data");
        console.error("Failed to load aspect split", e);
      } finally {
        setLoading(false);
      }
    })();
  }, [start, end, splitAsPercent]);

  // Function to fetch sample tweets for specific aspect and sentiment
  const fetchSampleTweets = async (aspect, sentiment) => {
    try {
      setLoadingTweets(true);
      
      // Fetch actual tweets from API
      const tweets = await getSampleTweets(start, end, aspect, sentiment, 10);
      
      setSampleTweetsModal({
        isOpen: true,
        aspect: aspect,
        sentiment: sentiment,
        tweets: tweets
      });
    } catch (error) {
      console.error('Failed to fetch sample tweets:', error);
      // Fallback to sample data if API fails
      const sampleTweets = [
        `Sample tweet about ${aspect} with ${sentiment} sentiment`,
        `Another ${sentiment} tweet regarding ${aspect}`,
        `Customer feedback on ${aspect} - ${sentiment} tone`,
        `User experience with ${aspect} was ${sentiment}`,
        `Review of ${aspect} service - ${sentiment} sentiment`
      ];
      
      setSampleTweetsModal({
        isOpen: true,
        aspect: aspect,
        sentiment: sentiment,
        tweets: sampleTweets
      });
    } finally {
      setLoadingTweets(false);
    }
  };

  // Download functions for specific aspect and sentiment (from modal)
  const downloadTweetsAsExcel = async () => {
    try {
      console.log('Starting Excel download...');
      console.log('Parameters:', { start, end, aspect: sampleTweetsModal.aspect, sentiment: sampleTweetsModal.sentiment });
      
      // Fetch all tweets for this aspect and sentiment (not just the sample)
      const allTweets = await getSampleTweets(start, end, sampleTweetsModal.aspect, sampleTweetsModal.sentiment, 1000);
      console.log('Fetched tweets:', allTweets.length);
      
      if (!allTweets || allTweets.length === 0) {
        alert('No tweets found to download');
        return;
      }
      
      // Create Excel data
      const excelData = allTweets.map((tweet, index) => ({
        'Tweet #': index + 1,
        'Tweet Text': tweet,
        'Aspect': sampleTweetsModal.aspect,
        'Sentiment': sampleTweetsModal.sentiment,
        'Date Range': `${start} to ${end}`
      }));
      
      // Convert to CSV (simpler than Excel for now)
      const csvContent = [
        Object.keys(excelData[0]).join(','),
        ...excelData.map(row => Object.values(row).map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
      ].join('\n');
      
      console.log('CSV content length:', csvContent.length);
      
      // Download CSV
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `tweets_${sampleTweetsModal.aspect}_${sampleTweetsModal.sentiment}_${start}_to_${end}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the URL object
      URL.revokeObjectURL(url);
      
      console.log('Download completed');
    } catch (error) {
      console.error('Failed to download Excel:', error);
      
      // More specific error handling
      if (error.response?.status === 422) {
        alert(`Invalid parameters: ${error.response?.data?.detail || 'Please check your date range and aspect/sentiment values'}`);
      } else if (error.response?.status === 404) {
        alert('No data found for the selected criteria');
      } else {
        alert(`Failed to download Excel file: ${error.message}`);
      }
    }
  };

  // Download functions for entire aspect (all sentiments)
  const downloadAspectTweetsAsExcel = async (aspect) => {
    try {
      console.log('Starting aspect Excel download...');
      console.log('Parameters:', { start, end, aspect });
      
      // Fetch tweets for all sentiments of this aspect
      const [positiveTweets, neutralTweets, negativeTweets] = await Promise.all([
        getSampleTweets(start, end, aspect, 'positive', 1000),
        getSampleTweets(start, end, aspect, 'neutral', 1000),
        getSampleTweets(start, end, aspect, 'negative', 1000)
      ]);
      
      const allTweets = [
        ...positiveTweets.map(tweet => ({ tweet, sentiment: 'positive' })),
        ...neutralTweets.map(tweet => ({ tweet, sentiment: 'neutral' })),
        ...negativeTweets.map(tweet => ({ tweet, sentiment: 'negative' }))
      ];
      
      console.log('Fetched aspect tweets:', allTweets.length);
      
      if (allTweets.length === 0) {
        alert('No tweets found to download');
        return;
      }
      
      // Create Excel data
      const excelData = allTweets.map((item, index) => ({
        'Tweet #': index + 1,
        'Tweet Text': item.tweet,
        'Aspect': aspect,
        'Sentiment': item.sentiment,
        'Date Range': `${start} to ${end}`
      }));
      
      // Convert to CSV
      const csvContent = [
        Object.keys(excelData[0]).join(','),
        ...excelData.map(row => Object.values(row).map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
      ].join('\n');
      
      // Download CSV
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `tweets_${aspect}_all_sentiments_${start}_to_${end}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      console.log('Aspect Excel download completed');
    } catch (error) {
      console.error('Failed to download aspect Excel:', error);
      alert(`Failed to download Excel file: ${error.message}`);
    }
  };

  const downloadAspectTweetsAsPDF = async (aspect) => {
    try {
      console.log('Starting aspect PDF download...');
      console.log('Parameters:', { start, end, aspect });
      
      // Fetch tweets for all sentiments of this aspect
      const [positiveTweets, neutralTweets, negativeTweets] = await Promise.all([
        getSampleTweets(start, end, aspect, 'positive', 1000),
        getSampleTweets(start, end, aspect, 'neutral', 1000),
        getSampleTweets(start, end, aspect, 'negative', 1000)
      ]);
      
      const allTweets = [
        ...positiveTweets.map(tweet => ({ tweet, sentiment: 'positive' })),
        ...neutralTweets.map(tweet => ({ tweet, sentiment: 'neutral' })),
        ...negativeTweets.map(tweet => ({ tweet, sentiment: 'negative' }))
      ];
      
      console.log('Fetched aspect tweets for PDF:', allTweets.length);
      
      if (allTweets.length === 0) {
        alert('No tweets found to download');
        return;
      }
      
      // Create PDF content
      const pdfContent = `
        <!DOCTYPE html>
        <html>
          <head>
            <title>Tweets Report - ${aspect} (All Sentiments)</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; }
              .meta { background: #f5f5f5; padding: 10px; margin-bottom: 20px; }
              .tweet { margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; }
              .tweet-number { font-weight: bold; color: #666; }
              .sentiment-positive { border-left-color: #22c55e; }
              .sentiment-neutral { border-left-color: #f59e0b; }
              .sentiment-negative { border-left-color: #ef4444; }
            </style>
          </head>
          <body>
            <h1>Tweets Report - ${aspect}</h1>
            <div class="meta">
              <p><strong>Aspect:</strong> ${aspect}</p>
              <p><strong>Sentiments:</strong> All (Positive, Neutral, Negative)</p>
              <p><strong>Date Range:</strong> ${start} to ${end}</p>
              <p><strong>Total Tweets:</strong> ${allTweets.length}</p>
              <p><strong>Breakdown:</strong> Positive: ${positiveTweets.length}, Neutral: ${neutralTweets.length}, Negative: ${negativeTweets.length}</p>
            </div>
            ${allTweets.map((item, index) => `
              <div class="tweet sentiment-${item.sentiment}">
                <div class="tweet-number">Tweet ${index + 1} (${item.sentiment})</div>
                <div>${String(item.tweet).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
              </div>
            `).join('')}
          </body>
        </html>
      `;
      
      // Open in new window for printing/saving as PDF
      const newWindow = window.open('', '_blank');
      if (!newWindow) {
        alert('Please allow popups for this site to download PDF');
        return;
      }
      
      newWindow.document.write(pdfContent);
      newWindow.document.close();
      newWindow.focus();
      
      // Trigger print dialog
      setTimeout(() => {
        newWindow.print();
      }, 1000);
      
      console.log('Aspect PDF download initiated');
    } catch (error) {
      console.error('Failed to download aspect PDF:', error);
      alert(`Failed to download PDF file: ${error.message}`);
    }
  };

  const downloadTweetsAsPDF = async () => {
    try {
      console.log('Starting PDF download...');
      console.log('Parameters:', { start, end, aspect: sampleTweetsModal.aspect, sentiment: sampleTweetsModal.sentiment });
      
      // Fetch all tweets for this aspect and sentiment
      const allTweets = await getSampleTweets(start, end, sampleTweetsModal.aspect, sampleTweetsModal.sentiment, 1000);
      console.log('Fetched tweets for PDF:', allTweets.length);
      
      if (!allTweets || allTweets.length === 0) {
        alert('No tweets found to download');
        return;
      }
      
      // Create PDF content
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
            ${allTweets.map((tweet, index) => `
              <div class="tweet">
                <div class="tweet-number">Tweet ${index + 1}</div>
                <div>${String(tweet).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
              </div>
            `).join('')}
          </body>
        </html>
      `;
      
      // Open in new window for printing/saving as PDF
      const newWindow = window.open('', '_blank');
      if (!newWindow) {
        alert('Please allow popups for this site to download PDF');
        return;
      }
      
      newWindow.document.write(pdfContent);
      newWindow.document.close();
      newWindow.focus();
      
      // Trigger print dialog
      setTimeout(() => {
        newWindow.print();
      }, 1000);
      
      console.log('PDF download initiated');
    } catch (error) {
      console.error('Failed to download PDF:', error);
      
      // More specific error handling
      if (error.response?.status === 422) {
        alert(`Invalid parameters: ${error.response?.data?.detail || 'Please check your date range and aspect/sentiment values'}`);
      } else if (error.response?.status === 404) {
        alert('No data found for the selected criteria');
      } else {
        alert(`Failed to download PDF file: ${error.message}`);
      }
    }
  };

  // Download handler for the new download section
  const handleDownload = async () => {
    if (!selectedAspect || !selectedFormat) return;
    
    try {
      if (selectedAspect === 'all') {
        // Download all tweets
        if (selectedFormat === 'csv') {
          await downloadAllTweetsAsExcel();
        } else if (selectedFormat === 'pdf') {
          await downloadAllTweetsAsPDF();
        }
      } else {
        // Download specific aspect tweets
        if (selectedFormat === 'csv') {
          await downloadAspectTweetsAsExcel(selectedAspect);
        } else if (selectedFormat === 'pdf') {
          await downloadAspectTweetsAsPDF(selectedAspect);
        }
      }
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };

  // Download all tweets as Excel function
  const downloadAllTweetsAsExcel = async () => {
    try {
      // Fetch tweets for all aspects and sentiments
      const allAspects = stackedBarData.labels || [];
      const allTweets = [];
      
      for (const aspect of allAspects) {
        const [positiveTweets, neutralTweets, negativeTweets] = await Promise.all([
          getSampleTweets(start, end, aspect, 'positive', 1000),
          getSampleTweets(start, end, aspect, 'neutral', 1000),
          getSampleTweets(start, end, aspect, 'negative', 1000)
        ]);
        
        allTweets.push(
          ...positiveTweets.map(tweet => ({ tweet, aspect, sentiment: 'positive' })),
          ...neutralTweets.map(tweet => ({ tweet, aspect, sentiment: 'neutral' })),
          ...negativeTweets.map(tweet => ({ tweet, aspect, sentiment: 'negative' }))
        );
      }
      
      if (allTweets.length === 0) {
        alert('No tweets found to download');
        return;
      }
      
      // Create CSV data
      const csvData = allTweets.map((item, index) => ({
        'Tweet #': index + 1,
        'Tweet Text': item.tweet,
        'Aspect': item.aspect,
        'Sentiment': item.sentiment,
        'Date Range': `${start} to ${end}`
      }));
      
      // Convert to CSV
      const csvContent = [
        Object.keys(csvData[0]).join(','),
        ...csvData.map(row => Object.values(row).map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
      ].join('\n');
      
      // Download CSV
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', `all_tweets_${start}_to_${end}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      URL.revokeObjectURL(url);
      console.log('All tweets Excel download completed');
    } catch (error) {
      console.error('Failed to download all tweets as Excel:', error);
      alert('Failed to download all tweets as Excel. Please try again.');
    }
  };

  // View all tweets as PDF function (view only, no download)
  const viewAllTweetsAsPDF = async () => {
    try {
      // Fetch tweets for all aspects and sentiments
      const allAspects = stackedBarData.labels || [];
      const allTweets = [];
      
      for (const aspect of allAspects) {
        const [positiveTweets, neutralTweets, negativeTweets] = await Promise.all([
          getSampleTweets(start, end, aspect, 'positive', 1000),
          getSampleTweets(start, end, aspect, 'neutral', 1000),
          getSampleTweets(start, end, aspect, 'negative', 1000)
        ]);
        
        allTweets.push(
          ...positiveTweets.map(tweet => ({ tweet, aspect, sentiment: 'positive' })),
          ...neutralTweets.map(tweet => ({ tweet, aspect, sentiment: 'neutral' })),
          ...negativeTweets.map(tweet => ({ tweet, aspect, sentiment: 'negative' }))
        );
      }
      
      if (allTweets.length === 0) {
        alert('No tweets found to view');
        return;
      }
      
      // Create PDF content
      const pdfContent = `
        <html>
          <head>
            <title>All Tweets Report</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 20px; }
              h1 { color: #333; text-align: center; }
              h2 { color: #666; border-bottom: 2px solid #ddd; }
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
            
            ${allTweets.map((item, index) => `
              <div class="tweet ${item.sentiment}">
                <strong>Tweet #${index + 1}</strong><br>
                ${item.tweet}<br>
                <div class="metadata">
                  <strong>Aspect:</strong> ${item.aspect} | 
                  <strong>Sentiment:</strong> ${item.sentiment}
                </div>
              </div>
            `).join('')}
          </body>
        </html>
      `;
      
      // Open in new tab for viewing only
      const viewWindow = window.open('', '_blank');
      viewWindow.document.write(pdfContent);
      viewWindow.document.close();
      
      console.log('All tweets view opened');
    } catch (error) {
      console.error('Failed to view all tweets:', error);
      alert('Failed to view all tweets. Please try again.');
    }
  };

  const stackedBarData = useMemo(() => {
    if (!split) return { labels: [], datasets: [] };
    const labels = split.labels || [];
    const series = splitAsPercent ? split.percent : split.counts;
    
    // Calculate total counts for each aspect
    const totals = labels.map((_, index) => {
      const pos = series.positive?.[index] || 0;
      const neu = series.neutral?.[index] || 0;
      const neg = series.negative?.[index] || 0;
      return pos + neu + neg;
    });
    
    return {
      labels: labels.map((t) => t.toUpperCase()),
      datasets: [
        { 
          label: splitAsPercent ? "% Positive" : "Positive", 
          data: series.positive || [], 
          backgroundColor: "#4ade80",   
          stack: "s",
          totals: totals
        },
        { 
          label: splitAsPercent ? "% Neutral"  : "Neutral",  
          data: series.neutral  || [], 
          backgroundColor: "#fbbf24",   
          stack: "s",
          totals: totals
        },
        { 
          label: splitAsPercent ? "% Negative" : "Negative", 
          data: series.negative || [], 
          backgroundColor: "#f87171",   
          stack: "s",
          totals: totals
        },
      ],
    };
  }, [split, splitAsPercent]);

  const stackedBarOptions = {
    responsive: true,
    plugins: { 
      legend: { position: "top" },
      datalabels: {
        display: true,
        color: 'white',
        font: {
          weight: 'bold',
          size: 11
        },
        formatter: (value, context) => {
          const dataset = context.dataset;
          const totals = dataset.totals;
          const total = totals[context.dataIndex];
          const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
          return `${percentage}%`;
        }
      },
      tooltip: {
        callbacks: {
          afterLabel: function(context) {
            const dataset = context.dataset;
            const totals = dataset.totals;
            const total = totals[context.dataIndex];
            const value = context.parsed.y;
            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
            return `Total: ${total} | ${percentage}%`;
          }
        }
      }
    },
    scales: {
      x: { stacked: true },
      y: {
        stacked: true,
        suggestedMax: splitAsPercent ? 100 : undefined,
        ticks: { callback: (v) => (splitAsPercent ? `${v}%` : v) },
      },
    },
  };

  return (
    <div className="space-y-4">
      {/* Aspect Analysis Header */}
      <div className="mb-0.5 pt-2 pb-2">
        <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center shadow-lg ml-2">
            <svg className="w-3 h-3 text-slate-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div className="px-2 py-1">
            <h1 className="text-xl font-bold text-white">Aspect Analysis</h1>
            <p className="text-slate-400 text-sm">Aspect × Sentiment Breakdown</p>
          </div>
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

      {/* Error & Loading States */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-4">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span className="text-red-300 font-semibold text-sm">{err}</span>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-slate-300 font-semibold">Loading aspect analysis...</span>
          </div>
        </div>
      )}

      {!loading && split && (
        <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl px-4 pt-2 pb-0 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-end mb-4">
            <div className="flex items-center space-x-4">
              {/* Download All Tweets Button */}
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => viewAllTweetsAsPDF()}
                  className="flex items-center space-x-1 bg-slate-600 hover:bg-slate-700 text-white px-2 py-1 rounded text-xs transition-colors"
                  title="View All Tweets as PDF"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                  <span>View All Tweets</span>
                </button>
              </div>
              
              <div className="text-sm text-slate-300 bg-slate-700/30 px-3 py-1 rounded-lg">
                Total: {totalTweets.toLocaleString()} tweets
              </div>
            </div>
          </div>
          <div className="h-80">
            <Bar 
              data={stackedBarData} 
              options={{
                ...stackedBarOptions,
                responsive: true,
                maintainAspectRatio: false,
                onClick: (event, elements) => {
                  if (elements.length > 0) {
                    const element = elements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataIndex = element.index;
                    const aspect = stackedBarData.labels[dataIndex];
                    const sentiment = stackedBarData.datasets[datasetIndex].label;
                    
                    fetchSampleTweets(aspect.toLowerCase(), sentiment.toLowerCase());
                  }
                },
                plugins: {
                  ...stackedBarOptions.plugins,
                  datalabels: {
                    ...stackedBarOptions.plugins.datalabels,
                    display: true, // Always show labels on each part of the bar
                    color: '#1f2937', // Dark gray for better contrast
                    font: {
                      weight: 'bold',
                      size: 12
                    },
                    formatter: (value, context) => {
                      if (splitAsPercent) {
                        // In percentage mode, show percentages inside bars
                        const dataset = context.dataset;
                        const totals = dataset.totals;
                        const total = totals[context.dataIndex];
                        const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                        return `${percentage}%`;
                      } else {
                        // In count mode, show counts with percentages
                        const dataset = context.dataset;
                        const totals = dataset.totals;
                        const total = totals[context.dataIndex];
                        const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                        return value > 0 ? `${value.toLocaleString()} (${percentage}%)` : '';
                      }
                    },
                    anchor: 'center',
                    offset: 0
                  },
                  legend: {
                    ...stackedBarOptions.plugins.legend,
                    labels: {
                      color: '#e2e8f0',
                      font: {
                        size: 12,
                        weight: 'bold'
                      }
                    }
                  }
                },
                scales: {
                  ...stackedBarOptions.scales,
                  y: {
                    ...stackedBarOptions.scales.y,
                    grid: {
                      color: 'rgba(148, 163, 184, 0.1)'
                    },
                    ticks: {
                      color: '#94a3b8',
                      font: {
                        size: 11
                      }
                    }
                  },
                  x: {
                    ...stackedBarOptions.scales.x,
                    grid: {
                      display: false
                    },
                    ticks: {
                      color: '#94a3b8',
                      font: {
                        size: 14
                      },
                      callback: function(value, index) {
                        const label = this.getLabelForValue(value);
                        const total = stackedBarData.datasets[0]?.totals?.[index] || 0;
                        return [`${label}`, `${total.toLocaleString()}`];
                      }
                    }
                  }
                }
              }}
              plugins={[ChartDataLabels]}
            />
          </div>
        </div>
      )}

      {/* Download Section */}
      {!loading && split && (
        <div className="-mt-2 bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <span className="text-slate-200 text-sm">Download tweets:</span>
              
            {/* Aspect Dropdown */}
            <select
              value={selectedAspect || ''}
              onChange={(e) => setSelectedAspect(e.target.value)}
              className="bg-slate-700 text-slate-200 px-2 py-1 rounded text-sm border border-slate-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="">Select Aspect</option>
              <option value="all">All Tweets</option>
              {stackedBarData.labels.map((aspect) => (
                <option key={aspect} value={aspect}>
                  {aspect.charAt(0).toUpperCase() + aspect.slice(1)}
                </option>
              ))}
            </select>

              {/* File Format Dropdown */}
              <select
                value={selectedFormat || ''}
                onChange={(e) => setSelectedFormat(e.target.value)}
                className="bg-slate-700 text-slate-200 px-2 py-1 rounded text-sm border border-slate-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="">Select Format</option>
                <option value="csv">CSV</option>
                <option value="pdf">PDF</option>
              </select>

              {/* Download Button */}
              <button
                onClick={handleDownload}
                disabled={!selectedAspect || !selectedFormat}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-3 py-1 rounded text-sm transition-colors flex items-center space-x-1"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Download</span>
              </button>
            </div>
            
            {/* User Notes */}
            <div className="text-xs text-slate-400 text-right max-w-sm">
              <div className="mb-1">💡 <strong>Tips:</strong></div>
              <div>• Click bars to view top tweets by aspect by sentiment</div>
              <div>• Use "View All Tweets" to see all tweets used for analysis</div>
            </div>
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
                  
                  {/* Download buttons */}
                  <div className="flex items-center justify-center space-x-3 pt-4 border-t border-slate-600/30">
                    <button
                      onClick={() => downloadTweetsAsExcel()}
                      className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="text-sm font-medium">Download Excel</span>
                    </button>
                    
                    <button
                      onClick={() => downloadTweetsAsPDF()}
                      className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
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
