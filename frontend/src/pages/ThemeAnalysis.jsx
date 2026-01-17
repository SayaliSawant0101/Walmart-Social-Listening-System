// frontend/src/pages/ThemeAnalysis.jsx
import React from "react";
import ThemePanel from "../ThemePanel";
import { useDate } from "../contexts/DateContext";

export default function ThemeAnalysis() {
  const { start, end, meta, setStart, setEnd, loading: metaLoading } = useDate();

  return (
    <div className="space-y-4">
      {/* Theme Analysis Header */}
      <div className="mb-0.5 pt-2 pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center shadow-lg ml-2">
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
            </div>
            <div className="px-2 py-1">
              <h1 className="text-xl font-bold text-white">Theme Analysis</h1>
              <p className="text-slate-400 text-sm">AI-powered theme discovery and analysis</p>
            </div>
          </div>
          
          {/* Date Card */}
          <div className="flex items-center bg-slate-700/30 backdrop-blur-sm rounded-md border border-slate-600/50 px-3 py-2">
            <div className="w-1 h-1 bg-gradient-to-r from-emerald-400 to-cyan-400 rounded-full"></div>
            <span className="text-xs text-slate-200 font-medium ml-1">Date:</span>
            <input
              type="date"
              value={start || ''}
              onChange={(e) => setStart(e.target.value)}
              className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all ml-0.5"
              disabled={metaLoading}
            />
            <span className="text-slate-400 text-xs ml-0.5">to</span>
            <input
              type="date"
              value={end || ''}
              onChange={(e) => setEnd(e.target.value)}
              className="px-1 py-0.5 bg-slate-800/50 border border-slate-600 rounded text-xs text-white focus:ring-1 focus:ring-emerald-400 focus:border-emerald-400 transition-all ml-0.5"
              disabled={metaLoading}
            />
            {metaLoading && (
              <div className="w-3 h-3 border border-slate-400 border-t-transparent rounded-full animate-spin ml-0.5"></div>
            )}
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl p-4 border border-slate-600/30 shadow-xl">
        <div className="h-full overflow-y-auto">
          <ThemePanel />
        </div>
      </div>
    </div>
  );
}

