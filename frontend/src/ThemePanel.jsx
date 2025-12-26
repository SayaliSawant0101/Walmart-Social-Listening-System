// frontend/src/ThemePanel.jsx
import React, { useState, useEffect } from "react";
import { fetchThemes, fetchThemesCompetitor } from "./api";
import { useDate } from "./contexts/DateContext";

export default function ThemePanel() {
  const { start, end } = useDate();
  const [themeCount, setThemeCount] = useState(null); // null = auto-detect
  const [loading, setLoading] = useState(false);
  const [loadingCostco, setLoadingCostco] = useState(false);
  const [data, setData] = useState(null); // {updated_at, themes: [...]} for Walmart
  const [dataCostco, setDataCostco] = useState(null); // {updated_at, themes: [...]} for Costco
  const [err, setErr] = useState("");
  const [errCostco, setErrCostco] = useState("");

  async function run() {
    setLoading(true);
    setLoadingCostco(true);
    setErr("");
    setErrCostco("");
    
    try {
      console.log(`Generating themes for date range: ${start} to ${end} with auto-detection`);
      
      // Fetch both Walmart and Costco themes in parallel
      const [walmartPayload, costcoPayload] = await Promise.allSettled([
        fetchThemes({
          start: start || null,
          end: end || null,
          n_clusters: themeCount, // Use selected theme count or auto-detect
        }),
        fetchThemesCompetitor({
          start: start || null,
          end: end || null,
          n_clusters: themeCount, // Use selected theme count or auto-detect
        })
      ]);
      
      // Handle Walmart results
      if (walmartPayload.status === 'fulfilled') {
        console.log('Walmart themes generated:', walmartPayload.value);
        setData(walmartPayload.value);
      } else {
        console.error('Failed to generate Walmart themes:', walmartPayload.reason);
        const e = walmartPayload.reason;
        let errorMessage = 'Failed to generate Walmart themes';
        
        // Check if we have a response from the server (even if it's an error)
        if (e.response) {
          const errorData = e.response.data || {};
          errorMessage = errorData.error || errorData.message || `Server returned ${e.response.status}`;
          if (errorData.hint) {
            errorMessage += `\n\n${errorData.hint}`;
          }
          if (errorData.file_path) {
            errorMessage += `\n\nFile path: ${errorData.file_path}`;
          }
        } else if (e.request) {
          // Request was made but no response received
          errorMessage = 'Network Error';
          if (e.code === 'ECONNABORTED' || e.message?.includes('timeout')) {
            errorMessage += '\n\nThe request timed out. This may take several minutes. Please try again.';
          } else {
            errorMessage += '\n\nCannot connect to the server. Please check if the backend is running.';
          }
        } else if (e.message) {
          errorMessage = e.message;
          if (e.message.includes('timeout') || e.code === 'ECONNABORTED') {
            errorMessage += '\n\nThe request took too long. Try reducing the date range or number of themes.';
          } else if (e.message.includes('Network Error') || e.message.includes('ECONNREFUSED')) {
            errorMessage += '\n\nCannot connect to the server. Please check if the backend is running.';
          }
        }
        setErr(errorMessage);
      }
      
      // Handle Costco results
      if (costcoPayload.status === 'fulfilled') {
        console.log('Costco themes generated:', costcoPayload.value);
        setDataCostco(costcoPayload.value);
      } else {
        console.error('Failed to generate Costco themes:', costcoPayload.reason);
        const e = costcoPayload.reason;
        let errorMessage = 'Failed to generate Costco themes';
        
        // Check if we have a response from the server (even if it's an error)
        if (e.response) {
          const errorData = e.response.data || {};
          errorMessage = errorData.error || errorData.message || `Server returned ${e.response.status}`;
          if (errorData.hint) {
            errorMessage += `\n\n${errorData.hint}`;
          }
          if (errorData.file_path) {
            errorMessage += `\n\nFile path: ${errorData.file_path}`;
          }
          if (errorData.file_paths) {
            errorMessage += `\n\nChecked paths:\n- Raw: ${errorData.file_paths.raw}\n- Sentiment: ${errorData.file_paths.sentiment}`;
          }
        } else if (e.request) {
          // Request was made but no response received
          errorMessage = 'Network Error';
          if (e.code === 'ECONNABORTED' || e.message?.includes('timeout')) {
            errorMessage += '\n\nThe request timed out. This may take several minutes. Please try again.';
          } else {
            errorMessage += '\n\nCannot connect to the server. Please check if the backend is running.';
          }
        } else if (e.message) {
          errorMessage = e.message;
          if (e.message.includes('timeout') || e.code === 'ECONNABORTED') {
            errorMessage += '\n\nThe request took too long. Try reducing the date range or number of themes.';
          } else if (e.message.includes('Network Error') || e.message.includes('ECONNREFUSED')) {
            errorMessage += '\n\nCannot connect to the server. Please check if the backend is running.';
          }
        }
        setErrCostco(errorMessage);
      }
    } catch (e) {
      console.error('Unexpected error:', e);
      setErr('An unexpected error occurred while generating themes.');
    } finally {
      setLoading(false);
      setLoadingCostco(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Theme Analysis</h2>
          <p className="text-slate-400 mt-1">
            {start && end ? `${start} to ${end}` : 'All time'} • Auto-detected themes
          </p>
          </div>
          
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-slate-300">Themes:</label>
            <select
              value={themeCount || 'auto'}
              onChange={(e) => setThemeCount(e.target.value === 'auto' ? null : parseInt(e.target.value))}
              className="px-3 py-2 bg-slate-700/50 border border-slate-600/30 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            >
              <option value="auto">Auto-detect (Recommended)</option>
              <option value="3">3 Themes</option>
              <option value="4">4 Themes</option>
              <option value="5">5 Themes</option>
              <option value="6">6 Themes</option>
              <option value="7">7 Themes</option>
              <option value="8">8 Themes</option>
            </select>
          </div>
          
          <button
            onClick={run}
            disabled={loading || loadingCostco}
            className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-slate-600 disabled:to-slate-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl disabled:shadow-none disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {(loading || loadingCostco) ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating themes... (this may take 2-3 minutes)</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Generate Themes</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Two Column Layout: Walmart (Left) and Costco (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Walmart Themes Section (Left) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold text-white flex items-center space-x-2">
              <span className="text-blue-400">Walmart</span>
              <span className="text-slate-400">Themes</span>
            </h3>
            {loading && (
              <div className="flex items-center space-x-2 text-slate-400 text-sm">
                <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                <span>Generating...</span>
              </div>
            )}
          </div>

          {/* Walmart Error Message */}
          {err && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="flex-1">
                  <span className="text-red-300 font-medium block mb-1">Error Generating Walmart Themes</span>
                  <p className="text-red-200 text-sm whitespace-pre-line">{err}</p>
                </div>
              </div>
            </div>
          )}

          {/* Walmart Results */}
          {data && data.themes && data.themes.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <p className="text-sm text-slate-400">
                    {data.themes.length} theme{data.themes.length !== 1 ? 's' : ''} generated
                  </p>
                  {data.used_llm !== undefined && (
                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-lg text-xs ${
                      data.used_llm 
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' 
                        : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                    }`}>
                      {data.used_llm ? (
                        <>
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <span>AI Generated</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                          </svg>
                          <span>Fallback Mode</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
                <p className="text-xs text-slate-500">
                  {new Date(data.updated_at).toLocaleString()}
                </p>
              </div>
              
              <div className="space-y-3">
                {data.themes.map((t, index) => {
                  const totalTweets = data.themes.reduce((sum, theme) => sum + theme.tweet_count, 0);
                  const percentage = totalTweets > 0 ? Math.round((t.tweet_count / totalTweets) * 100) : 0;
                  
                  return (
                    <div key={t.id} className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-5 shadow-lg">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-3 flex-1">
                          <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg ${
                            index === 0 ? 'bg-gradient-to-br from-yellow-500 to-orange-500' :
                            index === 1 ? 'bg-gradient-to-br from-gray-500 to-slate-500' :
                            'bg-gradient-to-br from-emerald-500 to-cyan-500'
                          }`}>
                            {index + 1}
                          </div>
                          <div className="flex-1">
                            <h4 className="text-base font-bold text-white">
                              {t.name ? `"${String(t.name).replace(/^"+|"+$/g, "").trim()}"` : `Theme ${t.id}`}
                            </h4>
                            <p className="text-xs text-slate-400 mt-1">
                              {t.tweet_count} tweets
                            </p>
                          </div>
                        </div>
                      </div>

                      <p className="text-slate-300 text-sm leading-relaxed mb-3">
                        {t.summary || 'No summary available for this theme.'}
                      </p>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-slate-400">Volume</span>
                          <span className="text-xs text-slate-300 font-medium">
                            {percentage}% of {totalTweets} tweets
                          </span>
                        </div>
                        <div className="w-full bg-slate-700/50 rounded-full h-1.5">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-cyan-500 h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : data && data.themes && data.themes.length === 0 ? (
            <div className="text-center py-8 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <p className="text-slate-400 text-sm">No Walmart themes found</p>
            </div>
          ) : !loading && !err ? (
            <div className="text-center py-8 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <p className="text-slate-400 text-sm">Click "Generate Themes" to start</p>
            </div>
          ) : null}
        </div>

        {/* Costco Themes Section (Right) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold text-white flex items-center space-x-2">
              <span className="text-orange-400">Costco</span>
              <span className="text-slate-400">Themes</span>
            </h3>
            {loadingCostco && (
              <div className="flex items-center space-x-2 text-slate-400 text-sm">
                <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                <span>Generating...</span>
              </div>
            )}
          </div>

          {/* Costco Error Message */}
          {errCostco && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="flex-1">
                  <span className="text-red-300 font-medium block mb-1">Error Generating Costco Themes</span>
                  <p className="text-red-200 text-sm whitespace-pre-line">{errCostco}</p>
                </div>
              </div>
            </div>
          )}

          {/* Costco Results */}
          {dataCostco && dataCostco.themes && dataCostco.themes.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <p className="text-sm text-slate-400">
                    {dataCostco.themes.length} theme{dataCostco.themes.length !== 1 ? 's' : ''} generated
                  </p>
                  {dataCostco.used_llm !== undefined && (
                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-lg text-xs ${
                      dataCostco.used_llm 
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' 
                        : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                    }`}>
                      {dataCostco.used_llm ? (
                        <>
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          <span>AI Generated</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                          </svg>
                          <span>Fallback Mode</span>
                        </>
                      )}
                    </div>
                  )}
                </div>
                <p className="text-xs text-slate-500">
                  {new Date(dataCostco.updated_at).toLocaleString()}
                </p>
              </div>
              
              <div className="space-y-3">
                {dataCostco.themes.map((t, index) => {
                  const totalTweets = dataCostco.themes.reduce((sum, theme) => sum + theme.tweet_count, 0);
                  const percentage = totalTweets > 0 ? Math.round((t.tweet_count / totalTweets) * 100) : 0;
                  
                  return (
                    <div key={t.id} className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-5 shadow-lg">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center space-x-3 flex-1">
                          <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg ${
                            index === 0 ? 'bg-gradient-to-br from-yellow-500 to-orange-500' :
                            index === 1 ? 'bg-gradient-to-br from-gray-500 to-slate-500' :
                            'bg-gradient-to-br from-orange-500 to-red-500'
                          }`}>
                            {index + 1}
                          </div>
                          <div className="flex-1">
                            <h4 className="text-base font-bold text-white">
                              {t.name ? `"${String(t.name).replace(/^"+|"+$/g, "").trim()}"` : `Theme ${t.id}`}
                            </h4>
                            <p className="text-xs text-slate-400 mt-1">
                              {t.tweet_count} tweets
                            </p>
                          </div>
                        </div>
                      </div>

                      <p className="text-slate-300 text-sm leading-relaxed mb-3">
                        {t.summary || 'No summary available for this theme.'}
                      </p>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-slate-400">Volume</span>
                          <span className="text-xs text-slate-300 font-medium">
                            {percentage}% of {totalTweets} tweets
                          </span>
                        </div>
                        <div className="w-full bg-slate-700/50 rounded-full h-1.5">
                          <div 
                            className="bg-gradient-to-r from-orange-500 to-red-500 h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : dataCostco && dataCostco.themes && dataCostco.themes.length === 0 ? (
            <div className="text-center py-8 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <p className="text-slate-400 text-sm">No Costco themes found</p>
            </div>
          ) : !loadingCostco && !errCostco ? (
            <div className="text-center py-8 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <p className="text-slate-400 text-sm">Click "Generate Themes" to start</p>
            </div>
          ) : null}
        </div>
      </div>

    </div>
  );
}