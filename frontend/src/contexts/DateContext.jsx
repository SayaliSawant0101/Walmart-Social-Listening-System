// frontend/src/contexts/DateContext.jsx
import React, { createContext, useContext, useState, useEffect } from "react";
import { getMeta } from "../api";

const DateContext = createContext();

export function DateProvider({ children }) {
  const [meta, setMeta] = useState(null);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [loading, setLoading] = useState(true);

  // Load meta once on app start
  useEffect(() => {
    const loadMeta = async () => {
      try {
        console.log('Loading metadata from API...');
        const mr = await getMeta();
        console.log('Metadata loaded:', mr);
        setMeta(mr);
        // Default to all days of August 2025 (Aug 1-31)
        const defaultStart = "2025-08-01";
        const defaultEnd = "2025-08-31";
        // Set to Aug 1-31 if data supports it, otherwise use available range
        const finalStart = mr?.min && mr.min <= "2025-08-01" ? defaultStart : (mr?.min || defaultStart);
        const finalEnd = mr?.max && mr.max >= "2025-08-31" ? defaultEnd : (mr?.max || defaultEnd);
        setStart(finalStart);
        setEnd(finalEnd);
        console.log('Date range set:', finalStart, 'to', finalEnd);
      } catch (error) {
        console.error("Failed to load metadata:", error);
        console.error("Error details:", error.response?.data || error.message);
        // Set fallback data if API fails - all days of August 2025
        setMeta({ min: "2025-08-01", max: "2025-08-31" });
        setStart("2025-08-01");
        setEnd("2025-08-31");
      } finally {
        setLoading(false);
      }
    };

    // Add timeout to prevent hanging
    const timeoutId = setTimeout(() => {
      console.log('Metadata loading timeout - using fallback');
      setMeta({ min: "2025-08-01", max: "2025-08-31" });
      setStart("2025-08-01");
      setEnd("2025-08-31");
      setLoading(false);
    }, 5000);

    loadMeta();

    return () => clearTimeout(timeoutId);
  }, []);

  const value = {
    meta,
    start,
    end,
    setStart,
    setEnd,
    loading
  };

  return (
    <DateContext.Provider value={value}>
      {children}
    </DateContext.Provider>
  );
}

export function useDate() {
  const context = useContext(DateContext);
  if (context === undefined) {
    throw new Error('useDate must be used within a DateProvider');
  }
  return context;
}

