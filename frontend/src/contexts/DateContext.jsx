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
        
        // Use the full available date range from the API
        const finalStart = mr?.date_range?.min || mr?.min || null;
        const finalEnd = mr?.date_range?.max || mr?.max || null;
        
        // Set meta with min/max for date input constraints
        setMeta({
          ...mr,
          min: finalStart,
          max: finalEnd
        });
        
        if (finalStart && finalEnd) {
          setStart(finalStart);
          setEnd(finalEnd);
          console.log('Date range set to full range:', finalStart, 'to', finalEnd);
        } else {
          // Fallback if date range not available
          console.warn('Date range not available from API, using fallback');
          setStart("2025-08-01");
          setEnd("2025-08-31");
        }
      } catch (error) {
        console.error("Failed to load metadata:", error);
        console.error("Error details:", error.response?.data || error.message);
        // Set fallback - will be updated when API responds
        setMeta({ min: null, max: null });
        setStart("");
        setEnd("");
      } finally {
        setLoading(false);
      }
    };

    // Add timeout to prevent hanging
    const timeoutId = setTimeout(() => {
      console.log('Metadata loading timeout - will retry when API is ready');
      setLoading(false);
    }, 10000); // Increased timeout to 10s to allow for S3 loading

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

