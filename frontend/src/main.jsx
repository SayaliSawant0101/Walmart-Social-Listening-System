import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Note: GA4 is no longer initialized here directly. Google Tag Manager
// (see index.html) now owns loading the Google tag and firing GA4 events.
// See src/analytics.js for how the app pushes events into GTM's dataLayer.

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
