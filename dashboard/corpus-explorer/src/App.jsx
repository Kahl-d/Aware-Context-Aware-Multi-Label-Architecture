import { useState, useEffect } from 'react'
import AnalyticsView from './components/AnalyticsView'
import DataExplorer from './components/DataExplorer'
import './App.css'

function App() {
  const [stats, setStats] = useState(null)
  const [essays, setEssays] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loadingEssays, setLoadingEssays] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('analytics')

  // Load stats first (112KB), then essays (30MB) in background
  useEffect(() => {
    fetch('/stats.json')
      .then(r => r.json())
      .then(s => { setStats(s); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })

    fetch('/essays_summary.json')
      .then(r => r.json())
      .then(e => { setEssays(Array.isArray(e) ? e : Object.values(e)); setLoadingEssays(false) })
      .catch(err => { setError(prev => prev || err.message); setLoadingEssays(false) })
  }, [])

  if (loading) return (
    <div className="loading-screen">
      <div className="loading-spinner" />
      <span>Loading research data...</span>
    </div>
  )
  if (error) return (
    <div className="error-screen">
      <h2>Error Loading Data</h2>
      <p>{error}</p>
    </div>
  )
  if (!stats) return (
    <div className="error-screen">
      <h2>No Data</h2>
      <p>stats.json not found in public/</p>
    </div>
  )

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="header-title">ALMA Research Dashboard</h1>
          <p className="header-subtitle">Community Cultural Wealth in STEM Education</p>
        </div>
        <nav className="header-nav">
          <button
            className={activeTab === 'analytics' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setActiveTab('analytics')}>
            Analytics
          </button>
          <button
            className={activeTab === 'data' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setActiveTab('data')}>
            Data Explorer
          </button>
        </nav>
      </header>
      <main className="main">
        {activeTab === 'analytics' && <AnalyticsView stats={stats} />}
        {activeTab === 'data' && (
          loadingEssays
            ? <div className="loading-inline"><div className="loading-spinner" /><span>Loading essay data (30MB)...</span></div>
            : <DataExplorer essays={essays} stats={stats} />
        )}
      </main>
    </div>
  )
}

export default App
