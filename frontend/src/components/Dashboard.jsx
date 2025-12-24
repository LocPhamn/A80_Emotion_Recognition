import { useState } from 'react'
import VideoUpload from './VideoUpload.jsx'
import WebcamDetection from './WebcamDetection.jsx'
import Statistics from './Statistics.jsx'
import './Dashboard.css'

function Dashboard() {
  const [activeTab, setActiveTab] = useState('home')
  const [detectionStats, setDetectionStats] = useState({
    totalFrames: 0,
    detectedObjects: 0,
    averageConfidence: 0
  })

  const menuItems = [
    { id: 'home', icon: '🏠', label: 'Dashboard', active: true },
    { id: 'webcam', icon: '📹', label: 'Realtime Detection' },
    { id: 'upload', icon: '📁', label: 'Upload Video' },
    { id: 'statistics', icon: '📊', label: 'Statistics' },
    { id: 'history', icon: '📋', label: 'History' },
    { id: 'settings', icon: '⚙️', label: 'Settings' }
  ]

  const renderContent = () => {
    switch(activeTab) {
      case 'home':
        return (
          <div className="home-content">
            <div className="welcome-section">
              <h1>🎯 AI Detection Dashboard</h1>
              <p>Computer Vision Object Detection System</p>
            </div>
            <Statistics stats={detectionStats} />
          </div>
        )
      case 'webcam':
        return (
          <div className="content-section">
            <h1>📹 Realtime Webcam Detection</h1>
            <p className="subtitle">Detect objects in real-time using your webcam</p>
            <WebcamDetection onStats={setDetectionStats} />
          </div>
        )
      case 'upload':
        return (
          <div className="content-section">
            <h1>📁 Upload Video</h1>
            <p className="subtitle">Upload and analyze video files</p>
            <VideoUpload onStats={setDetectionStats} />
          </div>
        )
      case 'statistics':
        return (
          <div className="content-section">
            <h1>📊 Statistics Overview</h1>
            <p className="subtitle">View detailed analytics and metrics</p>
            <Statistics stats={detectionStats} />
            <div className="stats-details">
              <div className="detail-card">
                <h3>Detection Performance</h3>
                <div className="detail-item">
                  <span>Average Processing Time:</span>
                  <strong>45ms/frame</strong>
                </div>
                <div className="detail-item">
                  <span>Model Accuracy:</span>
                  <strong>94.5%</strong>
                </div>
                <div className="detail-item">
                  <span>Total Sessions:</span>
                  <strong>127</strong>
                </div>
              </div>
            </div>
          </div>
        )
      case 'history':
        return (
          <div className="content-section">
            <h1>📋 Detection History</h1>
            <p className="subtitle">View past detection results</p>
            <div className="history-list">
              <div className="history-item">
                <span className="history-icon">🎥</span>
                <div className="history-info">
                  <strong>Video Analysis - traffic.mp4</strong>
                  <p>Detected 245 objects | Dec 20, 2025 14:30</p>
                </div>
              </div>
              <div className="history-item">
                <span className="history-icon">📹</span>
                <div className="history-info">
                  <strong>Webcam Session</strong>
                  <p>Detected 89 objects | Dec 20, 2025 12:15</p>
                </div>
              </div>
            </div>
          </div>
        )
      case 'settings':
        return (
          <div className="content-section">
            <h1>⚙️ Settings</h1>
            <p className="subtitle">Configure detection parameters</p>
            <div className="settings-group">
              <h3>Detection Settings</h3>
              <div className="setting-item">
                <label>Confidence Threshold:</label>
                <input type="range" min="0" max="100" defaultValue="70" />
                <span>70%</span>
              </div>
              <div className="setting-item">
                <label>Frame Rate:</label>
                <select>
                  <option>5 FPS</option>
                  <option defaultValue>10 FPS</option>
                  <option>15 FPS</option>
                  <option>30 FPS</option>
                </select>
              </div>
            </div>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>🤖 AI Vision</h2>
          <p>Admin Panel</p>
        </div>
        
        <nav className="sidebar-nav">
          {menuItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="user-info">
            <span className="user-avatar">👤</span>
            <div>
              <strong>John Doe</strong>
              <p>Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="top-bar">
          <div className="search-bar">
            <input type="text" placeholder="🔍 Search..." />
          </div>
          <div className="top-actions">
            <button className="icon-btn">🔔</button>
            <button className="icon-btn">⚙️</button>
          </div>
        </header>

        <div className="content-wrapper">
          {renderContent()}
        </div>
      </main>
    </div>
  )
}

export default Dashboard