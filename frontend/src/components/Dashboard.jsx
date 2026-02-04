import { useState } from 'react'
import VideoUpload from './VideoUpload.jsx'
import WebcamDetection from './WebcamDetection.jsx'
import Statistics from './Statistics.jsx'
import VideoManagement from './VideoManagement.jsx'
import './Dashboard.css'
import logo from '../assets/logo2.png'
import backgroundImage from '../assets/background1.jpg'

function Dashboard() {
  const [activeTab, setActiveTab] = useState('home')
  const [detectionStats, setDetectionStats] = useState({
    totalFrames: 0,
    detectedObjects: 0,
    averageConfidence: 0
  })

  const menuItems = [
    { id: 'home', icon: '🏠', label: 'Trang Chủ', active: true },
    { id: 'webcam', icon: '📹', label: 'Nhận dạng Webcam' },
    { id: 'upload', icon: '📁', label: 'Tải Video lên' },
    { id: 'videos', icon: '🎬', label: 'Quản lý Video' },
    { id: 'history', icon: '📋', label: 'Lịch sử' },
    { id: 'settings', icon: '⚙️', label: 'Cài đặt' }
  ]

  const renderContent = () => {
    switch(activeTab) {
      case 'home':
        return (
          <div className="home-content">
            <div className="welcome-section">
              <h1>Trang chủ nhận dạng</h1>
              <p>Hệ thống nhận diện cảm xúc triễn làm A80</p>
            </div>
            <Statistics stats={detectionStats} />
          </div>
        )
      case 'webcam':
        return (
          <div className="content-section">
            <h1>📹 Nhận dạng Webcam thời gian thực</h1>
            <p className="subtitle">Nhận dạng đối tượng thời gian thực sử dụng webcam của bạn</p>
            <WebcamDetection onStats={setDetectionStats} />
          </div>
        )
      case 'upload':
        return (
          <div className="content-section">
            <h1>📁 Tải Video lên</h1>
            <p className="subtitle">Tải lên và phân tích các tệp video</p>
            <VideoUpload onStats={setDetectionStats} />
          </div>
        )
      case 'videos':
        return (
          <div className="content-section">
            <VideoManagement />
          </div>
        )
      case 'statistics':
        return (
          <div className="content-section">
            <h1>📊 Tổng quan Thống kê</h1>
            <p className="subtitle">Xem phân tích chi tiết và các chỉ số</p>
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
            <h1>📋 Lịch sử Nhận dạng</h1>
            <p className="subtitle">Xem kết quả nhận dạng trước đây</p>
            <div className="history-list">
              <div className="history-item">
                <span className="history-icon">🎥</span>
                <div className="history-info">
                  <strong>Phân tích Video - traffic.mp4</strong>
                  <p>Đã phát hiện 245 đối tượng | 20 Tháng 12, 2025 14:30</p>
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
            <h1>⚙️ Cài đặt</h1>
            <p className="subtitle">Cấu hình các tham số nhận dạng</p>
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
    <div className="dashboard-container" style={{
      backgroundImage: `url(${backgroundImage})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat'
    }}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-container">
            <img src={logo} alt="Logo" className="sidebar-logo" />
          </div>
          <h2>A80</h2>
          <p>Nhận diện cảm xúc AI</p>
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
              <p>Quản trị viên</p>
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