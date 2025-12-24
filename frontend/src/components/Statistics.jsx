function Statistics({ stats }) {
  return (
    <div className="statistics">
      <div className="stat-card">
        <div className="stat-icon">📊</div>
        <div className="stat-content">
          <h3>{stats.totalFrames}</h3>
          <p>Total Frames</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">🎯</div>
        <div className="stat-content">
          <h3>{stats.detectedObjects}</h3>
          <p>Objects Detected</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">💯</div>
        <div className="stat-content">
          <h3>{stats.averageConfidence}%</h3>
          <p>Avg Confidence</p>
        </div>
      </div>
    </div>
  )
}

export default Statistics