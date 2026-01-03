function Statistics({ stats }) {
  return (
    <div className="statistics">
      <div className="stat-card">
        <div className="stat-icon">📊</div>
        <div className="stat-content">
          <h3>{stats.totalFrames}</h3>
          <p>Tổng Frames</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">🎯</div>
        <div className="stat-content">
          <h3>{stats.detectedObjects}</h3>
          <p>Số đối tượng phát hiện được</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">💯</div>
        <div className="stat-content">
          <h3>{stats.averageConfidence}%</h3>
          <p>Trung bình độ tự tin dự đoán</p>
        </div>
      </div>
    </div>
  )
}

export default Statistics