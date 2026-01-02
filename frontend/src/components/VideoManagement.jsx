import { useState, useEffect } from 'react'
import axios from 'axios'
import './VideoManagement.css'

function VideoManagement() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // Search & Filter states
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState('date')
  const [sortOrder, setSortOrder] = useState('desc')
  const [statusFilter, setStatusFilter] = useState('')
  const [zoneFilter, setZoneFilter] = useState('')
  
  // Modal states
  const [showModal, setShowModal] = useState(false)
  const [selectedVideo, setSelectedVideo] = useState(null)
  const [statistics, setStatistics] = useState(null)

  // Fetch videos
  const fetchVideos = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const params = {
        search: searchTerm || undefined,
        sort_by: sortBy,
        order: sortOrder,
        status: statusFilter || undefined,
        zone_id: zoneFilter || undefined
      }
      
      // Remove undefined params
      Object.keys(params).forEach(key => 
        params[key] === undefined && delete params[key]
      )
      
      const response = await axios.get('http://localhost:8000/api/videos', { params })
      setVideos(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Không thể tải danh sách video')
      console.error('Error fetching videos:', err)
    } finally {
      setLoading(false)
    }
  }

  // Load videos on mount and when filters change
  useEffect(() => {
    fetchVideos()
  }, [searchTerm, sortBy, sortOrder, statusFilter, zoneFilter])

  // View video details
  const handleViewDetails = async (video) => {
    setSelectedVideo(video)
    setShowModal(true)
    
    // Fetch statistics for this video
    try {
      const response = await axios.get(`http://localhost:8000/api/videos/${video.idvideo}/statistics`)
      setStatistics(response.data)
    } catch (err) {
      console.error('Error fetching statistics:', err)
      setStatistics(null)
    }
  }

  // Delete video
  const handleDelete = async (videoId) => {
    if (!window.confirm('Bạn có chắc muốn xóa video này?')) return
    
    try {
      await axios.delete(`http://localhost:8000/api/videos/${videoId}`)
      fetchVideos() // Reload list
      alert('Xóa video thành công!')
    } catch (err) {
      alert('Không thể xóa video: ' + (err.response?.data?.detail || err.message))
    }
  }

  // Update video status
  const handleUpdateStatus = async (video, newStatus) => {
    try {
      await axios.put(`http://localhost:8000/api/videos/${video.idvideo}`, {
        video_name: video.video_name,
        zone_id: video.zone_id,
        duration: video.duration,
        file_path: video.file_path,
        status: newStatus
      })
      fetchVideos() // Reload list
    } catch (err) {
      alert('Không thể cập nhật trạng thái: ' + (err.response?.data?.detail || err.message))
    }
  }

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString('vi-VN')
  }

  // Format duration
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A'
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="video-management">
      <div className="vm-header">
        <h1>📹 Quản lý Video</h1>
        <p className="vm-subtitle">Tìm kiếm, lọc và quản lý các video đã tải lên</p>
      </div>

      {/* Search & Filter Bar */}
      <div className="vm-controls">
        <div className="vm-search-box">
          <input
            type="text"
            placeholder="🔍 Tìm kiếm theo tên video..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="vm-search-input"
          />
        </div>

        <div className="vm-filters">
          <div className="vm-filter-group">
            <label>Sắp xếp theo:</label>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="date">Ngày tải lên</option>
              <option value="video_name">Tên video</option>
              <option value="duration">Thời lượng</option>
              <option value="status">Trạng thái</option>
            </select>
          </div>

          <div className="vm-filter-group">
            <label>Thứ tự:</label>
            <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value)}>
              <option value="desc">Giảm dần</option>
              <option value="asc">Tăng dần</option>
            </select>
          </div>

          <div className="vm-filter-group">
            <label>Trạng thái:</label>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="">Tất cả</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          <div className="vm-filter-group">
            <label>Zone:</label>
            <input
              type="number"
              placeholder="Zone ID"
              value={zoneFilter}
              onChange={(e) => setZoneFilter(e.target.value)}
              className="vm-zone-input"
            />
          </div>

          <button onClick={fetchVideos} className="vm-refresh-btn">
            🔄 Làm mới
          </button>
        </div>
      </div>

      {/* Video List */}
      {loading && <div className="vm-loading">⏳ Đang tải...</div>}
      
      {error && (
        <div className="vm-error">
          ❌ {error}
          <button onClick={fetchVideos} className="vm-retry-btn">Thử lại</button>
        </div>
      )}

      {!loading && !error && videos.length === 0 && (
        <div className="vm-empty">
          📭 Không tìm thấy video nào
        </div>
      )}

      {!loading && !error && videos.length > 0 && (
        <div className="vm-table-container">
          <table className="vm-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Tên Video</th>
                <th>Zone</th>
                <th>Thời lượng</th>
                <th>Ngày tải lên</th>
                <th>Trạng thái</th>
                <th>Hành động</th>
              </tr>
            </thead>
            <tbody>
              {videos.map((video) => (
                <tr key={video.idvideo}>
                  <td className="vm-text-black">{video.idvideo}</td>
                  <td className="vm-video-name vm-text-black">
                    <strong>{video.video_name}</strong>
                  </td>
                  <td className="vm-text-black">{video.zone_id || 'N/A'}</td>
                  <td className="vm-text-black">{formatDuration(video.duration)}</td>
                  <td className="vm-text-black">{formatDate(video.date)}</td>
                 <td>
                    <span
                      className={`vm-status ${
                        video.status === 'process'
                          ? 'vm-status-success'
                          : 'vm-status-error'
                      }`}
                    >
                      {video.status === 'process' ? (
                        <>
                          <i className="vm-icon success">✔</i> Processed
                        </>
                      ) : (
                        <>
                          <i className="vm-icon error">✖</i> Unprocessed
                        </>
                      )}
                    </span>
                  </td>
                  <td className="vm-actions">
                    <button
                      onClick={() => handleViewDetails(video)}
                      className="vm-btn vm-btn-view"
                      title="Xem chi tiết"
                    >
                      👁️
                    </button>
                    
                    <div className="vm-status-dropdown">
                      <button className="vm-btn vm-btn-status" title="Đổi trạng thái">
                        ⚙️
                      </button>
                      <div className="vm-dropdown-content">
                        <button onClick={() => handleUpdateStatus(video, 'pending')}>
                          Pending
                        </button>
                        <button onClick={() => handleUpdateStatus(video, 'processing')}>
                          Processing
                        </button>
                        <button onClick={() => handleUpdateStatus(video, 'completed')}>
                          Completed
                        </button>
                        <button onClick={() => handleUpdateStatus(video, 'failed')}>
                          Failed
                        </button>
                      </div>
                    </div>

                    <button
                      onClick={() => handleDelete(video.idvideo)}
                      className="vm-btn vm-btn-delete"
                      title="Xóa"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Detail Modal */}
      {showModal && selectedVideo && (
        <div className="vm-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="vm-modal" onClick={(e) => e.stopPropagation()}>
            <div className="vm-modal-header">
              <h2>Chi tiết Video</h2>
              <button onClick={() => setShowModal(false)} className="vm-modal-close">
                ✕
              </button>
            </div>
            
            <div className="vm-modal-body">
              <div className="vm-detail-grid">
                <div className="vm-detail-item">
                  <strong>ID:</strong>
                  <span>{selectedVideo.idvideo}</span>
                </div>
                <div className="vm-detail-item">
                  <strong>Tên video:</strong>
                  <span>{selectedVideo.video_name}</span>
                </div>
                <div className="vm-detail-item">
                  <strong>Zone ID:</strong>
                  <span>{selectedVideo.zone_id || 'N/A'}</span>
                </div>
                <div className="vm-detail-item">
                  <strong>Thời lượng:</strong>
                  <span>{formatDuration(selectedVideo.duration)}</span>
                </div>
                <div className="vm-detail-item">
                  <strong>Ngày tải lên:</strong>
                  <span>{formatDate(selectedVideo.date)}</span>
                </div>
                <div className="vm-detail-item">
                  <strong>Trạng thái:</strong>
                  <span className={`vm-status vm-status-${selectedVideo.status?.toLowerCase()}`}>
                    {selectedVideo.status}
                  </span>
                </div>
                <div className="vm-detail-item vm-detail-full">
                  <strong>Đường dẫn:</strong>
                  <span>{selectedVideo.file_path || 'N/A'}</span>
                </div>
              </div>

              {/* Statistics Section */}
              <div className="vm-statistics-section">
                <h3>📊 Thống kê cảm xúc</h3>
                {statistics && statistics.length > 0 ? (
                  <div className="vm-stats-list">
                    {statistics.map((stat, idx) => (
                      <div key={idx} className="vm-stat-card">
                        <div className="vm-stat-row">
                          <strong>Tổng khách:</strong>
                          <span>{stat.total_visitor || 0}</span>
                        </div>
                        <div className="vm-emotion-grid">
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😊 Vui:</span>
                            <span className="vm-emotion-value">
                              {(stat.happy_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😐 Bình thường:</span>
                            <span className="vm-emotion-value">
                              {(stat.neutral_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😠 Tức giận:</span>
                            <span className="vm-emotion-value">
                              {(stat.angry_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😢 Buồn:</span>
                            <span className="vm-emotion-value">
                              {(stat.sad_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😲 Ngạc nhiên:</span>
                            <span className="vm-emotion-value">
                              {(stat.surprise_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">😨 Sợ hãi:</span>
                            <span className="vm-emotion-value">
                              {(stat.fear_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="vm-emotion-item">
                            <span className="vm-emotion-label">🤢 Ghê tởm:</span>
                            <span className="vm-emotion-value">
                              {(stat.disgust_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="vm-no-stats">
                    Chưa có dữ liệu thống kê cho video này
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default VideoManagement
