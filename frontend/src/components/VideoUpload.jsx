import { useState, useEffect } from 'react'
import axios from 'axios'
import './VideoUpload.css'

function VideoUpload({ onStats }) {
  const [file, setFile] = useState(null)
  const [zoneId, setZoneId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setResults(null)
    setError(null)
    setProgress(0)
  }

  const handleUpload = async () => {
    if (!file) return
    if (!zoneId) {
      setError('Vui lòng nhập Zone ID')
      return
    }

    setUploading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('zone_id', zoneId)

    try {
      // Upload và bắt đầu xử lý
      const response = await axios.post('http://localhost:8000/api/video/upload-process', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      setJobId(response.data.job_id)
      setStatus('queued')
      setUploading(false)
      setProcessing(true)
      
      console.log('✅ Upload thành công, Job ID:', response.data.job_id)
    } catch (error) {
      console.error('Upload error:', error)
      setError(error.response?.data?.detail || 'Upload thất bại. Kiểm tra backend có chạy không.')
      setUploading(false)
    }
  }

  // Poll status khi có jobId
  useEffect(() => {
    if (!jobId || !processing) return

    const pollInterval = setInterval(async () => {
      try {
      const response = await axios.get(`http://localhost:8000/api/video/process-status/${jobId}`)
        const data = response.data
        
        setStatus(data.status)
        setProgress(data.progress || 0)
        
        if (data.status === 'completed') {
          setProcessing(false)
          setResults(data.result)
          
          // Cập nhật stats
          onStats({
            totalFrames: data.result.total_frames,
            processedFrames: data.result.processed_frames,
            fps: data.result.fps,
            resolution: data.result.resolution.join('x')
          })
          
          console.log('✅ Video xử lý xong!')
        } else if (data.status === 'failed') {
          setProcessing(false)
          setError(data.error || 'Xử lý video thất bại')
          console.error('❌ Lỗi:', data.error)
        }
      } catch (error) {
        console.error('Lỗi khi kiểm tra status:', error)
      }
    }, 1000) // Poll mỗi 1 giây

    return () => clearInterval(pollInterval)
  }, [jobId, processing, onStats])

  const handleDownload = () => {
    if (!jobId) return
    window.open(`http://localhost:8001/api/video/download/${jobId}`, '_blank')
  }

  return (
    <div className="video-upload">
      <div className="upload-area">
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange}
          disabled={uploading || processing}
        />
        
        <div className="zone-input-group">
          <label htmlFor="zone-id">Zone ID:</label>
          <input 
            id="zone-id"
            type="number" 
            placeholder="Nhập Zone ID" 
            value={zoneId}
            onChange={(e) => setZoneId(e.target.value)}
            disabled={uploading || processing}
            min="1"
          />
        </div>
        <button 
          onClick={handleUpload} 
          disabled={!file || uploading || processing}
          className="btn-primary"
        >
          {uploading ? '📤 Đang tải lên...' : processing ? '⏳ Đang xử lý...' : '🚀 Tải lên & Phân tích'}
        </button>
      </div>

      {/* Progress Bar */}
      {processing && (
        <div className="progress-section">
          <div className="progress-bar-container">
            <div 
              className="progress-bar" 
              style={{ width: `${progress}%` }}
            >
              <span className="progress-text">{progress}%</span>
            </div>
          </div>
          <p className="status-text">
            📊 Trạng thái: <strong>{status === 'processing' ? 'Đang xử lý' : 'Đang chờ'}</strong>
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="error-message">
          <h3>❌ Lỗi</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="results">
          <h3>✅ Kết quả xử lý</h3>
          <div className="result-grid">
            <div className="result-item">
              <span className="label">Tổng số frame:</span>
              <strong>{results.total_frames}</strong>
            </div>
            <div className="result-item">
              <span className="label">Frame đã xử lý:</span>
              <strong>{results.processed_frames}</strong>
            </div>
            <div className="result-item">
              <span className="label">FPS:</span>
              <strong>{results.fps}</strong>
            </div>
            <div className="result-item">
              <span className="label">Độ phân giải:</span>
              <strong>{results.resolution[0]}x{results.resolution[1]}</strong>
            </div>
          </div>
          
          <button 
            onClick={handleDownload}
            className="btn-download"
          >
            📥 Tải video đã xử lý
          </button>
        </div>
      )}
    </div>
  )
}

export default VideoUpload