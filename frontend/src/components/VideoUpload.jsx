import { useState } from 'react'
import axios from 'axios'

function VideoUpload({ onStats }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState(null)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('http://localhost:8001/predict_video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      setResults(response.data)
      
      // Tính toán thống kê
      const totalDetections = response.data.results.flat().length
      const avgScore = response.data.results.flat()
        .reduce((acc, obj) => acc + obj.score, 0) / totalDetections || 0
      
      onStats({
        totalFrames: response.data.frames,
        detectedObjects: totalDetections,
        averageConfidence: (avgScore * 100).toFixed(2)
      })
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed. Make sure backend is running on port 8000')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="video-upload">
      <div className="upload-area">
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange}
          disabled={uploading}
        />
        <button 
          onClick={handleUpload} 
          disabled={!file || uploading}
          className="btn-primary"
        >
          {uploading ? '⏳ Processing...' : '🚀 Upload & Analyze'}
        </button>
      </div>

      {results && (
        <div className="results">
          <h3>✅ Results</h3>
          <p>Total Frames: <strong>{results.frames}</strong></p>
          <p>Objects Detected: <strong>{results.results.flat().length}</strong></p>
        </div>
      )}
    </div>
  )
}

export default VideoUpload