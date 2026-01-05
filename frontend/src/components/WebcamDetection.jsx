import { useState, useRef, useEffect } from 'react'

// Helper function: Lấy màu theo emotion
const getEmotionColor = (emotion) => {
  const colorMap = {
    'happy': '#4CAF50',      // Xanh lá
    'sad': '#2196F3',        // Xanh dương
    'angry': '#F44336',      // Đỏ
    'surprise': '#FF9800',   // Cam
    'fear': '#9C27B0',       // Tím
    'disgust': '#795548',    // Nâu
    'neutral': '#9E9E9E'     // Xám
  }
  return colorMap[emotion] || '#00BCD4' // Cyan mặc định
}

// Helper function: Scale tọa độ bbox từ resolution gửi đi về resolution gốc
const scaleBboxCoordinates = (bbox, scaleFactor) => {
  // scaleFactor = 1 / scale gửi đi
  // Ví dụ: Gửi với scale=0.5 → scaleFactor=2
  return {
    x: bbox.x * scaleFactor,
    y: bbox.y * scaleFactor,
    width: bbox.width * scaleFactor,
    height: bbox.height * scaleFactor
  }
}

function WebcamDetection({ onStats }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fps, setFps] = useState(0)
  const [realFps, setRealFps] = useState(0)
  const [tracks, setTracks] = useState([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const lastFrameTimeRef = useRef(performance.now())
  const fpsHistoryRef = useRef([])
  const sendScaleRef = useRef(1) // Lưu scale factor để scale tọa độ

  const startWebcam = async () => {
    try {
      setError(null)
      setLoading(true)
      
      // 1. Khởi động webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      })

      streamRef.current = stream
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play()
            .then(() => {
              console.log('Video playing!')
              setIsActive(true)
              setLoading(false)
              
              // 2. Kết nối WebSocket
              connectWebSocket()
            })
            .catch(err => {
              console.error('Play error:', err)
              setError('Failed to play: ' + err.message)
              setLoading(false)
            })
        }
      }
      
    } catch (err) {
      console.error('Webcam error:', err)
      setLoading(false)
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access.')
      } else if (err.name === 'NotFoundError') {
        setError('No camera found on this device.')
      } else {
        setError('Cannot access webcam: ' + err.message)
      }
      setIsActive(false)
    }
  }

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8001/ws/process')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected!')
      startSendingFrames()
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      const now = performance.now()
      const deltaTime = now - lastFrameTimeRef.current
      lastFrameTimeRef.current = now
      
      if (deltaTime > 0) {
        const instantFps = 1000 / deltaTime
        fpsHistoryRef.current.push(instantFps)
        
        if (fpsHistoryRef.current.length > 30) {
          fpsHistoryRef.current.shift()
        }
        
        const avgFps = fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length
        setRealFps(avgFps)
      }
      
      // ✅ VẼ VIDEO GỐC + BBOX + EMOTION
      if (canvasRef.current && videoRef.current && videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const video = videoRef.current
        
        // Set canvas size theo video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
        
        // Vẽ video gốc
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        
        // Vẽ bbox + emotion cho mỗi track
        if (data.tracks && data.tracks.length > 0) {
          // Tính scale factor để scale tọa độ về resolution gốc
          const scaleFactor = 1 / sendScaleRef.current
          
          data.tracks.forEach(track => {
            const { bbox, emotion, confidence, id } = track
            
            // ✅ Scale tọa độ bbox về resolution gốc
            const scaledBbox = scaleBboxCoordinates(bbox, scaleFactor)
            
            // Lấy màu theo emotion
            const color = getEmotionColor(emotion)
            
            // Vẽ bounding box
            ctx.strokeStyle = color
            ctx.lineWidth = 3
            ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)
            
            // Vẽ background cho text
            const text = `ID:${id} ${emotion} ${(confidence * 100).toFixed(0)}%`
            ctx.font = 'bold 16px Arial'
            const textMetrics = ctx.measureText(text)
            const textHeight = 20
            
            ctx.fillStyle = color
            ctx.fillRect(scaledBbox.x, scaledBbox.y - textHeight - 5, textMetrics.width + 10, textHeight + 5)
            
            // Vẽ text
            ctx.fillStyle = '#ffffff'
            ctx.fillText(text, scaledBbox.x + 5, scaledBbox.y - 8)
          })
        }
      }

      // Cập nhật stats
      setFps(data.fps)
      setTracks(data.tracks || [])
      
      // Gửi stats lên parent component
      if (onStats) {
        onStats({
          fps: data.fps,
          totalFaces: data.tracks?.length || 0,
          emotions: data.tracks?.reduce((acc, track) => {
            acc[track.emotion] = (acc[track.emotion] || 0) + 1
            return acc
          }, {})
        })
      }
    }

    ws.onerror = (err) => {
      console.error('❌ WebSocket error:', err)
      setError('Connection error')
    }

    ws.onclose = () => {
      console.log('🔴 WebSocket closed')
    }
  }

  const startSendingFrames = () => {
  let isProcessing = false;
  let lastSendTime = performance.now();
  const minFrameInterval = 50; // Tối đa 20 FPS (50ms/frame)
  
  const sendFrame = () => {
    const now = performance.now();
    const timeSinceLastSend = now - lastSendTime;
    
    // THROTTLE: Đảm bảo không gửi quá nhanh
    if (isProcessing || timeSinceLastSend < minFrameInterval || 
        !videoRef.current || wsRef.current?.readyState !== WebSocket.OPEN) {
      requestAnimationFrame(sendFrame);
      return;
    }
    
    isProcessing = true;
    lastSendTime = now;
    
    const canvas = document.createElement('canvas');
    const scale = 1 // nếu muốn giảm độ phân giải gửi lên, thay 1 bằng 0.5 hoặc 0.75
    sendScaleRef.current = scale // Lưu scale để dùng khi scale tọa độ bbox
    canvas.width = videoRef.current.videoWidth * scale;
    canvas.height = videoRef.current.videoHeight * scale;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(blob);
      }
      isProcessing = false;
    }, 'image/jpeg', 0.6); // Giảm từ 0.8 xuống 0.6
    
    requestAnimationFrame(sendFrame);
  };
  
  requestAnimationFrame(sendFrame);
};

  const stopWebcam = () => {
    // Dừng gửi frames
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }

    // Đóng WebSocket
    if (wsRef.current) {
      wsRef.current.close()
    }

    // Dừng webcam
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    setIsActive(false)
    setLoading(false)
    setError(null)
    setFps(0)
    setRealFps(0)
    setTracks([])
    fpsHistoryRef.current = []
  }

  useEffect(() => {
    return () => {
      stopWebcam()
    }
  }, [])

  return (
    <div className="webcam-section">
      <div className="video-container" style={{ position: 'relative' }}>
        {/* Video gốc - ẩn đi nhưng vẫn render để vẽ lên canvas */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline
          muted
          style={{ display: 'none' }}
        />
        
        {/* Canvas hiển thị kết quả */}
        <canvas 
          ref={canvasRef}
          style={{
            width: '100%',
            height: 'auto',
            minHeight: '400px',
            maxHeight: '600px',
            borderRadius: '10px',
            background: '#000',
            display: isActive ? 'block' : 'none'
          }}
        />
        
        {!isActive && (
          <div style={{
            position: 'relative',
            width: '100%',
            minHeight: '400px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#000',
            borderRadius: '10px'
          }}>
            <div className="placeholder-content">
              {loading ? (
                <>
                  <span className="placeholder-icon">⏳</span>
                  <p>Đang tải webcam...</p>
                </>
              ) : (
                <>
                  <span className="placeholder-icon">📹</span>
                  <p>Click bắt đầu webcam</p>
                  {error && (
                    <p style={{ 
                      color: '#ff4757', 
                      marginTop: '15px',
                      padding: '10px 20px',
                      background: 'rgba(255,71,87,0.1)',
                      borderRadius: '8px',
                      fontSize: '0.9rem',
                      maxWidth: '400px'
                    }}>
                       {error}
                    </p>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
      
      <div className="controls">
        {!isActive ? (
          <button 
            onClick={startWebcam} 
            className="btn-success"
            disabled={loading}
          >
            {loading ? '⏳ Đang tải...' : '▶️ Bắt đầu Webcam'}
          </button>
        ) : (
          <button onClick={stopWebcam} className="btn-danger">
            ⏹️ Ngừng Webcam
          </button>
        )}
      </div>

      {isActive && (
        <div style={{
          marginTop: '20px',
          padding: '15px',
          background: '#e8f5e9',
          borderRadius: '10px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-around' }}>
            <span style={{ color: '#4caf50', fontWeight: 'bold' }}>
              🟢 Backend FPS: {fps.toFixed(1)}
            </span>
            <span style={{ color: realFps < 10 ? '#ff4757' : '#2196f3', fontWeight: 'bold' }}>
              📊 Real FPS: {realFps.toFixed(1)}
            </span>
            <span style={{ color: '#2196f3', fontWeight: 'bold' }}>
              👥 Faces: {tracks.length}
            </span>
          </div>
          
          {realFps < 10 && (
            <div style={{ 
              marginTop: '10px', 
              padding: '8px',
              background: 'rgba(255,71,87,0.1)',
              borderRadius: '5px',
              color: '#ff4757',
              fontSize: '0.85rem',
              textAlign: 'center'
            }}>
              FPS thực tế thấp hơn 10. Vui lòng kiểm tra kết nối mạng hoặc giảm độ phân giải webcam để cải thiện hiệu suất.
            </div>
          )}
          
          {tracks.length > 0 && (
            <div style={{ marginTop: '10px', fontSize: '0.9rem' }}>
              {tracks.map(track => (
                <div key={track.id} style={{ 
                  display: 'inline-block', 
                  margin: '5px',
                  padding: '5px 10px',
                  background: 'rgba(255,255,255,0.7)',
                  borderRadius: '5px'
                }}>
                  ID:{track.id} - {track.emotion} ({(track.confidence * 100).toFixed(0)}%)
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default WebcamDetection