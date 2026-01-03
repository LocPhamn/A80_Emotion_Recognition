import { useState, useRef, useEffect } from 'react'

function WebcamDetection({ onStats }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fps, setFps] = useState(0)
  const [realFps, setRealFps] = useState(0) // ✅ FPS thực tế từ frontend
  const [tracks, setTracks] = useState([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const lastFrameTimeRef = useRef(performance.now())
  const fpsHistoryRef = useRef([])

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
      console.log('✅ WebSocket connected!')
      startSendingFrames()
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      // ✅ TÍNH FPS THỰC TẾ Ở FRONTEND
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
      
      // Hiển thị frame đã xử lý
      if (data.frame && canvasRef.current) {
        const img = new Image()
        img.onload = () => {
          const canvas = canvasRef.current
          const ctx = canvas.getContext('2d')
          canvas.width = img.width
          canvas.height = img.height
          ctx.drawImage(img, 0, 0)
        }
        img.src = 'data:image/jpeg;base64,' + data.frame
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
    
    // ✅ THROTTLE: Đảm bảo không gửi quá nhanh
    if (isProcessing || timeSinceLastSend < minFrameInterval || 
        !videoRef.current || wsRef.current?.readyState !== WebSocket.OPEN) {
      requestAnimationFrame(sendFrame);
      return;
    }
    
    isProcessing = true;
    lastSendTime = now;
    
    const canvas = document.createElement('canvas');
    const scale = 0.5; // 640x360 thay vì 1280x720
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
        {/* Video gốc - ẩn đi */}
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