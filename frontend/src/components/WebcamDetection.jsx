import { useState, useRef, useEffect } from 'react'

function WebcamDetection({ onStats }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fps, setFps] = useState(0)
  const [tracks, setTracks] = useState([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

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
              console.log('✅ Video playing!')
              setIsActive(true)
              setLoading(false)
              
              // 2. Kết nối WebSocket
              connectWebSocket()
            })
            .catch(err => {
              console.error('❌ Play error:', err)
              setError('Failed to play: ' + err.message)
              setLoading(false)
            })
        }
      }
      
    } catch (err) {
      console.error('❌ Webcam error:', err)
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
    const ws = new WebSocket('ws://localhost:8000/ws')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('✅ WebSocket connected!')
      startSendingFrames()
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
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
  let isProcessing = false; // Thêm flag để tránh gửi liên tục
  
  const sendFrame = () => {
    if (isProcessing || !videoRef.current || wsRef.current?.readyState !== WebSocket.OPEN) {
      requestAnimationFrame(sendFrame);
      return;
    }
    
    isProcessing = true;
    
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
    setTracks([])
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
                  <p>Loading webcam...</p>
                </>
              ) : (
                <>
                  <span className="placeholder-icon">📹</span>
                  <p>Click start to begin detection</p>
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
                      ⚠️ {error}
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
            {loading ? '⏳ Loading...' : '▶️ Start Webcam'}
          </button>
        ) : (
          <button onClick={stopWebcam} className="btn-danger">
            ⏹️ Stop Webcam
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
              🟢 FPS: {fps.toFixed(1)}
            </span>
            <span style={{ color: '#2196f3', fontWeight: 'bold' }}>
              👥 Faces: {tracks.length}
            </span>
          </div>
          
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