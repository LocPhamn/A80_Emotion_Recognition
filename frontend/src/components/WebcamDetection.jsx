import { useState, useRef, useEffect } from 'react'

// Helper function: Lấy màu theo emotion
const getEmotionColor = (emotion) => {
  const colorMap = {
    'bình thường': '#4CAF50',      // Xanh lá
    'buồn bã': '#2196F3',        // Xanh dương
    'tức giận': '#F44336',      // Đỏ
    'bất ngờ': '#FF9800',   // Cam
    'sợ hãi': '#9C27B0',       // Tím
    'khó chịu': '#795548',    // Nâu
    'bình thường': '#9E9E9E'     // Xám
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
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSessionId, setRecordingSessionId] = useState(null)
  const [recordingFrameCount, setRecordingFrameCount] = useState(0)
  const [recordingStartTime, setRecordingStartTime] = useState(null)
  const [zoneId, setZoneId] = useState(1)
  
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
      
      // Xử lý recording status messages
      if (data.type === 'recording_started') {
        console.log('🔴 Recording started:', data.session_id)
        setIsRecording(true)
        setRecordingSessionId(data.session_id)
        setRecordingFrameCount(0)
        setRecordingStartTime(Date.now())
        return
      }
      
      if (data.type === 'recording_stopped') {
        console.log('⏹️ Recording stopped:', data.session_id, 'frames:', data.frame_count)
        setIsRecording(false)
        
        // Gọi API để lưu recording vào database
        handleRecordingComplete(data.session_id, data.frame_count)
        return
      }
      
      // Update recording frame count
      if (data.recording && data.frame_count) {
        setRecordingFrameCount(data.frame_count)
      }
      
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
  const minFrameInterval = 33; // Tối đa 30 FPS (33ms/frame)
  
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
    const scale = 0.75 // nếu muốn giảm độ phân giải gửi lên, thay 1 bằng 0.5 hoặc 0.75
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

  const startRecording = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Gửi command start recording qua WebSocket
      wsRef.current.send(JSON.stringify({
        command: 'start_recording'
      }))
      console.log('📤 Sent start_recording command')
    } else {
      console.error('❌ WebSocket not ready')
      setError('WebSocket chưa kết nối')
    }
  }

  const stopRecording = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Gửi command stop recording qua WebSocket
      wsRef.current.send(JSON.stringify({
        command: 'stop_recording'
      }))
      console.log('📤 Sent stop_recording command')
    } else {
      console.error('❌ WebSocket not ready')
      setError('WebSocket chưa kết nối')
    }
  }

  const handleRecordingComplete = async (sessionId, frameCount) => {
    console.log('💾 Saving recording to database...')
    
    try {
      // Tính duration từ frame count (giả sử 15 FPS)
      const duration = frameCount / 15.0
      
      // Tạo statistics giả (vì ta không có emotion tracking trong quá trình recording)
      // Trong thực tế, bạn có thể thu thập stats trong quá trình recording
      const statistics = {
        total_visitor: 0,
        emotion_ratios: {
          'angry': { ratio: 0 },
          'disgust': { ratio: 0 },
          'fear': { ratio: 0 },
          'happy': { ratio: 0 },
          'neutral': { ratio: 0 },
          'sad': { ratio: 0 },
          'surprise': { ratio: 0 }
        }
      }
      
      // Gọi API backend để lưu recording
      const formData = new FormData()
      formData.append('session_id', sessionId)
      formData.append('video_path', `webcam_${sessionId}.mp4`)
      formData.append('duration', duration.toString())
      formData.append('statistics', JSON.stringify(statistics))
      formData.append('zone_id', zoneId.toString())
      
      const response = await fetch('http://localhost:8000/api/webcam/save-recording', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Failed to save recording')
      }
      
      const result = await response.json()
      console.log('✅ Recording saved:', result)
      
      alert(`Recording đã lưu thành công!\nVideo ID: ${result.video_id}\nDuration: ${duration.toFixed(1)}s`)
      
    } catch (err) {
      console.error('❌ Failed to save recording:', err)
      setError('Không thể lưu recording: ' + err.message)
    }
  }

  const stopWebcam = () => {
    // Dừng recording nếu đang recording
    if (isRecording) {
      stopRecording()
    }
    
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
    setIsRecording(false)
    setRecordingSessionId(null)
    setRecordingFrameCount(0)
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
          <>
            <button onClick={stopWebcam} className="btn-danger">
              ⏹️ Ngừng Webcam
            </button>
            
            {!isRecording ? (
              <button 
                onClick={startRecording} 
                className="btn-success"
                style={{ marginLeft: '10px' }}
              >
                🔴 Bắt đầu Ghi
              </button>
            ) : (
              <button 
                onClick={stopRecording} 
                className="btn-warning"
                style={{ marginLeft: '10px' }}
              >
                ⏹️ Dừng Ghi
              </button>
            )}
            
            <div style={{ 
              display: 'inline-block', 
              marginLeft: '20px',
              fontSize: '0.9rem' 
            }}>
              <label htmlFor="zone-select">Zone ID: </label>
              <select 
                id="zone-select"
                value={zoneId} 
                onChange={(e) => setZoneId(parseInt(e.target.value))}
                disabled={isRecording}
                style={{
                  padding: '5px 10px',
                  borderRadius: '5px',
                  border: '1px solid #ccc',
                  marginLeft: '5px'
                }}
              >
                <option value={1}>Zone 1</option>
                <option value={2}>Zone 2</option>
                <option value={3}>Zone 3</option>
              </select>
            </div>
          </>
        )}
      </div>

      {isRecording && (
        <div style={{
          marginTop: '15px',
          padding: '15px',
          background: '#ffebee',
          borderRadius: '10px',
          border: '2px solid #f44336'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center',
            justifyContent: 'center',
            gap: '15px'
          }}>
            <div style={{
              width: '15px',
              height: '15px',
              borderRadius: '50%',
              background: '#f44336',
              animation: 'pulse 1.5s infinite'
            }}></div>
            <span style={{ fontWeight: 'bold', color: '#f44336' }}>
              🔴 ĐANG GHI VIDEO
            </span>
            <span style={{ color: '#666' }}>
              Frames: {recordingFrameCount}
            </span>
            <span style={{ color: '#666' }}>
              Duration: {recordingStartTime ? ((Date.now() - recordingStartTime) / 1000).toFixed(1) : 0}s
            </span>
          </div>
        </div>
      )}

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