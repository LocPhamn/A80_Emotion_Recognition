import { useState } from 'react'
import './Login.css'

function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    
    // Validation
    if (!username.trim()) {
      setError('Vui lòng nhập tên đăng nhập')
      return
    }
    
    if (!password) {
      setError('Vui lòng nhập mật khẩu')
      return
    }

    setLoading(true)
    
    try {
      // Call API login
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)
      
      const response = await fetch('http://localhost:8000/api/login', {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      
      if (data.success) {
        onLogin({ 
          username: data.username, 
          role: data.role,
          user_id: data.user_id 
        })
      } else {
        setError(data.message || 'Tên đăng nhập hoặc mật khẩu không đúng')
        setLoading(false)
      }
    } catch (error) {
      console.error('Login error:', error)
      setError('Lỗi kết nối đến server')
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-background">
        {/* Animated background elements */}
        <div className="bg-animation"></div>
      </div>
      
      <div className="login-box">
       <div className="login-header">
          <h1>AI Vision</h1>
          <p>Admin Panel</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error">
              <span>⚠️</span>
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="username">
              <span className="icon">👤</span>
              Tên đăng nhập
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Nhập tên đăng nhập"
              disabled={loading}
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">
              <span className="icon">🔒</span>
              Mật khẩu
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Nhập mật khẩu"
              disabled={loading}
              autoComplete="current-password"
            />
          </div>

          <button 
            type="submit" 
            className="login-button"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Đang đăng nhập...
              </>
            ) : (
              'Đăng nhập'
            )}
          </button>
        </form>

        <div className="login-footer">
          <p>© 2026 AI Vision System</p>
        </div>
      </div>
    </div>
  )
}

export default Login
