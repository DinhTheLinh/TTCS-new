
import { useState, useEffect } from 'react'; 
import { LogIn } from 'lucide-react'; 
import '../styles/AuthForm.css'; 


function LoginForm({ onLoginSuccess, onSwitchToRegister, onLoading }) {
 
  const [username, setUsername] = useState(''); 
  const [password, setPassword] = useState(''); 
  const [error, setError] = useState(''); 
  const [isLoading, setIsLoading] = useState(false); 
  const [registerSuccess, setRegisterSuccess] = useState(() => {
    const success = localStorage.getItem('registerSuccess');
    if (success) {
      localStorage.removeItem('registerSuccess');
      return true;
    }
    return false;
  });

  useEffect(() => {
    if (registerSuccess) {
      const timer = setTimeout(() => {
        setRegisterSuccess(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [registerSuccess]);

  
  const handleSubmit = async (e) => { 
    e.preventDefault(); 
    setError(''); 

    if (!username.trim()) {
      setError('Vui lòng nhập tên đăng nhập');
      return;
    }
    if (!password.trim()) {
      setError('Vui lòng nhập mật khẩu');
      return;
    }
    setIsLoading(true);
    onLoading(true);

    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST', 
        headers: {
          'Content-Type': 'application/json', 
        },
        body: JSON.stringify({ username, password }), 
      });
      
      if (!response.ok) { 
        const data = await response.json(); 
        setError(data.detail || 'Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin.');
        setIsLoading(false);
        onLoading(false);
        return;
      }


      const data = await response.json();
      localStorage.setItem('authToken', data.access_token || 'token_' + Date.now());
      localStorage.setItem('username', username);
      localStorage.setItem('isAuthenticated', 'true');
      setIsLoading(false);
      onLoading(false);
      onLoginSuccess();
    } catch {
      setError('Lỗi kết nối. Vui lòng kiểm tra backend.');
      setIsLoading(false);
      onLoading(false);
    }
  };


  return ( 
    <div className="auth-form-container">
      <div className="auth-form-box">
        {/* Logo của ứng dụng */}
        <div className="auth-logo">
          <span className="paper">📄</span>
          <span className="pencil">✏️</span>
        </div>
        
        {/* Tiêu đề của form */}
        <h2 className="auth-title">Đăng Nhập</h2>
        <p className="auth-subtitle">Draw & Find</p>

        {/* Hiển thị success message nếu vừa đăng ký */}
        {registerSuccess && (
          <div style={{
            backgroundColor: '#d4edda',
            color: '#155724',
            padding: '12px 16px',
            borderRadius: '6px',
            marginBottom: '16px',
            fontSize: '14px',
            textAlign: 'center',
            border: '1px solid #c3e6cb'
          }}>
            ✓ Đăng ký thành công! Vui lòng đăng nhập tại đây.
          </div>
        )}

        {/* Form đăng nhập */}
        <form onSubmit={handleSubmit}>
          {/* Trường nhập tên đăng nhập */}
          <div className="form-group">
            <label htmlFor="username">Tên đăng nhập</label>
            <input
              id="username"
              type="text"
              placeholder="Nhập tên đăng nhập"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>

          {/* Trường nhập mật khẩu */}
          <div className="form-group">
            <label htmlFor="password">Mật khẩu</label>
            <input
              id="password"
              type="password"
              placeholder="Nhập mật khẩu"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {/* Hiển thị thông báo lỗi nếu có */}
          {error && <div className="error-message">{error}</div>}

          {/* Nút đăng nhập */}
          <button type="submit" className="auth-submit-btn" disabled={isLoading}>
            {isLoading ? (
              <>
                <span className="spinner-small"></span>
                Đang đăng nhập...
              </>
            ) : (
              <>
                <LogIn size={18} />
                Đăng Nhập
              </>
            )}
          </button>
        </form>

        {/* Phần chuyển sang trang đăng ký */}
        <div className="auth-toggle">
          <p>Chưa có tài khoản?</p>
          <button
            type="button"
            className="toggle-btn"
            onClick={onSwitchToRegister}
          >
            Đăng ký tại đây
          </button>
        </div>
      </div>
    </div>
  );
}
export default LoginForm;
