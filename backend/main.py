
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from passlib.context import CryptContext
import sqlite3
import os
from datetime import datetime
from contextlib import asynccontextmanager
from inference import SketchRetrievalModel
import tempfile
import shutil
import base64
import uuid
import requests
DATABASE = "auth.db"
# Biến toàn cục lưu model AI
model = None


def init_db():
    """
    Khởi tạo database và Load model AI
    """
    global model
    if not os.path.exists(DATABASE): 
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # ===== Bảng users: lưu thông tin tài khoản =====
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database khởi tạo thành công")
    
    # Load model AI trong khối try-except để tránh sập server nếu thiếu file
    try:
        model = SketchRetrievalModel(
            model_path="clip_triplet_hard_margin_0_4.pth",
            embeddings_path="photo_embeddings_hard.pt",
            paths_json_path="photo_paths_hard.json",
        )
        print("✅ Model AI load thành công")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        model = None

# ========== QUẢN LÝ VÒNG ĐỜI ỨNG DỤNG ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    print("🛑 Ứng dụng đang dừng...")

# ========== TẠO FASTAPI APP ==========
app = FastAPI(lifespan=lifespan)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Mount thư mục chứa ảnh tĩnh
static_dir = os.path.join(os.path.dirname(__file__), "data", "photo", "photo", "photo")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"✅ Đã mount thư mục static: {static_dir}")
else:
    print(f"⚠️ Không tìm thấy thư mục static: {static_dir}")

# Cấu hình mã hóa mật khẩu
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ========== CÁC MODEL DỮ LIỆU ==========
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class SearchRequest(BaseModel):
    category: str
    sketch_data: str
    username: str
class LensRequest(BaseModel):
    image_url: str

# ==================== CÁC HÀM QUẢN LÝ DATABASE ====================
def get_user(username: str):
    conn = sqlite3.connect(DATABASE) 
    cursor = conn.cursor() 
    cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,)) 
    user = cursor.fetchone() 
    conn.close()
    return user

def create_user(username: str, password: str):
    password_hash = pwd_context.hash(password)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)

def generate_token(username: str) -> str:
    import base64
    token_data = f"{username}:{datetime.now().isoformat()}"
    token = base64.b64encode(token_data.encode()).decode()
    return token

# ==================== CÁC API ENDPOINTS ====================
@app.get("/")
def read_root():
    return {
        "message": "🔐 Draw & Find Authentication Backend",
        "endpoints": {
            "register": "POST /register",
            "login": "POST /login",
            "predict": "POST /predict (upload sketch image)",
            "search": "POST /search (Canvas Base64)",
            "health": "GET /health"
        }
    }

@app.post("/register")
async def register(user: UserRegister):
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username và password không được trống")
    if len(user.username) < 3:
        raise HTTPException(status_code=400, detail="Username phải ít nhất 3 ký tự")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Mật khẩu phải ít nhất 6 ký tự")
    if get_user(user.username):
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")
    
    success = create_user(user.username, user.password)
    if not success:
        raise HTTPException(status_code=400, detail="Lỗi khi tạo tài khoản")
    
    access_token = generate_token(user.username)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
async def login(user: UserLogin):
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username và password không được trống")
    
    db_user = get_user(user.username)
    if not db_user:
        raise HTTPException(status_code=401, detail="Tên đăng nhập không tồn tại")
    
    user_id, username, password_hash = db_user
    if not verify_password(user.password, password_hash):
        raise HTTPException(status_code=401, detail="Mật khẩu không chính xác")
    
    access_token = generate_token(username)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/search") 
async def search(request: SearchRequest):
    """
    API Tìm kiếm dùng Base64 từ Canvas với AI thật
    """
    if request.category not in ['animal', 'product']:
        raise HTTPException(status_code=400, detail="Category phải là 'animal' hoặc 'product'")
    
    if not request.username:
        raise HTTPException(status_code=400, detail="Username không được trống")
    
    db_user = get_user(request.username)
    if not db_user:
        raise HTTPException(status_code=401, detail="User không tồn tại hoặc phiên làm việc hết hạn")
    
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")

    # Xử lý Base64 thành file tạm an toàn bằng tempfile
    temp_dir = tempfile.mkdtemp()
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(temp_dir, filename)
    
    try: 
        sketch_data = request.sketch_data
        if "," in sketch_data:
            sketch_data = sketch_data.split(",")[1]
            
        img_bytes = base64.b64decode(sketch_data)
        
        with open(filepath, "wb") as f:
            f.write(img_bytes)
            
        # Đưa ảnh vào Core AI để tìm kiếm
        prediction_result = model.predict(filepath, category=request.category, top_k=5)
        
        # Định dạng lại kết quả cho frontend hiển thị
        formatted_results = [
            {
                "id": item["rank"],
                "name": f"{item['class_name']}_{str(item['rank']).zfill(3)}",
                "type": item["class_name"],
                "image_url": item["image_url"],
                "match_score": round(item["score"], 4)
            }
            for item in prediction_result["results"]
        ]
        
        return {
            "category": request.category,
            "results": formatted_results,
            "total_count": len(formatted_results),
            "predicted_class": prediction_result["predicted_class"],
            "message": f"Tìm kiếm thành công. Dự đoán thuộc nhóm: {prediction_result['predicted_class']}"
        }
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống AI: {str(e)}")
    finally:
        # LUÔN LUÔN DỌN SẠCH FILE TẠM ĐỂ TRÁNH TRÀN BỘ NHỚ SERVER
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/predict")
async def predict(file: UploadFile = File(...), category: str = Form(...)):
    """
    API Test upload file ảnh trực tiếp
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        raise HTTPException(status_code=400, detail="File phải là ảnh (PNG, JPG, JPEG, BMP)")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        prediction_result = model.predict(temp_path, category=category, top_k=5)
        
        formatted_results = [
            {
                "image_url": item["image_url"],
                "class_name": item["class_name"],
                "score": item["score"],
                "rank": item["rank"]
            }
            for item in prediction_result["results"]
        ]
        
        return {
            "results": formatted_results,
            "predicted_class": prediction_result["predicted_class"],
            "total_count": len(formatted_results)
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi prediction: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/health")
def health_check():
    """Kiểm tra trạng thái server"""
    return {"status": "🟢 Backend is running"}
@app.post("/upload-to-lens")
async def upload_to_lens(req: LensRequest):
    """
    API Tự động tải ảnh lên ImgBB để lấy link Public cho Google Lens
    """
    # ⚠️ THAY API KEY CỦA BẠN VÀO DÒNG DƯỚI ĐÂY:
    IMGBB_API_KEY = "d8295bd09d6444c6d378d89b89d97c66" 
    
    # 1. Xử lý đường dẫn: Biến "/static/..." thành đường dẫn thực tế trong máy
    url_path = req.image_url
    if "/static/" in url_path:
        relative_path = url_path.split("/static/")[1]
    else:
        relative_path = url_path # Đề phòng frontend gửi link kiểu khác
        
    # Biến static_dir đã được bạn khai báo ở dòng 55 trong code cũ
    local_file_path = os.path.join(static_dir, relative_path)
    
    if not os.path.exists(local_file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file ảnh gốc trên server")
        
    try:
        # 2. Đọc file ảnh và mã hóa Base64
        with open(local_file_path, "rb") as file:
            encoded_image = base64.b64encode(file.read())
            
        # 3. Đóng gói dữ liệu gửi lên ImgBB
        payload = {
            "key": IMGBB_API_KEY,
            "image": encoded_image
        }
        
        # 4. Gọi API của ImgBB
        response = requests.post("https://api.imgbb.com/1/upload", data=payload)
        result = response.json()
        
        if response.status_code == 200:
            public_url = result["data"]["url"]
            return {"status": "success", "public_url": public_url}
        else:
            raise HTTPException(status_code=400, detail="Lỗi từ máy chủ ImgBB")
            
    except Exception as e:
        print(f"❌ Upload to ImgBB error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ==================== CHẠY SERVER ====================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động Draw & Find Backend...")
    print("📍 Server chạy tại: http://localhost:8000")
    print("📚 API Docs (Swagger UI) tại: http://localhost:8000/docs")
    print("=" * 60)
    
    # Chạy server trên localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)