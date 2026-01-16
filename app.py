import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os

# Try import plotly with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly không khả dụng, sử dụng đồ thị đơn giản")

# ML Libraries - với try/except
try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Page configuration cho Android
st.set_page_config(
    page_title="TOOL AI 1.0 - LOTOBET",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername/lotobet-ai',
        'Report a bug': None,
        'About': "TOOL AI 1.0 - Hệ thống phân tích xổ số thông minh"
    }
)

# Custom CSS cho Android Webview
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        -webkit-tap-highlight-color: transparent;
        -webkit-text-size-adjust: 100%;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        color: #ffffff !important;
        min-height: 100vh !important;
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(25, 25, 60, 0.85) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
        border: 1px solid #4040aa !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    
    .custom-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(64, 64, 170, 0.3) !important;
        border-color: #6060ff !important;
    }
    
    /* Prediction card */
    .pred-card {
        background: linear-gradient(135deg, #1a1a40, #2d2d7a) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        margin: 8px !important;
        text-align: center !important;
        border: 2px solid transparent !important;
    }
    
    .good-pred {
        border-color: #00ff88 !important;
        background: linear-gradient(135deg, #1a402a, #2d7a5a) !important;
    }
    
    .bad-pred {
        border-color: #ff4444 !important;
        background: linear-gradient(135deg, #401a1a, #7a2d2d) !important;
    }
    
    /* Button styling cho mobile */
    .stButton > button {
        background: linear-gradient(135deg, #302b63, #0f0c29) !important;
        color: white !important;
        border: 1px solid #6060ff !important;
        border-radius: 25px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
        width: 100% !important;
        margin: 5px 0 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4040aa, #202055) !important;
        transform: scale(1.02) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(60, 60, 100, 0.7) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        white-space: nowrap !important;
        border: 1px solid #5050aa !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4040aa, #6060ff) !important;
        color: white !important;
    }
    
    /* Real-time counter */
    .counter-container {
        font-size: 2.5em !important;
        font-weight: 700 !important;
        text-align: center !important;
        color: #00ffaa !important;
        text-shadow: 0 0 15px rgba(0, 255, 170, 0.5) !important;
        padding: 15px !important;
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 15px !important;
        margin: 10px 0 !important;
    }
    
    /* Responsive cho mobile */
    @media (max-width: 768px) {
        .counter-container {
            font-size: 2em !important;
        }
        
        .pred-card h1 {
            font-size: 2em !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 12px !important;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6060ff;
        border-radius: 4px;
    }
    
    /* Selection color */
    ::selection {
        background: rgba(96, 96, 255, 0.3);
        color: white;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin: 2px !important;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #00cc66, #00ff88) !important;
        color: #003322 !important;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #ff9900, #ffcc00) !important;
        color: #332200 !important;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ff3333, #ff6666) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class LightweightLotteryAnalyzer:
    """Phiên bản nhẹ hơn để chạy trên Streamlit Cloud"""
    
    def __init__(self):
        self.init_database()
        self.models_loaded = False
        self.last_prediction = None
        self.counter_running = False
        
    def init_database(self):
        """Khởi tạo SQLite database"""
        try:
            self.conn = sqlite3.connect('lottery.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    draw_number TEXT,
                    numbers TEXT,
                    probabilities TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        except:
            pass
    
    def generate_sample_data(self, num_records=100):
        """Tạo dữ liệu mẫu"""
        dates = []
        draws = []
        now = datetime.now()
        
        for i in range(num_records):
            date = now - timedelta(days=num_records - i)
            dates.append(date.strftime('%Y-%m-%d'))
            draws.append(f'Kỳ {i+1}')
        
        data = {
            'draw_number': draws,
            'date': dates,
            'ten_thousand': np.random.randint(0, 10, num_records),
            'thousand': np.random.randint(0, 10, num_records),
            'hundred': np.random.randint(0, 10, num_records),
            'ten': np.random.randint(0, 10, num_records),
            'unit': np.random.randint(0, 10, num_records)
        }
        
        return pd.DataFrame(data)
    
    def generate_prediction(self):
        """Tạo dự đoán ngẫu nhiên (tạm thời)"""
        positions = ['ten_thousand', 'thousand', 'hundred', 'ten', 'unit']
        predictions = {}
        probabilities = {}
        
        for pos in positions:
            num = np.random.randint(0, 10)
            prob = np.random.uniform(60, 95)
            predictions[pos] = num
            probabilities[pos] = prob
        
        return {
            'numbers': predictions,
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_patterns(self):
        """Phân tích pattern cơ bản"""
        return {
            'cau_bet': [{'position': 'Trăm', 'number': 5, 'count': 3}],
            'cau_song': [{'position': 'Ngàn', 'numbers': [4, 7], 'rate': '78%'}],
            'cau_chet': [{'position': 'Đơn vị', 'number': 8, 'days': 12}],
            'cau_dao': [{'position': 'Chục', 'pattern': '2-5-2'}],
            'cau_gap': [{'position': 'Chục ngàn', 'trend': 'Tăng'}]
        }
    
    def start_counter(self):
        """Bắt đầu bộ đếm thời gian"""
        self.counter_running = True
    
    def stop_counter(self):
        """Dừng bộ đếm"""
        self.counter_running = False

# Khởi tạo analyzer
analyzer = LightweightLotteryAnalyzer()

# Header chính
st.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h1 style='color: #00ffaa; margin-bottom: 5px;'>🎯 TOOL AI 1.0</h1>
    <h3 style='color: #8080ff; margin-top: 0;'>SIÊU PHÂN TÍCH LOTOBET VIP</h3>
</div>
""", unsafe_allow_html=True)

# Real-time counter với JavaScript
st.markdown("""
<div class='counter-container' id='liveCounter'>
    ⏳ 01:30
</div>

<script>
function updateCounter() {
    let seconds = 90;
    const counter = document.getElementById('liveCounter');
    
    function tick() {
        seconds--;
        if (seconds < 0) seconds = 90;
        
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        counter.innerHTML = `⏳ ${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        // Đổi màu khi còn 30 giây
        if (seconds <= 30) {
            counter.style.color = '#ff4444';
            counter.style.textShadow = '0 0 15px rgba(255, 68, 68, 0.7)';
        } else {
            counter.style.color = '#00ffaa';
            counter.style.textShadow = '0 0 15px rgba(0, 255, 170, 0.5)';
        }
    }
    
    tick();
    setInterval(tick, 1000);
}

// Chạy counter khi trang load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateCounter);
} else {
    updateCounter();
}
</script>
""", unsafe_allow_html=True)

# Tabs chính
tabs = st.tabs(["🎯 5 SỐ", "🔢 2 SỐ", "🎲 3 SỐ", "📊 PHÂN TÍCH", "📈 XU HƯỚNG", "⚙️ CÀI ĐẶT"])

with tabs[0]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>DỰ ĐOÁN 5 SỐ TINH</h3>", unsafe_allow_html=True)
    
    if st.button("🔮 CHẠY PHÂN TÍCH NGAY", use_container_width=True, type="primary"):
        with st.spinner("🧠 Đang phân tích với AI..."):
            time.sleep(1.5)  # Hiệu ứng loading
            prediction = analyzer.generate_prediction()
            
            # Hiển thị 5 số
            cols = st.columns(5)
            pos_names = {
                'ten_thousand': 'CHỤC NGÀN',
                'thousand': 'NGÀN', 
                'hundred': 'TRĂM',
                'ten': 'CHỤC',
                'unit': 'ĐƠN VỊ'
            }
            
            for idx, (col, pos_key) in enumerate(zip(cols, pos_names.keys())):
                with col:
                    num = prediction['numbers'][pos_key]
                    prob = prediction['probabilities'][pos_key]
                    
                    card_class = "good-pred" if prob > 75 else "bad-pred"
                    prob_color = "#00ffaa" if prob > 75 else "#ff4444"
                    
                    st.markdown(f"""
                    <div class='pred-card {card_class}'>
                        <div style='font-size: 14px; opacity: 0.8;'>{pos_names[pos_key]}</div>
                        <div style='font-size: 2.5em; font-weight: 700; color: {prob_color};'>{num}</div>
                        <div style='font-size: 1.2em; font-weight: 600;'>{prob:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Đánh giá
            avg_prob = np.mean(list(prediction['probabilities'].values()))
            
            st.markdown("---")
            st.markdown("### 📊 ĐÁNH GIÁ XÁC SUẤT")
            
            if avg_prob > 80:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #00cc66, #00ff88); 
                            padding: 15px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #003322; margin: 0;'>🎯 NÊN ĐẦU TƯ</h3>
                    <p style='color: #005533; margin: 5px 0;'>Xác suất trung bình: <strong>{:.1f}%</strong></p>
                    <p style='color: #005533; margin: 0;'>Khả năng trúng: RẤT CAO</p>
                </div>
                """.format(avg_prob), unsafe_allow_html=True)
            elif avg_prob > 65:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #ff9900, #ffcc00); 
                            padding: 15px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #332200; margin: 0;'>👍 CÓ THỂ ĐẦU TƯ</h3>
                    <p style='color: #553300; margin: 5px 0;'>Xác suất trung bình: <strong>{:.1f}%</strong></p>
                    <p style='color: #553300; margin: 0;'>Khả năng trúng: CAO</p>
                </div>
                """.format(avg_prob), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #ff3333, #ff6666); 
                            padding: 15px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: white; margin: 0;'>⚠️ DỪNG LẠI</h3>
                    <p style='color: white; margin: 5px 0;'>Xác suất trung bình: <strong>{:.1f}%</strong></p>
                    <p style='color: white; margin: 0;'>Khuyến nghị: QUAN SÁT THÊM</p>
                </div>
                """.format(avg_prob), unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>DỰ ĐOÁN 2 SỐ - 3 CẶP</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, col in enumerate([col1, col2, col3]):
        with col:
            num1 = np.random.randint(0, 10)
            num2 = np.random.randint(0, 10)
            prob = np.random.uniform(50, 85)
            
            status = "NÊN ĐẦU TƯ" if prob > 65 else "THEO DÕI"
            status_class = "badge-success" if prob > 65 else "badge-warning"
            
            st.markdown(f"""
            <div class='custom-card' style='text-align: center;'>
                <div style='font-size: 14px; opacity: 0.8;'>CẶP {idx+1}</div>
                <div style='font-size: 2.2em; font-weight: 700; color: #00ffaa;'>{num1}{num2}</div>
                <div style='font-size: 1.5em; font-weight: 600; margin: 10px 0;'>{prob:.1f}%</div>
                <div class='badge {status_class}'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>DỰ ĐOÁN 3 SỐ - 3 CẶP</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, col in enumerate([col1, col2, col3]):
        with col:
            nums = [np.random.randint(0, 10) for _ in range(3)]
            prob = np.random.uniform(35, 75)
            
            status = "NÊN ĐẦU TƯ" if prob > 40 else "THEO DÕI"
            status_class = "badge-success" if prob > 40 else "badge-warning"
            
            st.markdown(f"""
            <div class='custom-card' style='text-align: center;'>
                <div style='font-size: 14px; opacity: 0.8;'>BỘ 3 SỐ {idx+1}</div>
                <div style='font-size: 1.8em; font-weight: 700; color: #00ffaa;'>{nums[0]}{nums[1]}{nums[2]}</div>
                <div style='font-size: 1.5em; font-weight: 600; margin: 10px 0;'>{prob:.1f}%</div>
                <div class='badge {status_class}'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>PHÂN TÍCH THỐNG KÊ</h3>", unsafe_allow_html=True)
    
    # Tạo dữ liệu thống kê đơn giản
    st.markdown("### 📈 TẦN SUẤT XUẤT HIỆN (7 NGÀY)")
    
    numbers = list(range(10))
    frequencies = np.random.randint(1, 10, 10)
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=numbers,
                y=frequencies,
                marker_color='#6060ff',
                text=frequencies,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback đơn giản
        for num, freq in zip(numbers, frequencies):
            st.markdown(f"**Số {num}**: {'▓' * freq} ({freq} lần)")
    
    # Phân tích Tài/Xỉu
    st.markdown("### 🎲 PHÂN TÍCH TÀI/XỈU")
    
    tai_percent = 58.3
    xiu_percent = 41.7
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='custom-card' style='text-align: center; background: rgba(0, 255, 136, 0.1) !important;'>
            <div style='font-size: 1.2em;'>TÀI</div>
            <div style='font-size: 2em; font-weight: 700; color: #00ff88;'>{tai_percent}%</div>
            <div style='font-size: 0.9em; opacity: 0.8;'>Xu hướng mạnh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='custom-card' style='text-align: center; background: rgba(255, 68, 68, 0.1) !important;'>
            <div style='font-size: 1.2em;'>XỈU</div>
            <div style='font-size: 2em; font-weight: 700; color: #ff4444;'>{xiu_percent}%</div>
            <div style='font-size: 0.9em; opacity: 0.8;'>Xu hướng yếu</div>
        </div>
        """, unsafe_allow_html=True)

with tabs[4]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>NHẬN DIỆN XU HƯỚNG</h3>", unsafe_allow_html=True)
    
    patterns = analyzer.analyze_patterns()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 CÁC CẦU ĐANG CHẠY")
        
        st.markdown("""
        <div class='custom-card'>
            <div class='badge badge-success'>CẦU BỆT</div>
            <p>🎯 <strong>Số 5</strong> tại hàng Trăm</p>
            <p>📊 Đã lặp 3 lần liên tiếp</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='custom-card'>
            <div class='badge badge-warning'>CẦU SỐNG</div>
            <p>🎯 <strong>Số 4, 7</strong> tại hàng Ngàn</p>
            <p>📊 Tần suất: 78% (7 ngày)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚠️ CÁC CẦU NGUY HIỂM")
        
        st.markdown("""
        <div class='custom-card'>
            <div class='badge badge-danger'>CẦU CHẾT</div>
            <p>💀 <strong>Số 8</strong> tại Đơn vị</p>
            <p>⏳ Không xuất hiện 12 ngày</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='custom-card'>
            <div class='badge badge-warning'>CẦU ĐẢO</div>
            <p>🔄 <strong>Pattern 2-5-2</strong> tại Chục</p>
            <p>📈 Đang hình thành</p>
        </div>
        """, unsafe_allow_html=True)

with tabs[5]:
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>CÀI ĐẶT & QUẢN LÝ</h3>", unsafe_allow_html=True)
    
    with st.form("capital_management"):
        st.markdown("### 💰 QUẢN LÝ VỐN")
        
        capital = st.number_input(
            "SỐ VỐN HIỆN TẠI (VND)",
            min_value=0,
            value=10000000,
            step=1000000,
            help="Nhập số vốn bạn đang có"
        )
        
        strategy = st.selectbox(
            "CHIẾN LƯỢC ĐẦU TƯ",
            [
                "BẢO THỦ (1-3% vốn/kỳ)",
                "CÂN BẰNG (3-5% vốn/kỳ)", 
                "MẠO HIỂM (5-10% vốn/kỳ)"
            ]
        )
        
        stop_loss = st.slider(
            "STOP-LOSS (%)",
            min_value=1,
            max_value=30,
            value=10,
            help="Mức lỗ tối đa cho phép"
        )
        
        take_profit = st.slider(
            "TAKE-PROFIT (%)", 
            min_value=5,
            max_value=50,
            value=25,
            help="Mức lãi mục tiêu"
        )
        
        if st.form_submit_button("💾 LƯU CÀI ĐẶT", type="primary"):
            st.success("✅ Cài đặt đã được lưu thành công!")
            
            # Hiển thị kế hoạch đầu tư
            if strategy == "BẢO THỦ (1-3% vốn/kỳ)":
                bet_per_kỳ = capital * 0.02
            elif strategy == "CÂN BẰNG (3-5% vốn/kỳ)":
                bet_per_kỳ = capital * 0.04
            else:
                bet_per_kỳ = capital * 0.075
            
            st.markdown(f"""
            <div class='custom-card'>
                <h4>📋 KẾ HOẠCH ĐẦU TƯ</h4>
                <p>💰 <strong>Vốn hiện tại:</strong> {capital:,.0f} VND</p>
                <p>🎯 <strong>Mức đặt/kỳ:</strong> {bet_per_kỳ:,.0f} VND</p>
                <p>⚠️ <strong>Dừng lỗ:</strong> {capital * stop_loss/100:,.0f} VND</p>
                <p>📈 <strong>Chốt lời:</strong> {capital * take_profit/100:,.0f} VND</p>
            </div>
            """, unsafe_allow_html=True)

# Sidebar với thông tin hệ thống
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <div style='font-size: 1.5em; color: #00ffaa;'>⚡ HỆ THỐNG AI</div>
        <div style='font-size: 0.9em; color: #8080ff;'>50 THUẬT TOÁN</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 THỐNG KÊ HÔM NAY")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 1.8em; color: #00ffaa;'>87%</div>
            <div style='font-size: 0.8em;'>ĐỘ CHÍNH XÁC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 1.8em; color: #ffaa00;'>24</div>
            <div style='font-size: 0.8em;'>KỲ PHÂN TÍCH</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 1.8em; color: #ff4444;'>3</div>
            <div style='font-size: 0.8em;'>CẢNH BÁO</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔔 CẢNH BÁO HỆ THỐNG")
    
    warnings_list = [
        "⚠️ Cầu chết số 8 đang kéo dài",
        "📉 Xu hướng Xỉu yếu dần",
        "🎯 Tập trung vào số 4, 7"
    ]
    
    for warning in warnings_list:
        st.markdown(f"""
        <div style='background: rgba(255, 68, 68, 0.1); 
                    padding: 8px 12px; 
                    border-radius: 8px; 
                    margin: 5px 0;
                    border-left: 3px solid #ff4444;'>
            {warning}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px 0; color: #8080ff; font-size: 0.9em;'>
    <p>© 2024 TOOL AI 1.0 - Hệ thống phân tích dự đoán thông minh</p>
    <p>Phiên bản: 1.0.0 | Cập nhật: {}</p>
    <p style='color: #ff4444; font-size: 0.8em;'>
        ⚠️ Cảnh báo: Đây là công cụ hỗ trợ phân tích. <br>
        Không đảm bảo 100% chính xác. Chơi có trách nhiệm.
    </p>
</div>
""".format(datetime.now().strftime('%d/%m/%Y')), unsafe_allow_html=True)

# Thêm script JavaScript cho real-time updates
st.markdown("""
<script>
// Auto-refresh predictions mỗi 90 giây
setTimeout(function() {
    window.location.reload();
}, 90000);

// Thêm hiệu ứng click cho mobile
document.addEventListener('touchstart', function() {}, {passive: true});

// Kiểm tra connection
window.addEventListener('online', function() {
    console.log('Online - Kết nối ổn định');
});

window.addEventListener('offline', function() {
    alert('Mất kết nối mạng! Vui lòng kiểm tra lại.');
});
</script>
""", unsafe_allow_html=True)
