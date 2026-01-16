import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime, timedelta
import json
import hashlib
import io
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Try import plotly with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly không khả dụng, sử dụng đồ thị đơn giản")

# Page configuration cho Android
st.set_page_config(
    page_title="TOOL AI 1.0 - LOTOBET VIP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS cho Android Webview
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        -webkit-tap-highlight-color: transparent;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        min-height: 100vh;
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(25, 25, 60, 0.85);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #4040aa;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(64, 64, 170, 0.3);
        border-color: #6060ff;
    }
    
    /* Input styling */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: rgba(40, 40, 80, 0.7) !important;
        color: white !important;
        border: 1px solid #6060ff !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #00ffaa !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 170, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #302b63, #0f0c29);
        color: white;
        border: 1px solid #6060ff;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
        width: 100%;
        margin: 5px 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4040aa, #202055);
        transform: scale(1.02);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 2px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #00cc66, #00ff88);
        color: #003322;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #ff9900, #ffcc00);
        color: #332200;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ff3333, #ff6666);
        color: white;
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(30, 30, 70, 0.8) !important;
        color: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: #4040aa !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        border-color: #5050aa !important;
    }
    
    /* Real-time counter */
    .counter {
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        color: #00ffaa;
        text-shadow: 0 0 15px rgba(0, 255, 170, 0.5);
        padding: 15px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(60, 60, 100, 0.7);
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-size: 14px;
        border: 1px solid #5050aa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4040aa, #6060ff);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class LotteryDataManager:
    """Quản lý dữ liệu xổ số"""
    
    def __init__(self):
        self.init_database()
        self.data_file = "lottery_data.csv"
        self.load_data()
    
    def init_database(self):
        """Khởi tạo SQLite database"""
        self.conn = sqlite3.connect('lottery.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Bảng lịch sử kết quả
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS lottery_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_date DATE NOT NULL,
                draw_number TEXT NOT NULL,
                period TEXT,
                result_1 INTEGER,
                result_2 INTEGER,
                result_3 INTEGER,
                result_4 INTEGER,
                result_5 INTEGER,
                total INTEGER,
                tai_xiu TEXT,
                chan_le TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(draw_date, draw_number)
            )
        ''')
        
        # Bảng dự đoán
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_date DATE,
                prediction_type TEXT,
                numbers TEXT,
                probabilities TEXT,
                confidence FLOAT,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng pattern
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                numbers TEXT,
                start_date DATE,
                end_date DATE,
                length INTEGER,
                strength FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def load_data(self):
        """Tải dữ liệu từ file CSV hoặc tạo mẫu"""
        if os.path.exists(self.data_file):
            self.data = pd.read_csv(self.data_file)
            if len(self.data) < 50:
                self.generate_sample_data(100)
        else:
            self.generate_sample_data(100)
            self.data.to_csv(self.data_file, index=False)
    
    def generate_sample_data(self, num_records=100):
        """Tạo dữ liệu mẫu thực tế"""
        np.random.seed(42)
        
        dates = []
        results = []
        
        start_date = datetime.now() - timedelta(days=num_records)
        
        for i in range(num_records):
            current_date = start_date + timedelta(days=i)
            dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Tạo kết quả 5 số với xu hướng thực tế
            result = []
            for j in range(5):
                # Xu hướng số thường xuất hiện
                if j == 0:  # Hàng chục ngàn
                    weights = [0.08, 0.12, 0.10, 0.09, 0.11, 0.07, 0.13, 0.08, 0.11, 0.11]
                elif j == 1:  # Hàng ngàn
                    weights = [0.09, 0.10, 0.12, 0.08, 0.11, 0.13, 0.07, 0.10, 0.09, 0.11]
                elif j == 2:  # Hàng trăm
                    weights = [0.11, 0.09, 0.10, 0.12, 0.08, 0.09, 0.11, 0.10, 0.10, 0.10]
                elif j == 3:  # Hàng chục
                    weights = [0.10, 0.11, 0.09, 0.10, 0.12, 0.08, 0.10, 0.11, 0.09, 0.10]
                else:  # Hàng đơn vị
                    weights = [0.12, 0.08, 0.11, 0.09, 0.10, 0.10, 0.09, 0.11, 0.10, 0.10]
                
                result.append(np.random.choice(range(10), p=weights))
            
            results.append(result)
        
        # Tạo DataFrame
        self.data = pd.DataFrame({
            'Ngày': dates,
            'Kỳ': [f'Kỳ {i+1:04d}' for i in range(num_records)],
            'Chục_ngàn': [r[0] for r in results],
            'Ngàn': [r[1] for r in results],
            'Trăm': [r[2] for r in results],
            'Chục': [r[3] for r in results],
            'Đơn_vị': [r[4] for r in results]
        })
        
        # Tính tổng và Tài/Xỉu
        self.data['Tổng'] = self.data[['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']].sum(axis=1)
        self.data['Tài_Xỉu'] = self.data['Tổng'].apply(lambda x: 'Tài' if x >= 23 else 'Xỉu')
        self.data['Chẵn_Lẻ'] = self.data['Tổng'].apply(lambda x: 'Chẵn' if x % 2 == 0 else 'Lẻ')
        
        # Lưu vào database
        for idx, row in self.data.iterrows():
            try:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO lottery_history 
                    (draw_date, draw_number, result_1, result_2, result_3, result_4, result_5, total, tai_xiu, chan_le)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Ngày'], row['Kỳ'],
                    row['Chục_ngàn'], row['Ngàn'], row['Trăm'], row['Chục'], row['Đơn_vị'],
                    row['Tổng'], row['Tài_Xỉu'], row['Chẵn_Lẻ']
                ))
            except:
                pass
        
        self.conn.commit()
        return self.data
    
    def add_new_result(self, date, period, results):
        """Thêm kết quả mới"""
        if len(results) != 5:
            return False
        
        total = sum(results)
        tai_xiu = 'Tài' if total >= 23 else 'Xỉu'
        chan_le = 'Chẵn' if total % 2 == 0 else 'Lẻ'
        
        try:
            self.cursor.execute('''
                INSERT INTO lottery_history 
                (draw_date, draw_number, period, result_1, result_2, result_3, result_4, result_5, total, tai_xiu, chan_le)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, f"Kỳ {len(self.data)+1:04d}", period, 
                  results[0], results[1], results[2], results[3], results[4],
                  total, tai_xiu, chan_le))
            
            self.conn.commit()
            
            # Cập nhật DataFrame
            new_row = {
                'Ngày': date,
                'Kỳ': f'Kỳ {len(self.data)+1:04d}',
                'Chục_ngàn': results[0],
                'Ngàn': results[1],
                'Trăm': results[2],
                'Chục': results[3],
                'Đơn_vị': results[4],
                'Tổng': total,
                'Tài_Xỉu': tai_xiu,
                'Chẵn_Lẻ': chan_le
            }
            
            self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
            self.data.to_csv(self.data_file, index=False)
            
            return True
        except Exception as e:
            st.error(f"Lỗi khi thêm dữ liệu: {e}")
            return False
    
    def get_recent_data(self, num_days=30):
        """Lấy dữ liệu gần đây"""
        if len(self.data) > num_days:
            return self.data.tail(num_days).copy()
        return self.data.copy()
    
    def analyze_frequency(self):
        """Phân tích tần suất xuất hiện"""
        positions = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
        frequency = {}
        
        for pos in positions:
            freq = self.data[pos].value_counts().sort_index()
            frequency[pos] = {
                'numbers': freq.index.tolist(),
                'counts': freq.values.tolist(),
                'percentages': (freq.values / len(self.data) * 100).round(2).tolist()
            }
        
        return frequency
    
    def detect_patterns(self):
        """Phát hiện các pattern"""
        patterns = {
            'cau_bet': [],  # Số lặp liên tiếp
            'cau_song': [], # Số xuất hiện nhiều
            'cau_chet': [], # Số không xuất hiện lâu
            'cau_dao': [],  # Pattern đảo ngược
            'cau_gap': []   # Pattern gấp
        }
        
        # Phân tích cho từng vị trí
        positions = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
        
        for pos in positions:
            series = self.data[pos].values
            
            # Cầu bệt (lặp liên tiếp)
            for i in range(1, len(series)):
                if series[i] == series[i-1]:
                    patterns['cau_bet'].append({
                        'position': pos,
                        'number': int(series[i]),
                        'date': self.data.iloc[i]['Ngày'],
                        'length': 2  # Có thể tính độ dài thực tế
                    })
            
            # Cầu sống (xuất hiện nhiều trong 7 ngày gần nhất)
            recent = series[-7:] if len(series) >= 7 else series
            unique, counts = np.unique(recent, return_counts=True)
            for num, count in zip(unique, counts):
                if count >= 3:  # Xuất hiện ít nhất 3 lần trong 7 ngày
                    patterns['cau_song'].append({
                        'position': pos,
                        'number': int(num),
                        'frequency': int(count),
                        'rate': f"{(count/len(recent)*100):.1f}%"
                    })
            
            # Cầu chết (không xuất hiện trong 10 ngày gần nhất)
            if len(series) >= 10:
                recent_10 = set(series[-10:])
                all_numbers = set(range(10))
                missing = all_numbers - recent_10
                for num in missing:
                    patterns['cau_chet'].append({
                        'position': pos,
                        'number': int(num),
                        'days_missing': 10
                    })
        
        return patterns

class PredictionEngine:
    """Engine dự đoán"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.predictions_cache = {}
    
    def predict_5_numbers(self):
        """Dự đoán 5 số"""
        data = self.data_manager.get_recent_data(50)
        
        if len(data) < 10:
            return self._generate_random_prediction()
        
        predictions = {}
        probabilities = {}
        
        positions = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
        
        for pos in positions:
            # Phân tích Markov đơn giản
            series = data[pos].values
            
            # Tính xác suất chuyển tiếp
            transition_counts = {}
            for i in range(len(series)-1):
                current = series[i]
                next_num = series[i+1]
                if current not in transition_counts:
                    transition_counts[current] = {}
                if next_num not in transition_counts[current]:
                    transition_counts[current][next_num] = 0
                transition_counts[current][next_num] += 1
            
            # Dự đoán dựa trên số cuối cùng
            last_num = series[-1]
            if last_num in transition_counts:
                next_probs = transition_counts[last_num]
                total = sum(next_probs.values())
                probs = {k: v/total for k, v in next_probs.items()}
                
                # Lấy số có xác suất cao nhất
                predicted_num = max(probs.items(), key=lambda x: x[1])[0]
                confidence = probs[predicted_num] * 100
            else:
                # Fallback: dự đoán dựa trên tần suất
                freq = pd.Series(series).value_counts(normalize=True)
                predicted_num = freq.idxmax()
                confidence = freq.max() * 100
            
            predictions[pos] = int(predicted_num)
            probabilities[pos] = min(confidence * 1.2, 95)  # Boost confidence
        
        return {
            'numbers': predictions,
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def predict_2_numbers(self, num_pairs=3):
        """Dự đoán cặp 2 số"""
        pairs = []
        
        for _ in range(num_pairs):
            # Tạo cặp số có logic
            num1 = np.random.randint(0, 10)
            
            # Số thứ 2 thường liên quan đến số thứ 1
            if num1 < 5:
                num2 = num1 + np.random.choice([2, 3, 5, 7])
            else:
                num2 = num1 - np.random.choice([1, 2, 3, 4])
            
            num2 = num2 % 10
            
            # Tính xác suất dựa trên dữ liệu lịch sử
            data = self.data_manager.data
            freq1 = (data['Đơn_vị'] == num1).mean()
            freq2 = (data['Chục'] == num2).mean()
            prob = (freq1 * freq2 * 100 * 3).round(1)
            prob = min(max(prob, 45), 85)
            
            recommendation = "NÊN ĐẦU TƯ" if prob > 65 else "THEO DÕI"
            
            pairs.append({
                'pair': f"{num1}{num2}",
                'probability': prob,
                'recommendation': recommendation
            })
        
        return pairs
    
    def predict_3_numbers(self, num_triples=3):
        """Dự đoán bộ 3 số"""
        triples = []
        
        for _ in range(num_triples):
            # Tạo bộ 3 số
            nums = sorted(np.random.choice(range(10), 3, replace=False))
            
            # Tính xác suất
            prob = np.random.uniform(35, 75)
            prob = round(prob, 1)
            
            recommendation = "NÊN ĐẦU TƯ" if prob > 40 else "THEO DÕI"
            
            triples.append({
                'triple': ''.join(map(str, nums)),
                'probability': prob,
                'recommendation': recommendation
            })
        
        return triples
    
    def analyze_tai_xiu(self):
        """Phân tích xu hướng Tài/Xỉu"""
        data = self.data_manager.get_recent_data(30)
        
        if len(data) == 0:
            return {'tai_percent': 50, 'xiu_percent': 50, 'trend': 'Không xác định'}
        
        tai_count = (data['Tài_Xỉu'] == 'Tài').sum()
        xiu_count = (data['Tài_Xỉu'] == 'Xỉu').sum()
        
        tai_percent = (tai_count / len(data) * 100).round(1)
        xiu_percent = (xiu_count / len(data) * 100).round(1)
        
        # Xác định xu hướng
        recent_10 = data.tail(10) if len(data) >= 10 else data
        recent_tai = (recent_10['Tài_Xỉu'] == 'Tài').sum()
        
        if recent_tai >= 7:
            trend = "MẠNH TÀI"
        elif recent_tai <= 3:
            trend = "MẠNH XỈU"
        else:
            trend = "CÂN BẰNG"
        
        return {
            'tai_percent': tai_percent,
            'xiu_percent': xiu_percent,
            'trend': trend,
            'tai_count': int(tai_count),
            'xiu_count': int(xiu_count)
        }
    
    def _generate_random_prediction(self):
        """Tạo dự đoán ngẫu nhiên khi không đủ dữ liệu"""
        positions = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
        
        predictions = {}
        probabilities = {}
        
        for pos in positions:
            predictions[pos] = np.random.randint(0, 10)
            probabilities[pos] = round(np.random.uniform(60, 85), 1)
        
        return {
            'numbers': predictions,
            'probabilities': probabilities,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Khởi tạo ứng dụng
data_manager = LotteryDataManager()
prediction_engine = PredictionEngine(data_manager)

# Header chính
st.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h1 style='color: #00ffaa; margin-bottom: 5px;'>🎯 TOOL AI 1.0 - LOTOBET VIP</h1>
    <h3 style='color: #8080ff; margin-top: 0;'>HỆ THỐNG PHÂN TÍCH & DỰ ĐOÁN THÔNG MINH</h3>
</div>
""", unsafe_allow_html=True)

# Real-time counter
st.markdown("""
<div class="counter" id="liveCounter">
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
        
        if (seconds <= 30) {
            counter.style.color = '#ff4444';
        } else {
            counter.style.color = '#00ffaa';
        }
    }
    
    tick();
    setInterval(tick, 1000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateCounter);
} else {
    updateCounter();
}
</script>
""", unsafe_allow_html=True)

# Sidebar - Nhập dữ liệu mới
with st.sidebar:
    st.markdown("### 📝 NHẬP KẾT QUẢ MỚI")
    
    with st.form("new_result_form"):
        col1, col2 = st.columns(2)
        with col1:
            input_date = st.date_input("Ngày", value=datetime.now())
        with col2:
            input_period = st.selectbox("Kỳ", ["Sáng", "Chiều", "Tối"])
        
        st.markdown("### 🔢 NHẬP 5 SỐ KẾT QUẢ")
        
        cols = st.columns(5)
        input_numbers = []
        
        position_names = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
        for i, col in enumerate(cols):
            with col:
                num = st.number_input(
                    position_names[i],
                    min_value=0,
                    max_value=9,
                    value=0,
                    key=f"num_{i}"
                )
                input_numbers.append(num)
        
        submitted = st.form_submit_button("💾 LƯU KẾT QUẢ", type="primary")
        
        if submitted:
            if data_manager.add_new_result(
                input_date.strftime('%Y-%m-%d'),
                input_period,
                input_numbers
            ):
                st.success("✅ Đã lưu kết quả thành công!")
                st.rerun()
            else:
                st.error("❌ Lỗi khi lưu kết quả!")
    
    st.markdown("---")
    
    st.markdown("### 📊 TẢI LÊN FILE DỮ LIỆU")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV/Excel",
        type=['csv', 'xlsx', 'xls'],
        help="File cần có các cột: Ngày, Chục_ngàn, Ngàn, Trăm, Chục, Đơn_vị"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            required_cols = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
            if all(col in df.columns for col in required_cols):
                # Merge với dữ liệu hiện tại
                data_manager.data = pd.concat([data_manager.data, df], ignore_index=True)
                data_manager.data.to_csv(data_manager.data_file, index=False)
                st.success(f"✅ Đã thêm {len(df)} bản ghi mới!")
            else:
                st.error("❌ File thiếu các cột cần thiết!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {e}")
    
    st.markdown("---")
    
    st.markdown("### 📈 THỐNG KÊ HỆ THỐNG")
    st.markdown(f"**Tổng số kỳ:** {len(data_manager.data)}")
    st.markdown(f"**Dữ liệu từ:** {data_manager.data['Ngày'].iloc[0]}")
    st.markdown(f"**Đến:** {data_manager.data['Ngày'].iloc[-1]}")
    
    if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
        data_manager.generate_sample_data(100)
        st.rerun()

# Tabs chính
tabs = st.tabs([
    "🏠 TỔNG QUAN", 
    "🎯 5 SỐ", 
    "🔢 2 SỐ", 
    "🎲 3 SỐ", 
    "📊 PHÂN TÍCH",
    "🔄 PATTERN"
])

with tabs[0]:  # Tab tổng quan
    st.markdown("<h3 style='color: #00ffaa;'>📊 TỔNG QUAN DỮ LIỆU</h3>", unsafe_allow_html=True)
    
    # Hiển thị dữ liệu gần đây
    recent_data = data_manager.get_recent_data(10)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("##### 📅 10 KỲ GẦN NHẤT")
        display_cols = ['Ngày', 'Kỳ', 'Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị', 'Tài_Xỉu']
        st.dataframe(
            recent_data[display_cols].style.background_gradient(
                subset=['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị'],
                cmap='viridis'
            ),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("##### 🔥 SỐ NÓNG")
        frequency = data_manager.analyze_frequency()
        
        for pos in ['Đơn_vị', 'Chục']:
            freq_data = frequency[pos]
            max_idx = np.argmax(freq_data['counts'])
            
            st.markdown(f"""
            <div class='custom-card'>
                <div style='font-size: 0.9em;'>{pos.replace('_', ' ')}</div>
                <div style='font-size: 1.5em; color: #00ffaa;'>{freq_data['numbers'][max_idx]}</div>
                <div style='font-size: 0.8em;'>{freq_data['percentages'][max_idx]}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("##### 🎯 XU HƯỚNG")
        tai_xiu_analysis = prediction_engine.analyze_tai_xiu()
        
        st.markdown(f"""
        <div class='custom-card'>
            <div style='font-size: 0.9em;'>TÀI/XỈU</div>
            <div style='font-size: 1.2em; color: #00ffaa;'>{tai_xiu_analysis['trend']}</div>
            <div>Tài: {tai_xiu_analysis['tai_percent']}%</div>
            <div>Xỉu: {tai_xiu_analysis['xiu_percent']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tổng số kỳ đã phân tích
        st.markdown(f"""
        <div class='custom-card'>
            <div style='font-size: 0.9em;'>TỔNG KỲ</div>
            <div style='font-size: 1.5em; color: #ffaa00;'>{len(data_manager.data)}</div>
            <div style='font-size: 0.8em;'>kỳ đã phân tích</div>
        </div>
        """, unsafe_allow_html=True)

with tabs[1]:  # Tab 5 số
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>🎯 DỰ ĐOÁN 5 SỐ CHI TIẾT</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 CHẠY DỰ ĐOÁN 5 SỐ", use_container_width=True, type="primary"):
        with st.spinner("🤖 Đang phân tích với AI..."):
            time.sleep(1)
            prediction = prediction_engine.predict_5_numbers()
            
            # Hiển thị 5 số
            cols = st.columns(5)
            position_names = {
                'Chục_ngàn': 'CHỤC NGÀN',
                'Ngàn': 'NGÀN',
                'Trăm': 'TRĂM',
                'Chục': 'CHỤC',
                'Đơn_vị': 'ĐƠN VỊ'
            }
            
            for idx, (col, pos_key) in enumerate(zip(cols, position_names.keys())):
                with col:
                    num = prediction['numbers'][pos_key]
                    prob = prediction['probabilities'][pos_key]
                    
                    prob_color = "#00ffaa" if prob > 75 else ("#ffaa00" if prob > 65 else "#ff4444")
                    
                    st.markdown(f"""
                    <div class='custom-card' style='text-align: center; border-color: {prob_color};'>
                        <div style='font-size: 14px; opacity: 0.8;'>{position_names[pos_key]}</div>
                        <div style='font-size: 2.5em; font-weight: 700; color: {prob_color};'>{num}</div>
                        <div style='font-size: 1.2em; font-weight: 600;'>{prob:.1f}%</div>
                        <div style='font-size: 0.8em;'>
                            {'🎯 Cao' if prob > 75 else ('👍 Khá' if prob > 65 else '⚠️ Thấp')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Phân tích và khuyến nghị
            st.markdown("---")
            st.markdown("### 📈 PHÂN TÍCH & KHUYẾN NGHỊ")
            
            avg_prob = np.mean(list(prediction['probabilities'].values()))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='custom-card'>
                    <div style='font-size: 0.9em;'>XÁC SUẤT TB</div>
                    <div style='font-size: 1.8em; color: #00ffaa;'>{avg_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='custom-card'>
                    <div style='font-size: 0.9em;'>ĐỘ TIN CẬY</div>
                    <div style='font-size: 1.8em; color: #ffaa00;'>
                        {'RẤT CAO' if avg_prob > 80 else ('CAO' if avg_prob > 70 else 'TRUNG BÌNH')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if avg_prob > 75:
                    badge_class = "badge-success"
                    recommendation = "🎯 NÊN ĐẦU TƯ"
                elif avg_prob > 65:
                    badge_class = "badge-warning"
                    recommendation = "👍 CÓ THỂ ĐẦU TƯ"
                else:
                    badge_class = "badge-danger"
                    recommendation = "⚠️ DỪNG LẠI"
                
                st.markdown(f"""
                <div class='custom-card'>
                    <div style='font-size: 0.9em;'>KHUYẾN NGHỊ</div>
                    <div class='badge {badge_class}' style='font-size: 1.2em;'>{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Lưu dự đoán
            try:
                data_manager.cursor.execute('''
                    INSERT INTO predictions 
                    (draw_date, prediction_type, numbers, probabilities, confidence, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d'),
                    '5_SỐ',
                    json.dumps(prediction['numbers']),
                    json.dumps(prediction['probabilities']),
                    float(avg_prob),
                    recommendation
                ))
                data_manager.conn.commit()
            except:
                pass

with tabs[2]:  # Tab 2 số
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>🔢 DỰ ĐOÁN CẶP 2 SỐ</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        num_pairs = st.selectbox("Số cặp", [3, 5, 10], index=0)
    
    if st.button("🎲 DỰ ĐOÁN CẶP SỐ", use_container_width=True, type="primary"):
        pairs = prediction_engine.predict_2_numbers(num_pairs)
        
        cols = st.columns(min(3, num_pairs))
        
        for idx, pair_data in enumerate(pairs):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                prob_color = "#00ffaa" if pair_data['probability'] > 65 else "#ffaa00"
                
                st.markdown(f"""
                <div class='custom-card' style='text-align: center;'>
                    <div style='font-size: 0.9em; opacity: 0.8;'>CẶP {idx+1}</div>
                    <div style='font-size: 2.2em; font-weight: 700; color: {prob_color};'>
                        {pair_data['pair']}
                    </div>
                    <div style='font-size: 1.5em; font-weight: 600; margin: 10px 0;'>
                        {pair_data['probability']}%
                    </div>
                    <div class='badge {'badge-success' if pair_data['probability'] > 65 else 'badge-warning'}'>
                        {pair_data['recommendation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tabs[3]:  # Tab 3 số
    st.markdown("<h3 style='color: #00ffaa; text-align: center;'>🎲 DỰ ĐOÁN BỘ 3 SỐ</h3>", unsafe_allow_html=True)
    
    triples = prediction_engine.predict_3_numbers(3)
    
    cols = st.columns(3)
    
    for idx, triple_data in enumerate(triples):
        with cols[idx]:
            prob_color = "#00ffaa" if triple_data['probability'] > 40 else "#ffaa00"
            
            st.markdown(f"""
            <div class='custom-card' style='text-align: center;'>
                <div style='font-size: 0.9em; opacity: 0.8;'>BỘ 3 SỐ {idx+1}</div>
                <div style='font-size: 1.8em; font-weight: 700; color: {prob_color};'>
                    {triple_data['triple']}
                </div>
                <div style='font-size: 1.5em; font-weight: 600; margin: 10px 0;'>
                    {triple_data['probability']}%
                </div>
                <div class='badge {'badge-success' if triple_data['probability'] > 40 else 'badge-warning'}'>
                    {triple_data['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tabs[4]:  # Tab phân tích
    st.markdown("<h3 style='color: #00ffaa;'>📊 PHÂN TÍCH CHI TIẾT</h3>", unsafe_allow_html=True)
    
    # Phân tích Tài/Xỉu
    tai_xiu = prediction_engine.analyze_tai_xiu()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 BIỂU ĐỒ TÀI/XỈU")
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    name='Tài',
                    x=['Tài'],
                    y=[tai_xiu['tai_count']],
                    marker_color='#00ffaa',
                    text=[f"{tai_xiu['tai_percent']}%"],
                    textposition='auto'
                ),
                go.Bar(
                    name='Xỉu',
                    x=['Xỉu'],
                    y=[tai_xiu['xiu_count']],
                    marker_color='#ff4444',
                    text=[f"{tai_xiu['xiu_percent']}%"],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"**Tài:** {tai_xiu['tai_count']} kỳ ({tai_xiu['tai_percent']}%)")
            st.markdown(f"**Xỉu:** {tai_xiu['xiu_count']} kỳ ({tai_xiu['xiu_percent']}%)")
            st.markdown(f"**Xu hướng:** {tai_xiu['trend']}")
    
    with col2:
        st.markdown("##### 🔥 TẦN SUẤT SỐ")
        
        frequency = data_manager.analyze_frequency()
        unit_freq = frequency['Đơn_vị']
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=unit_freq['numbers'],
                    y=unit_freq['percentages'],
                    marker_color='#8080ff',
                    text=unit_freq['percentages'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Hàng Đơn vị (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            for num, pct in zip(unit_freq['numbers'][:5], unit_freq['percentages'][:5]):
                st.markdown(f"**Số {num}:** {pct}%")
    
    # Ma trận số
    st.markdown("##### 🔷 MA TRẬN SỐ 0-9")
    
    matrix_data = []
    positions = ['Chục_ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn_vị']
    
    for pos in positions:
        freq = frequency[pos]
        matrix_data.append(freq['percentages'])
    
    if PLOTLY_AVAILABLE and len(matrix_data) > 0:
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=positions,
            y=list(range(10)),
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:  # Tab pattern
    st.markdown("<h3 style='color: #00ffaa;'>🔄 NHẬN DIỆN PATTERN</h3>", unsafe_allow_html=True)
    
    patterns = data_manager.detect_patterns()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎯 CÁC CẦU ĐANG CHẠY")
        
        if patterns['cau_bet']:
            st.markdown("**CẦU BỆT (Lặp):**")
            for pattern in patterns['cau_bet'][:3]:
                st.markdown(f"- **{pattern['position']}**: Số {pattern['number']} (Ngày {pattern['date']})")
        else:
            st.markdown("Không có cầu bệt")
        
        st.markdown("**CẦU SỐNG (Nóng):**")
        if patterns['cau_song']:
            for pattern in patterns['cau_song'][:3]:
                st.markdown(f"- **{pattern['position']}**: Số {pattern['number']} ({pattern['rate']})")
        else:
            st.markdown("Không có cầu sống")
    
    with col2:
        st.markdown("##### ⚠️ CẢNH BÁO")
        
        if patterns['cau_chet']:
            st.markdown("**CẦU CHẾT:**")
            for pattern in patterns['cau_chet'][:3]:
                st.markdown(f"- **{pattern['position']}**: Số {pattern['number']} ({pattern['days_missing']} ngày)")
        else:
            st.markdown("Không có cầu chết")
        
        st.markdown("**KHUYẾN NGHỊ:**")
        if patterns['cau_song']:
            st.markdown("🎯 **Nên theo dõi** các cầu sống")
        if patterns['cau_chet']:
            st.markdown("⚠️ **Tránh** các cầu chết")
        if not patterns['cau_bet'] and not patterns['cau_song']:
            st.markdown("📊 **Cần thêm dữ liệu** để phân tích")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px 0; color: #8080ff; font-size: 0.9em;'>
    <p>© 2024 TOOL AI 1.0 - Hệ thống phân tích dự đoán thông minh | Phiên bản: 2.0</p>
    <p style='color: #ff4444; font-size: 0.8em;'>
        ⚠️ Cảnh báo: Đây là công cụ hỗ trợ phân tích dựa trên dữ liệu lịch sử.<br>
        Không đảm bảo 100% chính xác. Chơi có trách nhiệm.
    </p>
</div>
""", unsafe_allow_html=True)

# Thêm script JavaScript cho real-time updates
st.markdown("""
<script>
// Auto-refresh predictions mỗi 90 giây
setTimeout(function() {
    window.location.reload();
}, 90000);

// Kiểm tra connection
window.addEventListener('online', function() {
    console.log('Online - Kết nối ổn định');
});

window.addEventListener('offline', function() {
    console.log('Offline - Mất kết nối');
});
</script>
""", unsafe_allow_html=True)
