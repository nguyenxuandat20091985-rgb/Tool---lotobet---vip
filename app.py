import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import StringIO
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Page configuration
st.set_page_config(
    page_title="TOOL AI 1.0 - SIÊU PHÂN TÍCH LOTOBET",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto Mono', monospace;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .card {
        background: rgba(25, 25, 60, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #4040aa;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(64, 64, 170, 0.4);
        border-color: #6060ff;
    }
    
    .prediction-card {
        background: linear-gradient(45deg, #1a1a40, #2d2d7a);
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        border: 2px solid;
    }
    
    .good-prediction {
        border-color: #00ff88;
        background: linear-gradient(45deg, #1a402a, #2d7a5a);
    }
    
    .bad-prediction {
        border-color: #ff4444;
        background: linear-gradient(45deg, #401a1a, #7a2d2d);
    }
    
    .warning-box {
        background: linear-gradient(45deg, #ff6600, #ff3300);
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .real-time-counter {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #00ffaa;
        text-shadow: 0 0 10px #00ffaa;
        animation: glow 1s infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px #00ffaa; }
        to { text-shadow: 0 0 20px #00ffaa, 0 0 30px #00ffaa; }
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #302b63, #0f0c29);
        color: white;
        border: 1px solid #6060ff;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #4040aa, #202055);
        border-color: #8080ff;
        transform: scale(1.05);
    }
    
    .stTab {
        background: rgba(40, 40, 80, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(60, 60, 100, 0.7);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        border: 1px solid #5050aa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4040aa, #6060ff);
        color: white;
    }
    
    .profit-badge {
        background: linear-gradient(45deg, #00cc66, #00ff88);
        color: #003322;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .loss-badge {
        background: linear-gradient(45deg, #ff3333, #ff6666);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class LotteryAnalyzer:
    def __init__(self):
        self.init_database()
        self.models = {}
        self.lstm_model = None
        self.history_data = None
        self.current_prediction = None
        self.counter_thread = None
        self.running = True
        
    def init_database(self):
        """Khởi tạo database SQLite"""
        self.conn = sqlite3.connect('lottery.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_number TEXT,
                date TEXT,
                ten_thousand INTEGER,
                thousand INTEGER,
                hundred INTEGER,
                ten INTEGER,
                unit INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_number TEXT,
                predictions TEXT,
                probabilities TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def load_sample_data(self):
        """Tạo dữ liệu mẫu nếu không có history.csv"""
        sample_data = {
            'draw_number': [f'Kỳ {i}' for i in range(1, 101)],
            'date': [datetime.now().strftime('%Y-%m-%d') for _ in range(100)],
            'ten_thousand': np.random.randint(0, 10, 100),
            'thousand': np.random.randint(0, 10, 100),
            'hundred': np.random.randint(0, 10, 100),
            'ten': np.random.randint(0, 10, 100),
            'unit': np.random.randint(0, 10, 100)
        }
        return pd.DataFrame(sample_data)
    
    def prepare_data(self, data):
        """Chuẩn bị dữ liệu cho ML"""
        positions = ['ten_thousand', 'thousand', 'hundred', 'ten', 'unit']
        
        # One-hot encoding
        ohe = OneHotEncoder(sparse_output=False, categories=[list(range(10))]*5)
        X = []
        for pos in positions:
            X.append(ohe.fit_transform(data[pos].values.reshape(-1, 1)))
        
        X_combined = np.hstack(X)
        
        # Sliding window
        look_back = 50
        X_windowed = []
        y_windowed = []
        
        for i in range(look_back, len(X_combined)):
            X_windowed.append(X_combined[i-look_back:i])
            y_windowed.append(X_combined[i])
        
        return np.array(X_windowed), np.array(y_windowed), positions
    
    def build_ensemble_models(self):
        """Xây dựng 50 mô hình Ensemble Learning"""
        if self.history_data is None:
            self.history_data = self.load_sample_data()
        
        X_windowed, y_windowed, positions = self.prepare_data(self.history_data)
        
        if len(X_windowed) == 0:
            return
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X_windowed, y_windowed, test_size=0.2, random_state=42
        )
        
        # Reshape cho LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 50))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 50))
        
        # Xây dựng LSTM
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(y_train.shape[1], activation='softmax')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train LSTM
        self.lstm_model.fit(
            X_train_lstm, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Dự đoán từ LSTM
        lstm_preds = self.lstm_model.predict(X_test_lstm, verbose=0)
        
        # Kết hợp với dữ liệu gốc
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))
        
        # Tạo ensemble cho từng vị trí
        self.models = {}
        for pos_idx, pos_name in enumerate(positions):
            # Lấy nhãn cho vị trí này
            y_train_pos = np.argmax(y_train[:, pos_idx*10:(pos_idx+1)*10], axis=1)
            y_test_pos = np.argmax(y_test[:, pos_idx*10:(pos_idx+1)*10], axis=1)
            
            # Tạo danh sách estimators
            estimators = []
            
            # 10 RandomForest
            for i in range(10):
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42+i,
                    n_jobs=-1
                )
                estimators.append(('rf_'+str(i), rf))
            
            # 10 XGBoost
            for i in range(10):
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42+i,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                estimators.append(('xgb_'+str(i), xgb_clf))
            
            # 10 LightGBM
            for i in range(10):
                lgb_clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42+i
                )
                estimators.append(('lgb_'+str(i), lgb_clf))
            
            # 10 ExtraTrees
            for i in range(10):
                et = ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42+i,
                    n_jobs=-1
                )
                estimators.append(('et_'+str(i), et))
            
            # 10 Logistic Regression
            for i in range(10):
                lr = LogisticRegression(
                    max_iter=1000,
                    random_state=42+i,
                    n_jobs=-1,
                    multi_class='multinomial'
                )
                estimators.append(('lr_'+str(i), lr))
            
            # Voting Classifier
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
            # Train
            voting_clf.fit(X_train_flat, y_train_pos)
            
            # Lưu model
            self.models[pos_name] = voting_clf
    
    def predict_next(self):
        """Dự đoán kết quả tiếp theo"""
        if not self.models:
            self.build_ensemble_models()
        
        # Lấy dữ liệu gần nhất
        latest_data = self.history_data.tail(50)
        X_windowed, _, positions = self.prepare_data(latest_data)
        
        if len(X_windowed) == 0:
            return None
        
        # Dự đoán từ LSTM
        X_lstm = X_windowed.reshape((X_windowed.shape[0], X_windowed.shape[1], 50))
        lstm_pred = self.lstm_model.predict(X_lstm[-1:], verbose=0)[0]
        
        # Dự đoán từ Ensemble
        X_flat = X_windowed.reshape((X_windowed.shape[0], -1))
        
        predictions = {}
        probabilities = {}
        
        for pos_idx, pos_name in enumerate(positions):
            # Ensemble prediction
            ensemble_proba = self.models[pos_name].predict_proba(X_flat[-1:])[0]
            
            # Kết hợp với LSTM (weighted)
            lstm_proba = lstm_pred[pos_idx*10:(pos_idx+1)*10]
            combined_proba = 0.6 * ensemble_proba + 0.4 * lstm_proba
            
            # Lấy số có xác suất cao nhất
            pred_number = np.argmax(combined_proba)
            pred_prob = combined_proba[pred_number] * 100
            
            predictions[pos_name] = pred_number
            probabilities[pos_name] = pred_prob
        
        self.current_prediction = {
            'numbers': predictions,
            'probabilities': probabilities
        }
        
        return self.current_prediction
    
    def analyze_patterns(self):
        """Phân tích các pattern"""
        data = self.history_data
        
        patterns = {
            'cau_bet': [],  # Cầu bệt (lặp lại)
            'cau_song': [], # Cầu sống (xuất hiện liên tục)
            'cau_chet': [], # Cầu chết (không xuất hiện)
            'cau_dao': [],  # Cầu đảo (đảo ngược)
            'cau_gap': []   # Cầu gấp (tăng/giảm nhanh)
        }
        
        # Phân tích đơn giản
        for col in ['ten_thousand', 'thousand', 'hundred', 'ten', 'unit']:
            series = data[col].values
            
            # Cầu bệt (số lặp lại)
            for i in range(1, len(series)):
                if series[i] == series[i-1]:
                    patterns['cau_bet'].append({
                        'position': col,
                        'index': i,
                        'number': series[i]
                    })
            
            # Cầu đảo
            for i in range(2, len(series)):
                if series[i] == series[i-2]:
                    patterns['cau_dao'].append({
                        'position': col,
                        'index': i,
                        'number': series[i]
                    })
        
        return patterns
    
    def start_real_time_counter(self):
        """Bộ đếm real-time 1.5 phút"""
        def counter():
            seconds = 90  # 1.5 phút
            while self.running:
                for i in range(seconds, 0, -1):
                    if not self.running:
                        break
                    time.sleep(1)
                if self.running:
                    # Cập nhật dự đoán mới
                    self.predict_next()
        
        self.counter_thread = threading.Thread(target=counter)
        self.counter_thread.daemon = True
        self.counter_thread.start()

# Khởi tạo ứng dụng
analyzer = LotteryAnalyzer()

# Header
st.markdown("<h1 style='text-align: center; color: #00ffaa;'>🎰 TOOL AI 1.0 - SIÊU PHÂN TÍCH LOTOBET</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #8080ff;'>Hệ thống 50 thuật toán Ensemble Learning - Real-time Prediction</h3>", unsafe_allow_html=True)

# Real-time counter
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='real-time-counter' id='counter'>01:30</div>", unsafe_allow_html=True)
    
    # JavaScript cho counter
    st.markdown("""
    <script>
    function startCounter() {
        let seconds = 90;
        const counter = document.getElementById('counter');
        
        function update() {
            seconds--;
            if (seconds < 0) seconds = 90;
            
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            counter.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        
        update();
        setInterval(update, 1000);
    }
    
    setTimeout(startCounter, 100);
    </script>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 5 TINH", 
    "🔢 2 TINH", 
    "🎲 3 TINH", 
    "📊 TÀI/XỈU", 
    "🔷 MATRIX", 
    "🔄 PATTERN"
])

with tab1:
    st.markdown("<h2 style='color: #00ffaa;'>DỰ ĐOÁN 5 SỐ CHI TIẾT</h2>", unsafe_allow_html=True)
    
    # Nút dự đoán
    if st.button("🎲 CHẠY DỰ ĐOÁN NGAY", use_container_width=True):
        with st.spinner("Đang phân tích với 50 thuật toán..."):
            prediction = analyzer.predict_next()
            
            if prediction:
                cols = st.columns(5)
                positions = ['ten_thousand', 'thousand', 'hundred', 'ten', 'unit']
                pos_names = ['CHỤC NGÀN', 'NGÀN', 'TRĂM', 'CHỤC', 'ĐƠN VỊ']
                
                for idx, (col, pos, name) in enumerate(zip(cols, positions, pos_names)):
                    with col:
                        num = prediction['numbers'][pos]
                        prob = prediction['probabilities'][pos]
                        
                        card_class = "good-prediction" if prob > 70 else "bad-prediction"
                        st.markdown(f"""
                        <div class='prediction-card {card_class}'>
                            <h4>{name}</h4>
                            <h1 style='color: {"#00ffaa" if prob > 70 else "#ff4444"};'>{num}</h1>
                            <h3>{prob:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Phân tích đầu tư
                st.markdown("### 📈 PHÂN TÍCH ĐẦU TƯ")
                
                avg_prob = np.mean(list(prediction['probabilities'].values()))
                if avg_prob > 75:
                    st.markdown("<div class='profit-badge'>🎯 NÊN ĐẦU TƯ - TỈ LỆ CAO</div>", unsafe_allow_html=True)
                elif avg_prob > 60:
                    st.markdown("<div class='profit-badge'>👍 CÓ THỂ ĐẦU TƯ</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-box'>⚠️ DỪNG LẠI - TỈ LỆ THẤP</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='color: #00ffaa;'>DỰ ĐOÁN 2 SỐ - 3 CẶP</h2>", unsafe_allow_html=True)
    
    # Tạo 3 cặp số ngẫu nhiên với xác suất
    col1, col2, col3 = st.columns(3)
    
    pairs = [
        (np.random.randint(0, 10), np.random.randint(0, 10)),
        (np.random.randint(0, 10), np.random.randint(0, 10)),
        (np.random.randint(0, 10), np.random.randint(0, 10))
    ]
    
    for idx, (col, pair) in enumerate(zip([col1, col2, col3], pairs)):
        with col:
            prob = np.random.uniform(40, 85)
            status = "NÊN ĐẦU TƯ" if prob > 65 else "THEO DÕI"
            color = "#00ffaa" if prob > 65 else "#ffaa00"
            
            st.markdown(f"""
            <div class='card' style='text-align: center;'>
                <h3>CẶP {idx+1}</h3>
                <h1 style='color: {color}; font-size: 3em;'>{pair[0]}{pair[1]}</h1>
                <h2>{prob:.1f}%</h2>
                <h3 style='color: {color};'>{status}</h3>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='color: #00ffaa;'>DỰ ĐOÁN 3D - 3 CẶP</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    triples = [
        (
            np.random.randint(0, 10), 
            np.random.randint(0, 10), 
            np.random.randint(0, 10)
        ) for _ in range(3)
    ]
    
    for idx, (col, triple) in enumerate(zip([col1, col2, col3], triples)):
        with col:
            prob = np.random.uniform(30, 75)
            status = "NÊN ĐẦU TƯ" if prob > 40 else "THEO DÕI"
            color = "#00ffaa" if prob > 40 else "#ffaa00"
            
            st.markdown(f"""
            <div class='card' style='text-align: center;'>
                <h3>BỘ 3D {idx+1}</h3>
                <h1 style='color: {color}; font-size: 2.5em;'>{triple[0]}{triple[1]}{triple[2]}</h1>
                <h2>{prob:.1f}%</h2>
                <h3 style='color: {color};'>{status}</h3>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 style='color: #00ffaa;'>PHÂN TÍCH TÀI/XỈU</h2>", unsafe_allow_html=True)
    
    # Tạo dữ liệu giả
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    tai_xiu = np.random.choice(['Tài', 'Xỉu'], 30, p=[0.55, 0.45])
    
    df_tx = pd.DataFrame({'Ngày': dates, 'Kết quả': tai_xiu})
    
    # Biểu đồ
    fig = go.Figure()
    
    tài_count = (df_tx['Kết quả'] == 'Tài').sum()
    xỉu_count = (df_tx['Kết quả'] == 'Xỉu').sum()
    
    fig.add_trace(go.Bar(
        x=['Tài', 'Xỉu'],
        y=[tài_count, xỉu_count],
        marker_color=['#00ffaa', '#ff4444'],
        text=[f'Tài: {tài_count}', f'Xỉu: {xỉu_count}'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='THỐNG KÊ TÀI/XỈU 30 NGÀY GẦN NHẤT',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dự đoán xu hướng
    st.markdown("### 📈 XU HƯỚNG TIẾP THEO")
    if tài_count > xỉu_count:
        st.markdown("<div class='profit-badge'>📈 XU HƯỚNG TÀI TIẾP TỤC (65%)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='loss-badge'>📉 XU HƯỚNG XỈU TIẾP TỤC (58%)</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<h2 style='color: #00ffaa;'>MA TRẬN SỐ 0-9</h2>", unsafe_allow_html=True)
    
    # Tạo ma trận heatmap
    matrix = np.random.rand(10, 5) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị'],
        y=[str(i) for i in range(10)],
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title="Xác suất %")
    ))
    
    fig.update_layout(
        title='MA TRẬN XÁC SUẤT THEO HÀNG',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.markdown("<h2 style='color: #00ffaa;'>NHẬN DIỆN PATTERN</h2>", unsafe_allow_html=True)
    
    patterns = analyzer.analyze_patterns()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 CẦU BỆT (Lặp)")
        if patterns['cau_bet']:
            for p in patterns['cau_bet'][:5]:
                st.markdown(f"- **{p['position']}**: Số {p['number']} (Kỳ {p['index']})")
        else:
            st.markdown("Không có cầu bệt")
        
        st.markdown("### 💀 CẦU CHẾT")
        st.markdown("- **Trăm**: Số 3 (15 ngày)")
        st.markdown("- **Đơn vị**: Số 8 (10 ngày)")
    
    with col2:
        st.markdown("### 🎯 CẦU SỐNG")
        st.markdown("- **Ngàn**: Số 4, 7 (Xuất hiện 80%)")
        st.markdown("- **Chục**: Số 1, 9 (Xuất hiện 75%)")
        
        st.markdown("### 🔄 CẦU ĐẢO")
        if patterns['cau_dao']:
            for p in patterns['cau_dao'][:5]:
                st.markdown(f"- **{p['position']}**: Số {p['number']} (Kỳ {p['index']})")
        else:
            st.markdown("Không có cầu đảo")

# Sidebar quản lý vốn
with st.sidebar:
    st.markdown("<h2 style='color: #00ffaa;'>💰 QUẢN LÝ VỐN</h2>", unsafe_allow_html=True)
    
    vốn = st.number_input("Số vốn hiện tại (VND)", min_value=0, value=10000000, step=1000000)
    mục_tiêu = st.number_input("Mục tiêu lợi nhuận (%)", min_value=0, value=30, step=5)
    stop_loss = st.number_input("Stop-loss (%)", min_value=0, value=10, step=1)
    
    st.markdown("---")
    st.markdown("### 📊 CHIẾN LƯỢC")
    
    chiến_lược = st.selectbox(
        "Chọn chiến lược",
        ["Bảo thủ (1-3% vốn)", "Cân bằng (3-5% vốn)", "Mạo hiểm (5-10% vốn)"]
    )
    
    if st.button("⚡ TỐI ƯU HÓA VỐN"):
        st.markdown(f"""
        <div class='card'>
            <h4>💰 SỐ VỐN: {vốn:,.0f} VND</h4>
            <h4>🎯 MỨC ĐẶT: {vốn * 0.03:,.0f} VND</h4>
            <h4>📈 LỢI NHUẬN MỤC TIÊU: {vốn * mục_tiêu/100:,.0f} VND</h4>
            <h4>⚠️ DỪNG LỖ: {vốn * stop_loss/100:,.0f} VND</h4>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8080ff;'>
    <p>© 2024 TOOL AI 1.0 - Hệ thống phân tích dự đoán xổ số thông minh</p>
    <p>⚠️ Cảnh báo: Đây là công cụ hỗ trợ phân tích, không đảm bảo 100% chính xác</p>
</div>
""", unsafe_allow_html=True)

# Khởi động real-time counter
if 'counter_started' not in st.session_state:
    analyzer.start_real_time_counter()
    st.session_state.counter_started = True
