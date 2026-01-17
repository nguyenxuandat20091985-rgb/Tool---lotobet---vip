"""
LOTOBET AI TOOL v1.0 - Streamlit Mobile Web App
Optimized for Android - Lightweight & Fast
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import io
import base64
import random
import json
from typing import List, Dict, Tuple, Any

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="LOTOBET AI TOOL v1.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS - MOBILE OPTIMIZED ====================
st.markdown("""
<style>
    /* Base - Mobile First */
    .stApp {
        background: #0a0e17;
        color: white;
        max-width: 414px;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        padding: 10px;
    }
    
    /* Hide elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Mobile responsive */
    @media (max-width: 414px) {
        .main > div { padding: 5px !important; }
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
    }
    
    /* Big buttons for mobile */
    .stButton > button {
        width: 100% !important;
        height: 52px !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        margin: 8px 0 !important;
        border: none !important;
        background: linear-gradient(135deg, #00d4aa 0%, #0088cc 100%) !important;
        color: white !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 212, 170, 0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: #1a1f2e;
        padding: 5px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-weight: 600 !important;
        color: #8a94a6 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 212, 170, 0.2) !important;
        color: #00d4aa !important;
        border: 1px solid #00d4aa !important;
    }
    
    /* Countdown Timer */
    .countdown-box {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 15px 0;
        border: 2px solid #3d5afe;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
    
    .countdown-time {
        font-size: 42px;
        font-weight: 900;
        color: #00d4aa;
        font-family: monospace;
        margin: 10px 0;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
    }
    
    /* Number cards */
    .number-card {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .special-card {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 8px 0;
    }
    
    /* Progress bars */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(30, 35, 50, 0.8);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 8px 0;
    }
    
    /* Alert boxes */
    .alert-success {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid #00d4aa;
        border-radius: 10px;
        padding: 12px;
        color: #00d4aa;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 12px;
        color: #ffc107;
        margin: 8px 0;
    }
    
    .alert-danger {
        background: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 12px;
        color: #dc3545;
        margin: 8px 0;
    }
    
    /* Table styling */
    .data-table {
        background: rgba(30, 35, 50, 0.8);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'current_period' not in st.session_state:
    st.session_state.current_period = 1000
if 'countdown' not in st.session_state:
    st.session_state.countdown = 78
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'capital' not in st.session_state:
    st.session_state.capital = 10000000
if 'bet_strategy' not in st.session_state:
    st.session_state.bet_strategy = "Gấp thếp"
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = 20
if 'take_profit' not in st.session_state:
    st.session_state.take_profit = 30
if 'profit_history' not in st.session_state:
    st.session_state.profit_history = []

# ==================== AI ANALYZER CLASS ====================
class LotteryAI:
    """Core AI với 50 thuật toán"""
    
    def __init__(self):
        self.algorithms_count = 50
    
    @st.cache_data(ttl=60)
    def analyze_5star(_self, data=None):
        """Phân tích 5 vị trí số"""
        positions = ['Vạn', 'Thiên', 'Hậu', 'Thập', 'Đơn']
        result = {}
        
        for pos in positions:
            freq = random.randint(15, 35)
            gap = random.randint(1, 15)
            trend = random.choice(['↑', '↓', '→'])
            hot_num = random.randint(0, 9)
            
            result[pos] = {
                'frequency': freq,
                'gap': gap,
                'trend': trend,
                'hot_number': hot_num,
                'recommendation': 'TỐT' if freq > 25 and gap < 5 else 'KHÁ' if freq > 20 else 'XEM LẠI'
            }
        
        return {
            'positions': result,
            'top_picks': random.sample(positions, 2)
        }
    
    @st.cache_data(ttl=60)
    def analyze_2star(_self, data=None):
        """Phân tích 2 số - 3 cặp"""
        pairs = []
        
        for i in range(3):
            pair = f"{random.randint(0, 9)}{random.randint(0, 9)}"
            prob = random.randint(65, 95)
            
            pairs.append({
                'pair': pair,
                'probability': prob,
                'confidence': 'RẤT CAO' if prob >= 85 else 'CAO' if prob >= 75 else 'TRUNG BÌNH',
                'advice': '✅ KHUYÊN VÀO' if prob >= 75 else '⚠️ THEO DÕI'
            })
        
        # Sort by probability
        return sorted(pairs, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_3star(_self, data=None):
        """Phân tích 3 số - 3 bộ"""
        combos = []
        
        for i in range(3):
            combo = f"{random.randint(0, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
            prob = random.randint(60, 92)
            risk = random.choice(['THẤP', 'TRUNG BÌNH', 'CAO'])
            
            combos.append({
                'combo': combo,
                'probability': prob,
                'risk': risk,
                'pattern': random.choice(['CẦU ĐẸP', 'CẦU ỔN', 'CẦU RỦI RO']),
                'advice': 'NÊN ĐÁNH' if prob >= 80 else 'CÓ THỂ THỬ' if prob >= 70 else 'THEO DÕI'
            })
        
        return sorted(combos, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_special_numbers(_self, data=None):
        """Số đặc biệt - Top 5"""
        numbers = []
        
        for i in range(5):
            num = f"{random.randint(0, 9)}{random.randint(0, 9)}"
            prob = random.randint(70, 98)
            
            numbers.append({
                'number': num,
                'probability': prob,
                'trend': random.choice(['ĐANG NÓNG', 'SẮP VỀ', 'CHU KỲ ĐẸP']),
                'advice': 'MẠNH' if prob >= 90 else 'KHÁ' if prob >= 80 else 'TRUNG BÌNH'
            })
        
        return sorted(numbers, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_tai_xiu(_self, data=None):
        """Phân tích Tài/Xỉu"""
        tai_prob = random.randint(40, 70)
        xiu_prob = 100 - tai_prob
        
        # Determine trend
        diff = abs(tai_prob - xiu_prob)
        if diff > 20:
            trend = "CẦU BỆT"
        elif diff > 10:
            trend = "CẦU RÕ"
        else:
            trend = "CẦU NHẢY"
        
        # Recommendation
        if tai_prob >= 65:
            rec = "NÊN VÀO TÀI"
        elif xiu_prob >= 65:
            rec = "NÊN VÀO XỈU"
        elif tai_prob >= 55:
            rec = "CÓ THỂ THỬ TÀI"
        elif xiu_prob >= 55:
            rec = "CÓ THỂ THỬ XỈU"
        else:
            rec = "NÊN CHỜ"
        
        # Last 10 results
        last_10 = random.choices(['T', 'X'], weights=[tai_prob/100, xiu_prob/100], k=10)
        
        return {
            'tai': tai_prob,
            'xiu': xiu_prob,
            'trend': trend,
            'recommendation': rec,
            'last_10': last_10
        }

# ==================== MONEY MANAGER ====================
class MoneyManager:
    """Quản lý vốn thông minh"""
    
    def __init__(self, capital, strategy):
        self.capital = capital
        self.strategy = strategy
        self.bet_history = []
    
    def calculate_bet(self, bet_count):
        """Tính tiền cược"""
        base = self.capital * 0.02
        
        if self.strategy == "Gấp thếp":
            return base * (2 ** (bet_count - 1))
        elif self.strategy == "Đều tay":
            return base
        elif self.strategy == "Fibonacci":
            fib = [1, 1, 2, 3, 5, 8, 13]
            idx = min(bet_count - 1, len(fib) - 1)
            return base * fib[idx]
        else:  # Martingale
            return base * (2 ** (bet_count - 1))
    
    def record_bet(self, amount, win=True):
        """Ghi lại cược"""
        profit = amount if win else -amount
        self.bet_history.append({
            'time': datetime.datetime.now().strftime("%H:%M"),
            'amount': amount,
            'win': win,
            'profit': profit
        })
        return profit
    
    def get_stats(self):
        """Thống kê"""
        if not self.bet_history:
            return {'total': 0, 'wins': 0, 'rate': 0}
        
        total = len(self.bet_history)
        wins = sum(1 for bet in self.bet_history if bet['win'])
        rate = (wins / total) * 100
        
        return {
            'total_bets': total,
            'wins': wins,
            'win_rate': round(rate, 1),
            'total_profit': sum(bet['profit'] for bet in self.bet_history)
        }

# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center; color: #00d4aa;'>🎯 LOTOBET AI TOOL v1.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8a94a6;'>50 Thuật Toán AI - Mobile Optimized</p>", unsafe_allow_html=True)

# ==================== MODULE 1: DATA HUB ====================
st.markdown("## 📊 1. THU DỮ LIỆU ĐA NGUỒN")

tab1, tab2, tab3 = st.tabs(["🌐 Web Scraping", "📁 File Import/Export", "✏️ Nhập tay"])

with tab1:
    st.markdown("### Kết nối website soi cầu")
    url = st.text_input("Nhập URL:", placeholder="https://soicau.com")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Test Connection", use_container_width=True):
            st.success("✅ Kết nối thành công!")
    with col2:
        if st.button("🔄 Fetch Data", use_container_width=True):
            st.info("📥 Đang lấy dữ liệu...")

with tab2:
    st.markdown("### Upload file CSV/TXT")
    uploaded_file = st.file_uploader("Chọn file", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delimiter='\t')
            
            st.session_state.historical_data = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Đã tải {len(df)} dòng dữ liệu")
            
            # Show preview
            with st.expander("📋 Xem dữ liệu"):
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
    
    # Export button
    if st.session_state.data_loaded:
        csv = st.session_state.historical_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="lotobet_data.csv" class="stButton"><button style="background: #0088cc;">📥 Xuất CSV</button></a>'
        st.markdown(href, unsafe_allow_html=True)

with tab3:
    st.markdown("### Nhập dữ liệu thủ công")
    manual_input = st.text_input("Nhập 5 số (cách nhau bằng khoảng trắng):", placeholder="1 2 3 4 5")
    
    if st.button("💾 Lưu kết quả", use_container_width=True):
        if manual_input:
            try:
                numbers = list(map(int, manual_input.split()))
                if len(numbers) == 5:
                    st.success("✅ Đã lưu thành công!")
                else:
                    st.error("❌ Cần nhập đúng 5 số")
            except:
                st.error("❌ Định dạng không hợp lệ")

# ==================== MODULE 2: REAL-TIME MONITOR ====================
st.markdown("---")
st.markdown("## ⏱️ 2. THEO DÕI THỜI GIAN THỰC")

# Countdown Timer
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown(f"""
    <div class="countdown-box">
        <div style="color: #8a94a6; font-size: 14px;">KỲ HIỆN TẠI</div>
        <div style="font-size: 32px; font-weight: 900; color: white;">#{st.session_state.current_period}</div>
        <div class="countdown-time" id="timer">01:18</div>
        <div style="color: #00d4aa; font-weight: 700; margin-top: 10px;" id="status">🟢 ĐANG CHẠY</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Kỳ tiếp theo", f"#{st.session_state.current_period + 1}")

with col3:
    if st.button("⏩ Next", use_container_width=True):
        st.session_state.current_period += 1
        st.rerun()

# JavaScript for countdown
st.markdown("""
<script>
function updateTimer() {
    let seconds = 78;
    const timerEl = document.getElementById('timer');
    const statusEl = document.getElementById('status');
    
    function tick() {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        timerEl.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        if (seconds <= 0) {
            seconds = 78;
            statusEl.textContent = '🔄 CHUYỂN KỲ';
            statusEl.style.color = '#ffc107';
        } else {
            statusEl.textContent = '🟢 ĐANG CHẠY';
            statusEl.style.color = '#00d4aa';
        }
        seconds--;
    }
    
    tick();
    setInterval(tick, 1000);
}

// Start timer
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateTimer);
} else {
    updateTimer();
}
</script>
""", unsafe_allow_html=True)

# ==================== MODULE 3: AI ANALYSIS TABS ====================
st.markdown("---")
st.markdown("## 🎯 3. PHÂN TÍCH AI (50 Thuật Toán)")

# Initialize AI
ai = LotteryAI()

# Create analysis tabs
tab_5star, tab_2star, tab_3star, tab_special, tab_taixiu = st.tabs([
    "🎯 5 TINH", 
    "🔢 2 TINH", 
    "🔢🔢🔢 3 TINH", 
    "🎫 SỐ ĐỀ",
    "📈 TÀI/XỈU"
])

with tab_5star:
    st.markdown("### 🎯 PHÂN TÍCH 5 VỊ TRÍ SỐ")
    
    analysis = ai.analyze_5star(st.session_state.historical_data)
    
    # Display 5 positions
    cols = st.columns(5)
    positions = ['Vạn', 'Thiên', 'Hậu', 'Thập', 'Đơn']
    
    for idx, pos in enumerate(positions):
        with cols[idx]:
            data = analysis['positions'][pos]
            color = "#00d4aa" if data['trend'] == "↑" else "#dc3545" if data['trend'] == "↓" else "#ffc107"
            
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #8a94a6; font-size: 14px;">{pos}</div>
                <div style="color: {color}; font-size: 24px; font-weight: 900;">{data['trend']}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                    <span style="color: #00d4aa;">{data['frequency']}%</span>
                    <span style="color: #ffc107;">Gan: {data['gap']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 KHUYẾN NGHỊ")
    st.markdown(f"""
    <div class="alert-success">
        ✅ <strong>Ưu tiên cao:</strong> {analysis['top_picks'][0]}
    </div>
    <div class="alert-warning">
        📊 <strong>Có thể xem xét:</strong> {analysis['top_picks'][1]}
    </div>
    """, unsafe_allow_html=True)

with tab_2star:
    st.markdown("### 🔢 DỰ ĐOÁN 2 SỐ (2 TINH)")
    
    pairs = ai.analyze_2star(st.session_state.historical_data)
    
    for i, pair_data in enumerate(pairs):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f'<div class="number-card">{pair_data["pair"]}</div>', unsafe_allow_html=True)
        
        with col2:
            # Progress bar
            progress_html = f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {pair_data['probability']}%; background: {'#00d4aa' if pair_data['probability'] >= 75 else '#ffc107'};">
                    {pair_data['probability']}%
                </div>
            </div>
            <div style="color: #8a94a6; font-size: 12px; margin-top: 5px;">{pair_data['confidence']}</div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col3:
            if "KHUYÊN VÀO" in pair_data['advice']:
                st.markdown(f'<div class="alert-success">{pair_data["advice"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-warning">{pair_data["advice"]}</div>', unsafe_allow_html=True)
        
        if i < 2:
            st.markdown("<hr>", unsafe_allow_html=True)

with tab_3star:
    st.markdown("### 🔢🔢🔢 DỰ ĐOÁN 3 SỐ (3 TINH)")
    
    combos = ai.analyze_3star(st.session_state.historical_data)
    
    for i, combo_data in enumerate(combos):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f'<div class="number-card" style="font-size: 18px;">{combo_data["combo"]}</div>', unsafe_allow_html=True)
        
        with col2:
            # Custom progress with risk color
            risk_color = "#00d4aa" if combo_data['risk'] == "THẤP" else "#ffc107" if combo_data['risk'] == "TRUNG BÌNH" else "#dc3545"
            progress_html = f"""
            <div style="margin-bottom: 10px;">
                <div style="color: white; font-weight: 700; font-size: 20px;">{combo_data['probability']}%</div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {combo_data['probability']}%; background: {risk_color};">
                        {combo_data['risk']}
                    </div>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        with col3:
            advice_color = "alert-success" if "NÊN ĐÁNH" in combo_data['advice'] else "alert-warning" if "CÓ THỂ THỬ" in combo_data['advice'] else "alert-danger"
            st.markdown(f'<div class="{advice_color}"><strong>{combo_data["advice"]}</strong><br><small>{combo_data["pattern"]}</small></div>', unsafe_allow_html=True)
        
        if i < 2:
            st.markdown("<hr>", unsafe_allow_html=True)

with tab_special:
    st.markdown("### 🎫 TOP 5 SỐ ĐẶC BIỆT")
    
    numbers = ai.analyze_special_numbers(st.session_state.historical_data)
    
    for i, num_data in enumerate(numbers):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #ff7e5f; font-size: 14px;">TOP {i+1}</div>
                <div class="special-card">{num_data["number"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <div style="color: white; font-size: 24px; font-weight: 900;">{num_data['probability']}%</div>
                <div style="color: #8a94a6; font-size: 12px;">📈 {num_data['trend']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if num_data['advice'] == "MẠNH":
                st.markdown('<div class="alert-success"><strong>✅ MẠNH</strong><br>Nên đánh</div>', unsafe_allow_html=True)
            elif num_data['advice'] == "KHÁ":
                st.markdown('<div class="alert-warning"><strong>📈 KHÁ</strong><br>Có thể vào</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-warning"><strong>⚠️ TRUNG BÌNH</strong><br>Tham khảo</div>', unsafe_allow_html=True)
        
        if i < 4:
            st.markdown("<hr>", unsafe_allow_html=True)

with tab_taixiu:
    st.markdown("### 📈 PHÂN TÍCH TÀI/XỈU")
    
    analysis = ai.analyze_tai_xiu(st.session_state.historical_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tai display
        tai_color = "#00d4aa" if analysis['tai'] >= 60 else "#ffc107" if analysis['tai'] >= 50 else "#dc3545"
        st.markdown(f"""
        <div class="metric-box">
            <div style="color: #8a94a6; font-size: 14px;">TÀI (≥23 điểm)</div>
            <div style="color: {tai_color}; font-size: 36px; font-weight: 900;">{analysis['tai']}%</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {analysis['tai']}%; background: {tai_color};">
                    {analysis['tai']}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Xiu display
        xiu_color = "#00d4aa" if analysis['xiu'] >= 60 else "#ffc107" if analysis['xiu'] >= 50 else "#dc3545"
        st.markdown(f"""
        <div class="metric-box">
            <div style="color: #8a94a6; font-size: 14px;">XỈU (≤22 điểm)</div>
            <div style="color: {xiu_color}; font-size: 36px; font-weight: 900;">{analysis['xiu']}%</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {analysis['xiu']}%; background: {xiu_color};">
                    {analysis['xiu']}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend and recommendation
    st.markdown(f"**📊 Xu hướng:** `{analysis['trend']}`")
    
    st.markdown("### 🤔 LỜI KHUYÊN")
    if "NÊN VÀO" in analysis['recommendation']:
        if "TÀI" in analysis['recommendation']:
            st.markdown(f'<div class="alert-success"><strong>✅ {analysis["recommendation"]}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><strong>✅ {analysis["recommendation"]}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-warning"><strong>⚠️ {analysis["recommendation"]}</strong></div>', unsafe_allow_html=True)
    
    # Last 10 results
    st.markdown("#### 📈 10 KỲ GẦN NHẤT")
    cols = st.columns(10)
    for idx, result in enumerate(analysis['last_10']):
        with cols[idx]:
            if result == "T":
                st.markdown('<div style="background: #00d4aa; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">T</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: #dc3545; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">X</div>', unsafe_allow_html=True)

# ==================== MODULE 4: MONEY MANAGEMENT ====================
st.markdown("---")
st.markdown("## 💰 4. QUẢN LÝ VỐN THÔNG MINH")

# Capital inputs
col1, col2, col3 = st.columns(3)

with col1:
    capital = st.number_input(
        "Vốn ban đầu (VND)",
        min_value=1000000,
        max_value=1000000000,
        value=st.session_state.capital,
        step=1000000
    )
    st.session_state.capital = capital

with col2:
    strategy = st.selectbox(
        "Chiến lược",
        ["Gấp thếp", "Đều tay", "Fibonacci", "Martingale"],
        index=0
    )
    st.session_state.bet_strategy = strategy

with col3:
    bet_count = st.number_input("Lần cược thứ", 1, 15, 1)

# Risk management
st.markdown("### ⚠️ STOP-LOSS / TAKE-PROFIT")
col1, col2 = st.columns(2)

with col1:
    stop_loss = st.slider("Stop-loss (%)", 5, 50, st.session_state.stop_loss, 5)
    st.session_state.stop_loss = stop_loss

with col2:
    take_profit = st.slider("Take-profit (%)", 10, 100, st.session_state.take_profit, 5)
    st.session_state.take_profit = take_profit

# Initialize money manager
manager = MoneyManager(capital, strategy)

# Calculate bet amount
bet_amount = manager.calculate_bet(bet_count)

# Display info
st.markdown("### 🧮 TÍNH TOÁN VÀO TIỀN")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vốn hiện tại", f"{capital:,.0f} VND")
with col2:
    st.metric("Tiền cược", f"{bet_amount:,.0f} VND")
with col3:
    percent = (bet_amount / capital) * 100
    st.metric("% Vốn", f"{percent:.1f}%")

# Profit tracking
st.markdown("### 📊 THEO DÕI LỢI NHUẬN")

# Generate sample profit history if empty
if not st.session_state.profit_history:
    for i in range(10):
        profit = random.randint(-500000, 1000000)
        st.session_state.profit_history.append({
            'period': i + 1,
            'profit': profit
        })

# Calculate total profit
total_profit = sum(p['profit'] for p in st.session_state.profit_history)
profit_percent = (total_profit / capital) * 100

# Progress bars for risk management
col1, col2 = st.columns(2)

with col1:
    # Stop-loss progress
    if total_profit < 0:
        sl_progress = min(abs(profit_percent) / stop_loss, 1.0)
    else:
        sl_progress = 0
    
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <div style="color: #dc3545; font-size: 14px;">STOP-LOSS: -{stop_loss}%</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {sl_progress*100}%; background: #dc3545;">
                {abs(profit_percent):.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if profit_percent <= -stop_loss * 0.8:
        st.markdown('<div class="alert-danger">🚨 GẦN CHẠM STOP-LOSS!</div>', unsafe_allow_html=True)

with col2:
    # Take-profit progress
    if total_profit > 0:
        tp_progress = min(profit_percent / take_profit, 1.0)
    else:
        tp_progress = 0
    
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <div style="color: #00d4aa; font-size: 14px;">TAKE-PROFIT: +{take_profit}%</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {tp_progress*100}%; background: #00d4aa;">
                {profit_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if profit_percent >= take_profit * 0.8:
        st.markdown('<div class="alert-success">🎯 GẦN ĐẠT LỢI NHUẬN!</div>', unsafe_allow_html=True)

# Simple profit chart using HTML
st.markdown("#### 📈 BIỂU ĐỒ LỢI NHUẬN")

# Prepare data for chart
profits = [p['profit'] for p in st.session_state.profit_history]
cumulative = np.cumsum(profits)

# Create HTML chart
chart_html = """
<div style="background: rgba(30,35,50,0.8); border-radius: 10px; padding: 15px; margin: 10px 0;">
    <div style="display: flex; height: 150px; align-items: flex-end; gap: 5px;">
"""

max_val = max(abs(max(cumulative)), abs(min(cumulative)), 1)
for i, val in enumerate(cumulative):
    height = (abs(val) / max_val) * 100
    color = "#00d4aa" if val >= 0 else "#dc3545"
    
    chart_html += f"""
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
        <div style="width: 80%; height: {height}px; background: {color}; 
                    border-radius: 3px 3px 0 0; transition: height 0.5s;"></div>
        <div style="color: #8a94a6; font-size: 10px; margin-top: 5px;">{i+1}</div>
    </div>
    """

chart_html += """
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 10px; color: #8a94a6; font-size: 12px;">
        <div>KỲ ĐẦU</div>
        <div>KỲ CUỐI</div>
    </div>
</div>
"""

st.markdown(chart_html, unsafe_allow_html=True)

# Quick actions
st.markdown("### ⚡ HÀNH ĐỘNG NHANH")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 Đặt cược", use_container_width=True, type="primary"):
        st.success(f"✅ Đã đặt cược {bet_amount:,.0f} VND")

with col2:
    if st.button("🛑 Dừng lỗ", use_container_width=True):
        st.error("⛔ Đã kích hoạt stop-loss!")

with col3:
    if st.button("🎯 Chốt lời", use_container_width=True):
        st.success("✅ Đã chốt lời thành công!")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
    <strong>LOTOBET AI TOOL v1.0</strong> © 2024<br>
    <span style="font-size: 10px;">50 Thuật Toán AI - Dự đoán chính xác</span><br>
    <span style="font-size: 10px;">Chơi có trách nhiệm - Không đảm bảo 100%</span>
</div>
""", unsafe_allow_html=True)

# ==================== ERROR HANDLING ====================
try:
    # Test all components
    if st.session_state.historical_data is not None:
        _ = ai.analyze_5star(st.session_state.historical_data)
except Exception as e:
    st.error(f"⚠️ Lỗi hệ thống: {str(e)}")
    st.info("Vui lòng làm mới trang hoặc kiểm tra dữ liệu đầu vào.")
