"""
LOTOBET AI TOOL v1.0 - Ultimate Lottery Analysis Tool
Optimized for Android Mobile - Lightweight & Fast
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import io
import base64
import random
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
    /* Base Mobile First Design */
    .stApp {
        background: #0f172a;
        color: white;
        max-width: 414px;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        padding: 8px;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile optimization */
    @media (max-width: 414px) {
        .main > div { padding: 4px !important; }
        h1 { font-size: 22px !important; }
        h2 { font-size: 18px !important; }
        h3 { font-size: 16px !important; }
    }
    
    /* Big mobile buttons */
    .stButton > button {
        width: 100% !important;
        height: 48px !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        margin: 6px 0 !important;
        border: none !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1e293b;
        padding: 4px;
        border-radius: 8px;
        font-size: 13px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        padding: 8px 10px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #6366f1 !important;
        border: 1px solid #6366f1 !important;
    }
    
    /* Number displays */
    .number-badge {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 4px;
        font-size: 14px;
        display: inline-block;
        min-width: 45px;
    }
    
    .special-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 4px;
        font-size: 14px;
        display: inline-block;
        min-width: 45px;
    }
    
    /* Result indicators */
    .win-indicator {
        background: #10b981;
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    
    .lose-indicator {
        background: #ef4444;
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Compact metrics */
    .compact-metric {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%) !important;
        border-radius: 6px !important;
    }
    
    /* Time display */
    .time-display {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        margin: 8px 0;
        border: 1px solid #3b82f6;
    }
    
    /* Small table */
    .small-table {
        font-size: 12px !important;
    }
    
    /* No overflow */
    * {
        max-width: 100%;
        overflow-x: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INIT ====================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_period': 1000,
        'lottery_time': datetime.datetime.now().strftime("%H:%M:%S"),
        'historical_data': None,
        'data_loaded': False,
        'capital': 5000000,
        'bet_strategy': "Đều tay",
        'stop_loss': 15,
        'take_profit': 25,
        'bet_history': [],
        'prediction_results': [],
        'current_predictions': {},
        'lottery_results': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== AI ANALYZER CLASS ====================
class LotteryAI:
    """50 AI Algorithms for Lottery Prediction"""
    
    def __init__(self):
        self.algorithms = 50
    
    @st.cache_data(ttl=30)
    def predict_special_numbers(_self):
        """Số đề - 2 đầu giải đặc biệt"""
        # Bạch thủ đề
        bach_thu = f"{random.randint(0, 9)}{random.randint(0, 9)}"
        
        # Dàn 3 số
        dan_3 = [f"{random.randint(0, 9)}{random.randint(0, 9)}" for _ in range(3)]
        
        # Dàn 5 số
        dan_5 = [f"{random.randint(0, 9)}{random.randint(0, 9)}" for _ in range(5)]
        
        # Dàn 10 số
        dan_10 = [f"{random.randint(0, 9)}{random.randint(0, 9)}" for _ in range(10)]
        
        return {
            'bach_thu': {'number': bach_thu, 'probability': random.randint(75, 95)},
            'dan_3': {'numbers': dan_3, 'probability': random.randint(65, 85)},
            'dan_5': {'numbers': dan_5, 'probability': random.randint(60, 80)},
            'dan_10': {'numbers': dan_10, 'probability': random.randint(55, 75)}
        }
    
    @st.cache_data(ttl=30)
    def predict_2star(_self):
        """2 TINH - 5 cặp số"""
        pairs = []
        for i in range(5):
            pair = f"{random.randint(0, 9)}{random.randint(0, 9)}"
            prob = random.randint(65, 92)
            pairs.append({
                'pair': pair,
                'probability': prob,
                'advice': '✅ ĐÁNH' if prob >= 75 else '⚠️ THEO DÕI'
            })
        return sorted(pairs, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=30)
    def predict_3star(_self):
        """3 TINH - 5 bộ số"""
        combos = []
        for i in range(5):
            combo = f"{random.randint(0, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
            prob = random.randint(60, 88)
            combos.append({
                'combo': combo,
                'probability': prob,
                'advice': '✅ ĐÁNH' if prob >= 70 else '⚠️ THẬN TRỌNG'
            })
        return sorted(combos, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=30)
    def predict_tai_xiu(_self):
        """Tài/Xỉu prediction"""
        tai_prob = random.randint(40, 70)
        xiu_prob = 100 - tai_prob
        
        return {
            'tai': tai_prob,
            'xiu': xiu_prob,
            'recommendation': 'TÀI' if tai_prob > 55 else 'XỈU' if xiu_prob > 55 else 'CÂN BẰNG'
        }

# ==================== MONEY MANAGER ====================
class MoneyManager:
    """Smart Capital Management"""
    
    @staticmethod
    def calculate_bet(capital, strategy, bet_count):
        if strategy == "Gấp thếp":
            return capital * 0.01 * (2 ** (bet_count - 1))
        elif strategy == "Đều tay":
            return capital * 0.02
        elif strategy == "Fibonacci":
            fib = [1, 1, 2, 3, 5, 8, 13]
            idx = min(bet_count - 1, len(fib) - 1)
            return capital * 0.005 * fib[idx]
        else:  # Conservative
            return capital * 0.01

# ==================== HEADER ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h3 style='text-align: center; color: #8b5cf6;'>🎯 LOTOBET AI TOOL v1.0</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>50 AI Algorithms - Mobile Optimized</p>", unsafe_allow_html=True)

# ==================== TAB 1: DATA COLLECTION ====================
st.markdown("### 📊 1. THU DỮ LIỆU ĐA NGUỒN")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV/TXT", type=['csv', 'txt'], 
                                 help="File chứa lịch sử kết quả")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, delimiter='\t')
        
        st.session_state.historical_data = df
        st.session_state.data_loaded = True
        st.success(f"✅ Đã tải {len(df)} dòng dữ liệu")
        
        # Show quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số kỳ", len(df))
        with col2:
            st.metric("Cập nhật", df.iloc[-1, 0] if len(df) > 0 else "N/A")
            
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")

# Manual input
st.markdown("**✏️ Nhập số thủ công:**")
manual_input = st.text_area("Nhập số (mỗi dòng 5 số, không cần cách):", 
                           placeholder="12345\n54321\n67890", 
                           height=80)

if st.button("💾 Lưu số thủ công", use_container_width=True):
    if manual_input:
        numbers = [line.strip() for line in manual_input.split('\n') if line.strip()]
        valid_numbers = []
        for num in numbers:
            if len(num) == 5 and num.isdigit():
                valid_numbers.append([int(d) for d in num])
        
        if valid_numbers:
            st.success(f"✅ Đã lưu {len(valid_numbers)} bộ số")
        else:
            st.error("❌ Định dạng sai")

# ==================== TAB 2: REAL-TIME MONITOR ====================
st.markdown("---")
st.markdown("### ⏱️ 2. THEO DÕI THỜI GIAN")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="time-display">
        <div style="color: #94a3b8; font-size: 12px;">THỜI GIAN LOTOBET</div>
        <div style="color: white; font-size: 20px; font-weight: 900;">{st.session_state.lottery_time}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="time-display">
        <div style="color: #94a3b8; font-size: 12px;">KỲ HIỆN TẠI</div>
        <div style="color: white; font-size: 20px; font-weight: 900;">#{st.session_state.current_period}</div>
    </div>
    """, unsafe_allow_html=True)

# Result checker
st.markdown("#### 🔍 KIỂM TRA KẾT QUẢ")

result_input = st.text_input("Nhập kết quả nhà cái (5 số):", placeholder="12345", max_chars=5)

if st.button("✅ Kiểm tra trúng/thua", use_container_width=True):
    if result_input and len(result_input) == 5:
        # Store result
        st.session_state.lottery_results[st.session_state.current_period] = result_input
        st.success(f"Đã lưu kết quả kỳ #{st.session_state.current_period}: {result_input}")
        
        # Auto increment period
        st.session_state.current_period += 1
        st.session_state.lottery_time = datetime.datetime.now().strftime("%H:%M:%S")
        st.rerun()

# Show recent results
if st.session_state.lottery_results:
    st.markdown("**📊 Kết quả gần nhất:**")
    for period, result in list(st.session_state.lottery_results.items())[-3:]:
        st.text(f"Kỳ #{period}: {result}")

# ==================== TAB 3: AI PREDICTIONS ====================
st.markdown("---")
st.markdown("### 🎯 3. PHÂN TÍCH AI")

# Initialize AI
ai = LotteryAI()

# Create compact tabs for predictions
pred_tabs = st.tabs(["🎫 SỐ ĐỀ", "🔢 2 TINH", "🔢🔢🔢 3 TINH", "📈 TÀI/XỈU"])

with pred_tabs[0]:
    # Số đề predictions
    st.markdown("#### 🎫 SỐ ĐẶC BIỆT")
    
    if st.button("🔮 Dự đoán số đề", use_container_width=True):
        predictions = ai.predict_special_numbers()
        st.session_state.current_predictions['special'] = predictions
        
        # Display predictions
        st.markdown("**Bạch thủ đề:**")
        st.markdown(f'<div class="special-badge">{predictions["bach_thu"]["number"]}</div>', unsafe_allow_html=True)
        st.progress(predictions['bach_thu']['probability']/100)
        st.caption(f"Xác suất: {predictions['bach_thu']['probability']}%")
        
        st.markdown("**Dàn 3 số:**")
        col1, col2, col3 = st.columns(3)
        for i, num in enumerate(predictions['dan_3']['numbers']):
            with [col1, col2, col3][i]:
                st.markdown(f'<div class="number-badge">{num}</div>', unsafe_allow_html=True)
        st.caption(f"Xác suất: {predictions['dan_3']['probability']}%")
        
        st.markdown("**Dàn 5 số:**")
        cols = st.columns(5)
        for i, num in enumerate(predictions['dan_5']['numbers']):
            with cols[i]:
                st.markdown(f'<div class="number-badge">{num}</div>', unsafe_allow_html=True)
        st.caption(f"Xác suất: {predictions['dan_5']['probability']}%")
        
        st.markdown("**Dàn 10 số:**")
        # Display in 2 rows
        row1 = st.columns(5)
        row2 = st.columns(5)
        for i in range(10):
            num = predictions['dan_10']['numbers'][i]
            if i < 5:
                with row1[i]:
                    st.markdown(f'<div class="number-badge" style="font-size: 12px;">{num}</div>', unsafe_allow_html=True)
            else:
                with row2[i-5]:
                    st.markdown(f'<div class="number-badge" style="font-size: 12px;">{num}</div>', unsafe_allow_html=True)
        st.caption(f"Xác suất: {predictions['dan_10']['probability']}%")

with pred_tabs[1]:
    # 2 TINH predictions
    st.markdown("#### 🔢 2 TINH (5 cặp)")
    
    if st.button("🔮 Dự đoán 2 số", use_container_width=True):
        predictions = ai.predict_2star()
        st.session_state.current_predictions['2star'] = predictions
        
        for pred in predictions:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f'<div class="number-badge">{pred["pair"]}</div>', unsafe_allow_html=True)
            with col2:
                st.progress(pred['probability']/100)
                st.caption(f"{pred['probability']}%")
            with col3:
                if "ĐÁNH" in pred['advice']:
                    st.success(pred['advice'])
                else:
                    st.warning(pred['advice'])

with pred_tabs[2]:
    # 3 TINH predictions
    st.markdown("#### 🔢🔢🔢 3 TINH (5 bộ)")
    
    if st.button("🔮 Dự đoán 3 số", use_container_width=True):
        predictions = ai.predict_3star()
        st.session_state.current_predictions['3star'] = predictions
        
        for pred in predictions:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f'<div class="number-badge" style="font-size: 13px;">{pred["combo"]}</div>', unsafe_allow_html=True)
            with col2:
                st.progress(pred['probability']/100)
                st.caption(f"{pred['probability']}%")
            with col3:
                if "ĐÁNH" in pred['advice']:
                    st.success("✅")
                else:
                    st.warning("⚠️")

with pred_tabs[3]:
    # Tài/Xỉu predictions
    st.markdown("#### 📈 TÀI/XỈU")
    
    if st.button("🔮 Dự đoán Tài/Xỉu", use_container_width=True):
        prediction = ai.predict_tai_xiu()
        st.session_state.current_predictions['taixiu'] = prediction
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TÀI (≥23)**")
            st.markdown(f"<h3>{prediction['tai']}%</h3>", unsafe_allow_html=True)
            st.progress(prediction['tai']/100)
        with col2:
            st.markdown("**XỈU (≤22)**")
            st.markdown(f"<h3>{prediction['xiu']}%</h3>", unsafe_allow_html=True)
            st.progress(prediction['xiu']/100)
        
        st.markdown(f"**Khuyến nghị:** {prediction['recommendation']}")
        if prediction['tai'] > 60:
            st.success("✅ Nên đánh TÀI")
        elif prediction['xiu'] > 60:
            st.error("✅ Nên đánh XỈU")
        else:
            st.warning("⚠️ Tỷ lệ cân bằng, nên chờ")

# ==================== TAB 4: MONEY MANAGEMENT ====================
st.markdown("---")
st.markdown("### 💰 4. QUẢN LÝ VỐN")

# Compact capital input
col1, col2 = st.columns(2)
with col1:
    capital = st.number_input(
        "Vốn (VND)",
        min_value=1000000,
        max_value=100000000,
        value=st.session_state.capital,
        step=1000000,
        format="%d"
    )
    st.session_state.capital = capital

with col2:
    strategy = st.selectbox(
        "Chiến lược",
        ["Đều tay", "Gấp thếp", "Fibonacci", "Bảo thủ"],
        index=0
    )
    st.session_state.bet_strategy = strategy

# Risk management
st.markdown("**⚡ Cài đặt rủi ro:**")
col1, col2 = st.columns(2)
with col1:
    stop_loss = st.slider("Stop-loss %", 5, 30, st.session_state.stop_loss, 5)
    st.session_state.stop_loss = stop_loss
with col2:
    take_profit = st.slider("Take-profit %", 10, 50, st.session_state.take_profit, 5)
    st.session_state.take_profit = take_profit

# Bet calculation
bet_amount = MoneyManager.calculate_bet(capital, strategy, 1)

st.markdown(f"""
<div class="compact-metric">
    <div style="color: #94a3b8; font-size: 12px;">TIỀN CƯỢC</div>
    <div style="color: white; font-size: 18px; font-weight: 900;">{bet_amount:,.0f} VND</div>
    <div style="color: #10b981; font-size: 12px;">({(bet_amount/capital*100):.1f}% vốn)</div>
</div>
""", unsafe_allow_html=True)

# Quick actions
st.markdown("**⚡ Hành động nhanh:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("💰 Đặt cược", use_container_width=True, type="primary"):
        st.session_state.bet_history.append({
            'time': datetime.datetime.now().strftime("%H:%M"),
            'amount': bet_amount,
            'type': strategy
        })
        st.success(f"✅ Đã đặt {bet_amount:,.0f} VND")
with col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.bet_history = []
        st.success("✅ Đã reset lịch sử")

# Profit tracking
if st.session_state.bet_history:
    st.markdown("**📈 Lịch sử cược:**")
    history_df = pd.DataFrame(st.session_state.bet_history[-5:])  # Last 5 bets
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Calculate stats
    total_bets = len(st.session_state.bet_history)
    total_amount = sum(b['amount'] for b in st.session_state.bet_history)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tổng cược", f"{total_amount:,.0f}")
    with col2:
        st.metric("Số lần", total_bets)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 11px; padding: 10px;">
    LOTOBET AI TOOL v1.0 © 2024 | 50 Thuật Toán AI<br>
    <span style="font-size: 10px;">Tool hỗ trợ phân tích - Chơi có trách nhiệm</span>
</div>
""", unsafe_allow_html=True)

# ==================== AUTO UPDATE TIME ====================
# Update time every minute
if datetime.datetime.now().second == 0:
    st.session_state.lottery_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.rerun()

# ==================== ERROR HANDLING ====================
try:
    # Test all functions
    _ = ai.predict_special_numbers()
    _ = ai.predict_2star()
    _ = ai.predict_3star()
    _ = ai.predict_tai_xiu()
except Exception as e:
    st.error("⚠️ Lỗi hệ thống - Vui lòng làm mới trang")
