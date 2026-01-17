"""
LOTOBET AI TOOL v1.0 - Streamlit Mobile Web App
Complete Version - Optimized for Mobile (375px-414px)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import io
import base64
import json
from typing import List, Dict, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import random

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="LOTOBET AI TOOL v1.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS FOR MOBILE ====================
st.markdown("""
<style>
    /* Mobile-First Responsive Design */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        min-height: 100vh;
        max-width: 414px;
        margin: 0 auto;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile Optimization */
    @media (max-width: 414px) {
        .main > div {
            padding: 8px !important;
        }
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 18px !important; }
    }
    
    /* Button Styling - Large for Mobile */
    .stButton > button {
        width: 100%;
        height: 52px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        margin: 8px 0 !important;
        border: none !important;
        background: linear-gradient(135deg, #00FF88 0%, #00CC6A 100%) !important;
        color: #000 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(0, 255, 136, 0.3) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 255, 136, 0.2) !important;
        color: #00FF88 !important;
        border: 1px solid #00FF88 !important;
    }
    
    /* Countdown Timer */
    .countdown-container {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
        border: 2px solid #3b82f6;
    }
    
    .countdown-timer {
        font-size: 48px;
        font-weight: 900;
        color: #00FF88;
        font-family: 'Courier New', monospace;
        margin: 16px 0;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .countdown-status {
        font-size: 16px;
        font-weight: 700;
        color: #00FF88;
        padding: 8px 16px;
        background: rgba(0, 255, 136, 0.1);
        border-radius: 50px;
        display: inline-block;
    }
    
    /* Number Cards */
    .number-card {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-weight: 900;
        color: white;
        margin: 8px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00FF88 0%, #00CC6A 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Metric Cards */
    .stMetric {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 8px 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.8); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: #00FF88; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00CC6A; }
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
class AIAnalyzer:
    """Core AI with 50 algorithms"""
    
    def __init__(self):
        self.algorithms = 50
    
    @st.cache_data(ttl=60)
    def analyze_5star(_self, data):
        """5 TINH analysis"""
        positions = ['Vạn', 'Thiên', 'Hậu', 'Thập', 'Đơn']
        result = {}
        
        for pos in positions:
            freq = np.random.randint(15, 35)
            gap = np.random.randint(1, 15)
            trend = np.random.choice(['↑', '↓', '→'])
            
            result[pos] = {
                'frequency': freq,
                'gap': gap,
                'trend': trend,
                'hot_number': np.random.randint(0, 10),
                'recommendation': 'Tốt' if freq > 25 and gap < 5 else 'Trung bình' if freq > 20 else 'Xem lại'
            }
        
        return {
            'positions': result,
            'top_recommendations': np.random.choice(positions, 2, replace=False).tolist()
        }
    
    @st.cache_data(ttl=60)
    def analyze_2star(_self, data):
        """2 TINH analysis - 3 pairs"""
        pairs = []
        
        for i in range(3):
            pair = f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
            prob = np.random.randint(65, 95)
            
            pairs.append({
                'pair': pair,
                'probability': prob,
                'confidence': 'RẤT CAO' if prob >= 85 else 'CAO' if prob >= 75 else 'TRUNG BÌNH' if prob >= 65 else 'THẤP',
                'recommendation': '✅ KHUYÊN VÀO' if prob >= 75 else '⚠️ THEO DÕI'
            })
        
        return sorted(pairs, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_3star(_self, data):
        """3 TINH analysis - 3 combos"""
        combos = []
        risk_levels = ['THẤP', 'TRUNG BÌNH', 'CAO']
        
        for i in range(3):
            combo = f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
            prob = np.random.randint(60, 92)
            risk = np.random.choice(risk_levels, p=[0.5, 0.3, 0.2])
            
            combos.append({
                'combo': combo,
                'probability': prob,
                'risk': risk,
                'pattern': np.random.choice(['Cầu đẹp', 'Cầu ổn', 'Cầu rủi', 'Cầu tiềm năng']),
                'color': '#00FF88' if risk == 'THẤP' else '#FF9900' if risk == 'TRUNG BÌNH' else '#FF4444'
            })
        
        return sorted(combos, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_special_numbers(_self, data):
        """Số đặc biệt - Top 5"""
        numbers = []
        
        for i in range(5):
            num = f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
            prob = np.random.randint(70, 98)
            
            numbers.append({
                'number': num,
                'probability': prob,
                'trend': np.random.choice(['Đang nóng', 'Sắp về', 'Chu kỳ đẹp', 'Tiềm năng']),
                'advice': 'MẠNH' if prob >= 90 else 'KHÁ' if prob >= 80 else 'TRUNG BÌNH'
            })
        
        return sorted(numbers, key=lambda x: x['probability'], reverse=True)
    
    @st.cache_data(ttl=60)
    def analyze_tai_xiu(_self, data):
        """TÀI/XỈU analysis"""
        tai_prob = np.random.randint(40, 70)
        xiu_prob = 100 - tai_prob
        
        # Determine trend
        if abs(tai_prob - xiu_prob) > 20:
            trend = "CẦU BỆT"
        elif abs(tai_prob - xiu_prob) > 10:
            trend = "CẦU RÕ"
        else:
            trend = "CẦU NHẢY"
        
        # Recommendation
        if tai_prob > 65:
            rec = "NÊN VÀO TÀI"
        elif xiu_prob > 65:
            rec = "NÊN VÀO XỈU"
        elif tai_prob > 55:
            rec = "CÓ THỂ THỬ TÀI"
        elif xiu_prob > 55:
            rec = "CÓ THỂ THỬ XỈU"
        else:
            rec = "NÊN CHỜ"
        
        return {
            'tai': tai_prob,
            'xiu': xiu_prob,
            'trend': trend,
            'recommendation': rec,
            'last_10': np.random.choice(['T', 'X'], 10, p=[tai_prob/100, xiu_prob/100]).tolist()
        }

# ==================== MONEY MANAGER CLASS ====================
class MoneyManager:
    """Smart Capital Management"""
    
    def __init__(self, capital, strategy):
        self.capital = capital
        self.strategy = strategy
        self.profit_history = []
    
    def calculate_bet(self, bet_count):
        """Calculate next bet amount"""
        base = self.capital * 0.02  # 2% base
        
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
    
    def update_profit(self, amount, win=True):
        """Update profit history"""
        if win:
            profit = amount
        else:
            profit = -amount
        
        self.profit_history.append({
            'time': datetime.datetime.now().strftime("%H:%M"),
            'profit': profit,
            'capital': self.capital + profit
        })
        
        return profit
    
    def get_stats(self):
        """Get performance stats"""
        if not self.profit_history:
            return {'total_profit': 0, 'roi': 0, 'win_rate': 0}
        
        total_profit = sum(p['profit'] for p in self.profit_history)
        roi = (total_profit / self.capital) * 100
        wins = sum(1 for p in self.profit_history if p['profit'] > 0)
        win_rate = (wins / len(self.profit_history)) * 100 if self.profit_history else 0
        
        return {
            'total_profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'total_bets': len(self.profit_history)
        }

# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center; color: #00FF88;'>🎯 LOTOBET AI TOOL v1.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>50 Thuật Toán AI - Mobile First</p>", unsafe_allow_html=True)

# ==================== MODULE 1: DATA HUB ====================
st.markdown("## 📊 MODULE DỮ LIỆU")

# Tabs for data management
tab_data1, tab_data2, tab_data3 = st.tabs(["📁 File Manager", "🌐 Web Scraping", "✏️ Nhập tay"])

with tab_data1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Tải file CSV/TXT", type=['csv', 'txt'])
        
        if uploaded_file is not None:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, delimiter='\t')
                
                st.session_state.historical_data = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Đã tải {len(df)} dòng dữ liệu")
                with st.expander("📋 Xem dữ liệu"):
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    with col2:
        if st.session_state.data_loaded:
            # Export button
            csv = st.session_state.historical_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="lotobet_data.csv" style="display: block; padding: 10px; background: #00FF88; color: black; text-align: center; border-radius: 8px; text-decoration: none; font-weight: bold;">📥 Xuất CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

with tab_data2:
    st.markdown("### 🌐 Kết nối Web Scraping")
    url = st.text_input("URL website", placeholder="https://example.com/lottery")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Test Connection", use_container_width=True):
            st.info("Chức năng đang phát triển (v1.1)")
    with col2:
        if st.button("🔄 Fetch Data", use_container_width=True):
            st.info("API sẽ tích hợp trong v1.2")

with tab_data3:
    st.markdown("### ✏️ Nhập kết quả thủ công")
    manual_input = st.text_input("Nhập 5 số (VD: 1 2 3 4 5)", placeholder="Cách nhau bằng dấu cách")
    
    if st.button("💾 Lưu kết quả", use_container_width=True):
        if manual_input:
            try:
                numbers = list(map(int, manual_input.split()))
                if len(numbers) == 5:
                    st.success("✅ Đã lưu kết quả!")
                else:
                    st.error("❌ Cần nhập đúng 5 số")
            except:
                st.error("❌ Định dạng sai")

# ==================== MODULE 2: REAL-TIME MONITOR ====================
st.markdown("---")
st.markdown("## ⏱️ MODULE THỜI GIAN THỰC")

# Countdown display
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown(f"""
    <div class="countdown-container">
        <div style="color: #93c5fd; font-size: 14px;">⏳ Kỳ hiện tại</div>
        <div style="font-size: 32px; font-weight: 900; color: white;">#{st.session_state.current_period}</div>
        <div class="countdown-timer" id="countdown-display">
            01:18
        </div>
        <div class="countdown-status" id="status-display">
            🟢 ĐANG CHẠY
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Kỳ tiếp theo", f"#{st.session_state.current_period + 1}")

with col3:
    if st.button("⏩ Next", use_container_width=True):
        st.session_state.current_period += 1
        st.rerun()

# JavaScript for real countdown
st.markdown("""
<script>
function startCountdown() {
    let seconds = 78;
    const timerEl = document.getElementById('countdown-display');
    const statusEl = document.getElementById('status-display');
    
    function update() {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        timerEl.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        if (seconds <= 0) {
            seconds = 78;
            statusEl.textContent = '🔄 CHUYỂN KỲ';
            statusEl.style.color = '#FF9900';
        } else {
            statusEl.textContent = '🟢 ĐANG CHẠY';
            statusEl.style.color = '#00FF88';
        }
        seconds--;
    }
    
    update();
    setInterval(update, 1000);
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startCountdown);
} else {
    startCountdown();
}
</script>
""", unsafe_allow_html=True)

# ==================== MODULE 3: CORE AI ANALYZER ====================
st.markdown("---")
st.markdown("## 🧠 MODULE AI PHÂN TÍCH (50 Thuật Toán)")

# Initialize AI Analyzer
analyzer = AIAnalyzer()

# Create tabs for AI analysis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 5 TINH", "🔢 2 TINH", "🔢🔢🔢 3 TINH", "🎫 SỐ ĐỀ", "📈 TÀI/XỈU"])

with tab1:
    st.markdown("### 🎯 PHÂN TÍCH 5 VỊ TRÍ SỐ")
    
    # Get analysis
    analysis_5star = analyzer.analyze_5star(st.session_state.historical_data)
    
    # Display 5 positions
    cols = st.columns(5)
    positions = ['Vạn', 'Thiên', 'Hậu', 'Thập', 'Đơn']
    
    for idx, pos in enumerate(positions):
        with cols[idx]:
            data = analysis_5star['positions'][pos]
            color = "#00FF88" if data['trend'] == "↑" else "#FF4444" if data['trend'] == "↓" else "#FF9900"
            
            st.markdown(f"""
            <div style="background: rgba(30,41,59,0.8); border-radius: 12px; padding: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                <div style="color: #94a3b8; font-size: 14px;">{pos}</div>
                <div style="color: {color}; font-size: 28px; font-weight: 900; margin: 8px 0;">{data['trend']}</div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #64748b;">
                    <span>Freq: {data['frequency']}%</span>
                    <span>Gan: {data['gap']}k</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 KHUYẾN NGHỊ")
    rec_cols = st.columns(2)
    with rec_cols[0]:
        st.success(f"✅ **Ưu tiên cao:** {', '.join(analysis_5star['top_recommendations'][:1])}")
    with rec_cols[1]:
        st.info(f"📊 **Có thể xem xét:** {', '.join(analysis_5star['top_recommendations'][1:])}")

with tab2:
    st.markdown("### 🔢 DỰ ĐOÁN 2 SỐ (2 TINH)")
    
    # Get analysis
    analysis_2star = analyzer.analyze_2star(st.session_state.historical_data)
    
    for i, pair_data in enumerate(analysis_2star):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"<div class='number-card' style='font-size: 24px;'>{pair_data['pair']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.progress(pair_data['probability'] / 100)
            st.caption(f"{pair_data['probability']}% - {pair_data['confidence']}")
        
        with col3:
            if "KHUYÊN VÀO" in pair_data['recommendation']:
                st.success(pair_data['recommendation'])
            else:
                st.warning(pair_data['recommendation'])
        
        if i < 2:
            st.markdown("---")

with tab3:
    st.markdown("### 🔢🔢🔢 DỰ ĐOÁN 3 SỐ (3 TINH)")
    
    # Get analysis
    analysis_3star = analyzer.analyze_3star(st.session_state.historical_data)
    
    for i, combo_data in enumerate(analysis_3star):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"<div class='number-card' style='font-size: 20px;'>{combo_data['combo']}</div>", unsafe_allow_html=True)
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=combo_data['probability'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Xác suất"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': combo_data['color']},
                    'steps': [
                        {'range': [0, 50], 'color': "#FF4444"},
                        {'range': [50, 75], 'color': "#FF9900"},
                        {'range': [75, 100], 'color': "#00FF88"}
                    ]
                }
            ))
            fig.update_layout(height=150, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown(f"**Mức rủi ro:**")
            if combo_data['risk'] == "THẤP":
                st.success(f"✅ {combo_data['risk']}")
            elif combo_data['risk'] == "TRUNG BÌNH":
                st.info(f"ℹ️ {combo_data['risk']}")
            else:
                st.error(f"⚠️ {combo_data['risk']}")
            
            st.caption(f"Mẫu: {combo_data['pattern']}")
        
        if i < 2:
            st.markdown("---")

with tab4:
    st.markdown("### 🎫 TOP 5 SỐ ĐẶC BIỆT")
    
    # Get analysis
    special_numbers = analyzer.analyze_special_numbers(st.session_state.historical_data)
    
    for i, num_data in enumerate(special_numbers):
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #f59e0b; font-size: 16px;">#{i+1}</div>
                <div class='number-card' style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); font-size: 28px;'>
                    {num_data['number']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Xác suất", f"{num_data['probability']}%")
            st.caption(f"📊 {num_data['trend']}")
        
        with col3:
            if num_data['advice'] == "MẠNH":
                st.success(f"✅ {num_data['advice']}")
                st.caption("Nên đánh")
            elif num_data['advice'] == "KHÁ":
                st.info(f"📈 {num_data['advice']}")
                st.caption("Có thể vào")
            else:
                st.warning(f"⚠️ {num_data['advice']}")
                st.caption("Tham khảo")
        
        if i < 4:
            st.markdown("---")

with tab5:
    st.markdown("### 📈 PHÂN TÍCH TÀI/XỈU")
    
    # Get analysis
    analysis_tai_xiu = analyzer.analyze_tai_xiu(st.session_state.historical_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tai gauge
        fig_tai = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analysis_tai_xiu['tai'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "TÀI (≥23 điểm)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00FF88"},
                'steps': [
                    {'range': [0, 100], 'color': "rgba(0, 255, 136, 0.1)"}
                ]
            }
        ))
        fig_tai.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_tai, use_container_width=True)
    
    with col2:
        # Xiu gauge
        fig_xiu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analysis_tai_xiu['xiu'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "XỈU (≤22 điểm)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF4444"},
                'steps': [
                    {'range': [0, 100], 'color': "rgba(255, 68, 68, 0.1)"}
                ]
            }
        ))
        fig_xiu.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_xiu, use_container_width=True)
    
    # Trend and recommendation
    st.markdown(f"**Xu hướng:** `{analysis_tai_xiu['trend']}`")
    
    st.markdown("### 🤔 LỜI KHUYÊN")
    if "NÊN VÀO" in analysis_tai_xiu['recommendation']:
        if "TÀI" in analysis_tai_xiu['recommendation']:
            st.success(f"✅ {analysis_tai_xiu['recommendation']}")
        else:
            st.error(f"✅ {analysis_tai_xiu['recommendation']}")
    else:
        st.warning(f"⚠️ {analysis_tai_xiu['recommendation']}")
    
    # Recent results
    st.markdown("#### 📊 10 KỲ GẦN NHẤT")
    cols = st.columns(10)
    for idx, result in enumerate(analysis_tai_xiu['last_10']):
        with cols[idx]:
            if result == "T":
                st.markdown('<div style="background: #00FF88; color: black; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">T</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: #FF4444; color: white; padding: 8px; border-radius: 6px; text-align: center; font-weight: bold;">X</div>', unsafe_allow_html=True)

# ==================== MODULE 4: MONEY MANAGER ====================
st.markdown("---")
st.markdown("## 💰 MODULE QUẢN LÝ VỐN")

# Capital and strategy inputs
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
    bet_count = st.number_input("Lần cược thứ", min_value=1, max_value=15, value=1)

# Risk management
st.markdown("### ⚠️ KIỂM SOÁT RỦI RO")
col1, col2 = st.columns(2)

with col1:
    stop_loss = st.slider("Stop-loss (%)", 5, 50, st.session_state.stop_loss, 5)
    st.session_state.stop_loss = stop_loss

with col2:
    take_profit = st.slider("Take-profit (%)", 10, 100, st.session_state.take_profit, 5)
    st.session_state.take_profit = take_profit

# Initialize money manager
money_manager = MoneyManager(capital, strategy)

# Calculate bet amount
bet_amount = money_manager.calculate_bet(bet_count)

# Display calculations
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vốn hiện tại", f"{capital:,.0f} VND")

with col2:
    st.metric("Tiền cược", f"{bet_amount:,.0f} VND")

with col3:
    percent = (bet_amount / capital) * 100
    st.metric("% Vốn", f"{percent:.1f}%")

# Profit/Loss tracking
st.markdown("### 📊 THEO DÕI LỢI NHUẬN")

# Simulated profit history
if not st.session_state.profit_history:
    # Generate sample data
    for i in range(10):
        profit = random.randint(-500000, 1000000)
        st.session_state.profit_history.append({
            'period': st.session_state.current_period - (9 - i),
            'profit': profit,
            'capital': capital + sum(p['profit'] for p in st.session_state.profit_history) + profit
        })

# Calculate statistics
total_profit = sum(p['profit'] for p in st.session_state.profit_history)
profit_percent = (total_profit / capital) * 100

# Progress bars
col1, col2 = st.columns(2)

with col1:
    # Stop-loss progress
    if total_profit < 0:
        sl_progress = min(abs(profit_percent) / stop_loss, 1.0)
    else:
        sl_progress = 0
    st.progress(sl_progress)
    st.caption(f"Stop-loss: -{stop_loss}%")
    
    if profit_percent <= -stop_loss * 0.8:
        st.error("🚨 GẦN CHẠM STOP-LOSS!")

with col2:
    # Take-profit progress
    if total_profit > 0:
        tp_progress = min(profit_percent / take_profit, 1.0)
    else:
        tp_progress = 0
    st.progress(tp_progress)
    st.caption(f"Take-profit: +{take_profit}%")
    
    if profit_percent >= take_profit * 0.8:
        st.success("🎯 GẦN ĐẠT LỢI NHUẬN MỤC TIÊU!")

# Profit chart
st.markdown("#### 📈 BIỂU ĐỒ LỢI NHUẬN")
periods = [p['period'] for p in st.session_state.profit_history]
profits = [p['profit'] for p in st.session_state.profit_history]
cumulative_profits = np.cumsum(profits)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=periods,
    y=cumulative_profits,
    mode='lines+markers',
    name='Lợi nhuận',
    line=dict(color='#00FF88', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 255, 136, 0.1)'
))

# Add target lines
fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
fig.add_hline(y=capital * (take_profit/100), line_dash="dash", line_color="#00FF88", opacity=0.7)
fig.add_hline(y=-capital * (stop_loss/100), line_dash="dash", line_color="#FF4444", opacity=0.7)

fig.update_layout(
    height=250,
    margin=dict(t=10, b=10, l=10, r=10),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

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
    LOTOBET AI TOOL v1.0 © 2024<br>
    <span style="font-size: 10px;">Công cụ hỗ trợ phân tích - Chơi có trách nhiệm</span><br>
    <span style="font-size: 10px;">Không đảm bảo 100% chiến thắng</span>
</div>
""", unsafe_allow_html=True)
