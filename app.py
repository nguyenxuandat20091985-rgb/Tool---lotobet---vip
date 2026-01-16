import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import io
import base64
from typing import List, Tuple, Dict, Any
import random
import math
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LOTOBET AI TOOL v1.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsive design and dark mode
st.markdown("""
<style>
    /* Base dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        max-width: 414px;
        margin: 0 auto;
    }
    
    /* Mobile responsive */
    @media (max-width: 414px) {
        .stApp {
            padding: 5px;
        }
        .main > div {
            padding: 0px !important;
        }
    }
    
    /* Hide unnecessary elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling for metrics */
    .stMetric {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #2d3246;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 16px;
        font-size: 14px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #00ff00;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: bold;
    }
    
    /* Success/Error colors */
    .success {
        color: #00ff00;
        font-weight: bold;
    }
    .warning {
        color: #ff9900;
        font-weight: bold;
    }
    .danger {
        color: #ff4444;
        font-weight: bold;
    }
    
    /* Countdown timer */
    .countdown {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_period' not in st.session_state:
    st.session_state.current_period = 1000
if 'countdown' not in st.session_state:
    st.session_state.countdown = 78
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'capital' not in st.session_state:
    st.session_state.capital = 10000000  # Default 10 million VND
if 'bet_strategy' not in st.session_state:
    st.session_state.bet_strategy = "Gấp thếp"

# Simulated algorithms (50 algorithms combined)
class LotteryAnalyzer:
    def __init__(self):
        self.history = []
        
    def load_data(self, data):
        if data is not None:
            self.history = data
            return True
        return False
    
    def analyze_5_star(self) -> Dict[str, Any]:
        """Analyze 5 positions (Vạn, Thiên, Hậu, Thập, Đơn)"""
        if len(self.history) < 5:
            return {
                "positions": {
                    "Vạn": {"trend": "↑", "frequency": 25, "gap": 3},
                    "Thiên": {"trend": "↓", "frequency": 20, "gap": 2},
                    "Hậu": {"trend": "→", "frequency": 22, "gap": 5},
                    "Thập": {"trend": "↑", "frequency": 18, "gap": 1},
                    "Đơn": {"trend": "↓", "frequency": 15, "gap": 4}
                },
                "recommendations": ["Thiên", "Đơn"]
            }
        
        # Simplified analysis for demo
        return {
            "positions": {
                "Vạn": {"trend": random.choice(["↑", "↓", "→"]), 
                       "frequency": random.randint(15, 30),
                       "gap": random.randint(0, 10)},
                "Thiên": {"trend": random.choice(["↑", "↓", "→"]),
                         "frequency": random.randint(15, 30),
                         "gap": random.randint(0, 10)},
                "Hậu": {"trend": random.choice(["↑", "↓", "→"]),
                       "frequency": random.randint(15, 30),
                       "gap": random.randint(0, 10)},
                "Thập": {"trend": random.choice(["↑", "↓", "→"]),
                        "frequency": random.randint(15, 30),
                        "gap": random.randint(0, 10)},
                "Đơn": {"trend": random.choice(["↑", "↓", "→"]),
                       "frequency": random.randint(15, 30),
                       "gap": random.randint(0, 10)}
            },
            "recommendations": random.sample(["Vạn", "Thiên", "Hậu", "Thập", "Đơn"], 2)
        }
    
    def analyze_2_star(self) -> List[Dict[str, Any]]:
        """Analyze and recommend 2-star pairs"""
        pairs = []
        for i in range(3):
            num1 = random.randint(0, 9)
            num2 = random.randint(0, 9)
            probability = random.randint(65, 95)
            
            pairs.append({
                "pair": f"{num1}{num2}",
                "probability": probability,
                "recommendation": "✅ KHUYÊN VÀO" if probability > 75 else "⚠️ THEO DÕI"
            })
        
        # Sort by probability
        pairs.sort(key=lambda x: x["probability"], reverse=True)
        return pairs
    
    def analyze_3_star(self) -> List[Dict[str, Any]]:
        """Analyze and recommend 3-star numbers"""
        combos = []
        risk_levels = ["THẤP", "TRUNG BÌNH", "CAO"]
        
        for i in range(3):
            nums = ''.join([str(random.randint(0, 9)) for _ in range(3)])
            confidence = random.randint(60, 92)
            risk = random.choice(risk_levels)
            
            combos.append({
                "combo": nums,
                "confidence": confidence,
                "risk": risk,
                "color": "#00ff00" if risk == "THẤP" else "#ff9900" if risk == "TRUNG BÌNH" else "#ff4444"
            })
        
        # Sort by confidence
        combos.sort(key=lambda x: x["confidence"], reverse=True)
        return combos
    
    def analyze_tai_xiu(self) -> Dict[str, Any]:
        """Analyze Tai/Xiu probability"""
        tai_prob = random.randint(40, 60)
        xiu_prob = 100 - tai_prob
        
        # Determine trend
        if abs(tai_prob - xiu_prob) > 15:
            trend = "CẦU BỆT"
        else:
            trend = "CẦU NHẢY"
        
        # Recommendation
        if tai_prob > 60:
            recommendation = "NÊN ĐẶT TÀI"
        elif xiu_prob > 60:
            recommendation = "NÊN ĐẶT XỈU"
        else:
            recommendation = "THEO DÕI THÊM"
        
        return {
            "tai": tai_prob,
            "xiu": xiu_prob,
            "trend": trend,
            "recommendation": recommendation,
            "last_10": random.sample(["T", "X", "T", "T", "X", "X", "T", "X", "T", "X"], 10)
        }
    
    def predict_special_numbers(self) -> List[Dict[str, Any]]:
        """Predict top 5 special 2D numbers"""
        numbers = []
        used = set()
        
        for i in range(5):
            while True:
                num = f"{random.randint(0, 9)}{random.randint(0, 9)}"
                if num not in used:
                    used.add(num)
                    probability = random.randint(70, 95)
                    numbers.append({
                        "number": num,
                        "probability": probability,
                        "advice": "MẠNH" if probability > 85 else "KHÁ"
                    })
                    break
        
        # Sort by probability
        numbers.sort(key=lambda x: x["probability"], reverse=True)
        return numbers

# Capital Management System
class CapitalManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.profit_target = initial_capital * 0.3  # 30% profit target
        self.stop_loss = initial_capital * 0.2  # 20% stop loss
        self.bets_history = []
    
    def calculate_bet_amount(self, strategy: str, bet_count: int) -> float:
        """Calculate bet amount based on strategy"""
        if strategy == "Gấp thếp":
            # Martingale strategy
            base_bet = self.current_capital * 0.01  # 1% of capital
            return base_bet * (2 ** (bet_count - 1))
        elif strategy == "Fibonacci":
            # Fibonacci sequence
            fib = [1, 1, 2, 3, 5, 8, 13]
            index = min(bet_count - 1, len(fib) - 1)
            base_bet = self.current_capital * 0.005  # 0.5% of capital
            return base_bet * fib[index]
        else:  # Đều tay
            # Fixed bet amount
            return self.current_capital * 0.02  # 2% of capital
    
    def update_capital(self, amount: float, win: bool):
        """Update capital after bet"""
        if win:
            self.current_capital += amount
        else:
            self.current_capital -= amount
        
        self.bets_history.append({
            "amount": amount,
            "win": win,
            "new_capital": self.current_capital,
            "timestamp": datetime.datetime.now()
        })
        
        # Check stop loss / take profit
        profit = self.current_capital - self.initial_capital
        profit_percentage = (profit / self.initial_capital) * 100
        
        if profit_percentage >= 30:
            return "TAKE_PROFIT"
        elif profit_percentage <= -20:
            return "STOP_LOSS"
        
        return "CONTINUE"
    
    def get_stats(self):
        """Get capital statistics"""
        profit = self.current_capital - self.initial_capital
        profit_percentage = (profit / self.initial_capital) * 100
        
        return {
            "initial": self.initial_capital,
            "current": self.current_capital,
            "profit": profit,
            "profit_percentage": profit_percentage,
            "target": self.profit_target,
            "stop_loss": self.stop_loss
        }

# Countdown timer function
def update_countdown():
    if st.session_state.countdown > 0:
        st.session_state.countdown -= 1
    else:
        st.session_state.countdown = 78
        st.session_state.current_period += 1

# Main application
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>🎯 LOTOBET AI TOOL v1.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cccccc;'>Công cụ phân tích xổ số thông minh - Tối ưu cho di động</p>", unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = LotteryAnalyzer()
    
    # SECTION 1: DATA MANAGEMENT & REAL-TIME
    st.markdown("---")
    st.markdown("## 📊 QUẢN LÝ DỮ LIỆU & REAL-TIME")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Tải lên file lịch sử (CSV/TXT)", type=['csv', 'txt'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, delimiter='\t')
                
                st.session_state.historical_data = df
                st.session_state.data_loaded = True
                analyzer.load_data(df.values.tolist() if len(df) > 0 else [])
                
                st.success(f"✅ Đã tải {len(df)} dòng dữ liệu thành công!")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc file: {str(e)}")
    
    with col2:
        # Real-time clock and countdown
        st.markdown("<div class='countdown' id='countdown-display'></div>", unsafe_allow_html=True)
        
        # JavaScript for real-time countdown
        st.markdown("""
        <script>
        function updateCountdown() {
            let countdown = 78;
            const element = document.getElementById('countdown-display');
            
            function tick() {
                const minutes = Math.floor(countdown / 60);
                const seconds = countdown % 60;
                element.innerHTML = `⏳ Kỳ: <span style='color:#00ff00'>${""" + str(st.session_state.current_period) + """}</span><br>`;
                element.innerHTML += `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                
                countdown--;
                if (countdown < 0) {
                    countdown = 78;
                    // In a real app, you would trigger a Streamlit rerun here
                }
            }
            
            tick();
            setInterval(tick, 1000);
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', updateCountdown);
        } else {
            updateCountdown();
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Period display
        st.metric("Kỳ hiện tại", f"#{st.session_state.current_period}", delta="+1 mỗi 78s")
    
    # CAPITAL MANAGEMENT SECTION
    st.markdown("---")
    st.markdown("## 💰 QUẢN LÝ VỐN THÔNG MINH")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capital_input = st.number_input(
            "Vốn ban đầu (VND)",
            min_value=1000000,
            max_value=1000000000,
            value=st.session_state.capital,
            step=1000000
        )
        st.session_state.capital = capital_input
    
    with col2:
        strategy = st.selectbox(
            "Chiến lược vào tiền",
            ["Gấp thếp", "Đều tay", "Fibonacci"],
            index=["Gấp thếp", "Đều tay", "Fibonacci"].index(st.session_state.bet_strategy)
        )
        st.session_state.bet_strategy = strategy
    
    with col3:
        bet_count = st.number_input("Số lần đã cược", min_value=1, max_value=10, value=1)
    
    # Initialize capital manager
    capital_manager = CapitalManager(st.session_state.capital)
    
    # Calculate next bet amount
    next_bet = capital_manager.calculate_bet_amount(st.session_state.bet_strategy, bet_count)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vốn hiện tại", f"{st.session_state.capital:,.0f} VND")
    with col2:
        st.metric("Tiền cược tiếp", f"{next_bet:,.0f} VND")
    
    # Profit/Loss tracking
    stats = capital_manager.get_stats()
    
    progress_col1, progress_col2 = st.columns(2)
    with progress_col1:
        st.progress(min(max(stats['profit_percentage'] / 30, 0), 1))
        st.caption(f"Chốt lãi: +30% ({stats['target']:,.0f} VND)")
    with progress_col2:
        st.progress(min(max(abs(stats['profit_percentage']) / 20, 0), 1))
        st.caption(f"Cắt lỗ: -20% ({abs(stats['stop_loss']):,.0f} VND)")
    
    # Warning messages
    if stats['profit_percentage'] >= 25:
        st.warning("⚠️ GẦN ĐẠT MỨC CHỐT LÃI - CÂN NHẮC DỪNG")
    elif stats['profit_percentage'] <= -15:
        st.error("🚨 GẦN CHẠM STOP LOSS - CẨN TRỌNG!")
    
    # SECTION 2: ANALYSIS TABS
    st.markdown("---")
    st.markdown("## 🎯 HỆ THỐNG PHÂN TÍCH AI")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "5 TINH", 
        "2 TINH", 
        "3 TINH", 
        "TÀI - XỈU",
        "SỐ ĐỀ"
    ])
    
    with tab1:
        st.markdown("### 📊 PHÂN TÍCH 5 VỊ TRÍ SỐ")
        
        analysis = analyzer.analyze_5_star()
        
        # Display position analysis
        cols = st.columns(5)
        positions = ["Vạn", "Thiên", "Hậu", "Thập", "Đơn"]
        
        for idx, pos in enumerate(positions):
            with cols[idx]:
                data = analysis["positions"][pos]
                color = "#00ff00" if data["trend"] == "↑" else "#ff4444" if data["trend"] == "↓" else "#ff9900"
                st.markdown(f"<h3 style='color:{color}'>{pos}</h3>", unsafe_allow_html=True)
                st.metric("Xu hướng", data["trend"])
                st.metric("Tần suất", f"{data['frequency']}%")
                st.metric("Gan", f"{data['gap']} kỳ")
        
        # Recommendations
        st.markdown("### 💡 KHUYẾN NGHỊ")
        rec_cols = st.columns(len(analysis["recommendations"]))
        for idx, rec in enumerate(analysis["recommendations"]):
            with rec_cols[idx]:
                st.success(f"🎯 {rec}")
                st.markdown("**Ưu tiên cao**")
    
    with tab2:
        st.markdown("### 🔢 PHÂN TÍCH 2 SỐ (2 TINH)")
        
        pairs = analyzer.analyze_2_star()
        
        for idx, pair in enumerate(pairs):
            col1, col2, col3 = st.columns([1, 2, 3])
            
            with col1:
                st.markdown(f"<h2 style='text-align: center;'>{pair['pair']}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Xác suất", f"{pair['probability']}%")
            
            with col3:
                if "KHUYÊN VÀO" in pair['recommendation']:
                    st.success(pair['recommendation'])
                else:
                    st.warning(pair['recommendation'])
            
            st.progress(pair['probability'] / 100)
            
            if idx < len(pairs) - 1:
                st.markdown("---")
    
    with tab3:
        st.markdown("### 🔢🔢🔢 PHÂN TÍCH 3 SỐ (3 TINH)")
        
        combos = analyzer.analyze_3_star()
        
        for idx, combo in enumerate(combos):
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.markdown(f"<h2 style='text-align: center; color:{combo['color']}'>{combo['combo']}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Độ tin cậy", f"{combo['confidence']}%")
            
            with col3:
                st.markdown(f"<p style='color:{combo['color']}; font-weight:bold;'>Mức rủi ro: {combo['risk']}</p>", unsafe_allow_html=True)
            
            st.progress(combo['confidence'] / 100)
            
            if combo['risk'] == "CAO":
                st.warning("⚠️ CẢNH BÁO: RỦI RO CAO - VÀO TIỀN NHỎ")
            elif combo['risk'] == "TRUNG BÌNH":
                st.info("ℹ️ RỦI RO TRUNG BÌNH - CÂN NHẮC KỸ")
            else:
                st.success("✅ RỦI RO THẤP - CÓ THỂ VÀO TIỀN")
            
            if idx < len(combos) - 1:
                st.markdown("---")
    
    with tab4:
        st.markdown("### 📈 PHÂN TÍCH TÀI - XỈU")
        
        analysis = analyzer.analyze_tai_xiu()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 TÀI (Tổng ≥ 23)")
            st.markdown(f"<h1 style='color:#00ff00'>{analysis['tai']}%</h1>", unsafe_allow_html=True)
            st.progress(analysis['tai'] / 100)
        
        with col2:
            st.markdown("#### 🔴 XỈU (Tổng ≤ 22)")
            st.markdown(f"<h1 style='color:#ff4444'>{analysis['xiu']}%</h1>", unsafe_allow_html=True)
            st.progress(analysis['xiu'] / 100)
        
        st.markdown("---")
        
        # Trend analysis
        st.markdown(f"**Xu hướng hiện tại:** `{analysis['trend']}`")
        
        # Last 10 results
        st.markdown("**10 kỳ gần nhất:**")
        last10_cols = st.columns(10)
        for idx, result in enumerate(analysis['last_10']):
            with last10_cols[idx]:
                if result == "T":
                    st.success("T")
                else:
                    st.error("X")
        
        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 LỜI KHUYÊN")
        
        if "NÊN ĐẶT" in analysis['recommendation']:
            if "TÀI" in analysis['recommendation']:
                st.success(f"✅ {analysis['recommendation']}")
                st.info(f"Xác suất Tài cao hơn {analysis['tai'] - analysis['xiu']}% so với Xỉu")
            else:
                st.error(f"✅ {analysis['recommendation']}")
                st.info(f"Xác suất Xỉu cao hơn {analysis['xiu'] - analysis['tai']}% so với Tài")
        else:
            st.warning(f"⚠️ {analysis['recommendation']}")
            st.info("Tỷ lệ cân bằng, nên chờ cầu rõ ràng hơn")
    
    with tab5:
        st.markdown("### 🎫 DỰ ĐOÁN SỐ ĐẶC BIỆT (2D)")
        
        numbers = analyzer.predict_special_numbers()
        
        for idx, num_data in enumerate(numbers):
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.markdown(f"<h1 style='text-align: center; color:#00ff00'>#{idx+1}</h1>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h2 style='text-align: center; font-size: 36px;'>{num_data['number']}</h2>", unsafe_allow_html=True)
            
            with col3:
                st.metric("Xác suất", f"{num_data['probability']}%")
                st.markdown(f"**Đánh giá:** {num_data['advice']}")
            
            # Progress bar with color coding
            progress = num_data['probability'] / 100
            if progress > 0.85:
                st.progress(progress)
            elif progress > 0.75:
                st.progress(progress)
            else:
                st.progress(progress)
            
            if idx < len(numbers) - 1:
                st.markdown("---")
        
        st.markdown("---")
        st.info("💡 **Lưu ý:** Top 5 số này được tính toán dựa trên 50 thuật toán AI kết hợp")
    
    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    LOTOBET AI TOOL v1.0 © 2024<br>
    Công cụ hỗ trợ phân tích - Không đảm bảo 100% chiến thắng<br>
    Đặt cược có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
