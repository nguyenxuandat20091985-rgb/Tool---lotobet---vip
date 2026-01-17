"""
LOTOBET ELITE v3.0 - Professional Lottery Analysis System
Enhanced Version with Advanced Algorithms & Mobile Optimization
Author: Senior Python Developer + Data Analyst
Version: 3.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="LOTOBET ELITE v3.0",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;700&display=swap');
    
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Premium Card - Glassmorphism */
    .elite-card {
        background: rgba(20, 25, 40, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 49, 49, 0.3);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 
            0 8px 32px rgba(255, 49, 49, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    
    .elite-card:hover {
        border-color: rgba(0, 255, 194, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 255, 194, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    /* Main Prediction Number - Neon Effect */
    .main-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 100px !important;
        font-weight: 900;
        background: linear-gradient(45deg, #FF3131, #FF6B6B, #00FFC2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 
            0 0 20px rgba(255, 49, 49, 0.5),
            0 0 40px rgba(255, 49, 49, 0.3),
            0 0 60px rgba(0, 255, 194, 0.2);
        margin: 0 !important;
        line-height: 1;
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 0.95; }
        to { opacity: 1; }
    }
    
    /* Professional Table Styling */
    .dataframe {
        background: rgba(15, 20, 35, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #FF3131, #B20000) !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        padding: 16px 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dataframe td {
        background: rgba(25, 30, 45, 0.9) !important;
        color: #ffffff !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 14px 12px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF3131, #B20000) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 28px !important;
        height: auto !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 49, 49, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00FFC2, #009970) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 255, 194, 0.4) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF3131, #00FFC2) !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15, 20, 35, 0.9) !important;
        color: white !important;
        border: 1px solid rgba(255, 49, 49, 0.3) !important;
        border-radius: 12px !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(255, 49, 49, 0.1), rgba(0, 255, 194, 0.1)) !important;
        border-radius: 12px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .main-number {
            font-size: 70px !important;
        }
        .elite-card {
            padding: 20px;
            margin-bottom: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'statistics' not in st.session_state:
    st.session_state.statistics = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'accuracy_rate': 0.0,
        'streak_current': 0,
        'streak_max': 0,
        'last_update': datetime.now()
    }

# ==================== ADVANCED ALGORITHM CLASS ====================
class EliteAnalyzerV3:
    """Enhanced Lottery Analysis Engine with 7-Layer Algorithm"""
    
    def __init__(self, data_list):
        self.data = data_list
        self.analyzed_digits = {str(i): {} for i in range(10)}
    
    def layer_1_frequency_analysis(self):
        """Layer 1: Comprehensive Frequency Analysis"""
        digit_appearance = {str(i): [] for i in range(10)}
        
        for idx, numbers in enumerate(self.data):
            for digit in digit_appearance:
                digit_appearance[digit].append(1 if digit in numbers else 0)
        
        for digit in digit_appearance:
            appearances = digit_appearance[digit]
            total = sum(appearances)
            frequency = total / len(self.data) if self.data else 0
            
            # Weighted frequency (recent period has higher weight)
            weights = np.linspace(0.5, 1.5, len(appearances))
            weighted_freq = np.dot(appearances, weights) / sum(weights) if appearances else 0
            
            self.analyzed_digits[digit]['frequency_total'] = frequency * 100
            self.analyzed_digits[digit]['frequency_weighted'] = weighted_freq * 100
            self.analyzed_digits[digit]['appearance_count'] = total
    
    def layer_2_gap_analysis(self):
        """Layer 2: Advanced Gap (Delay) Analysis"""
        for digit in self.analyzed_digits:
            last_seen = -1
            gaps = []
            current_gap = 0
            
            for idx, numbers in enumerate(self.data):
                if digit in numbers:
                    if last_seen != -1:
                        gaps.append(idx - last_seen)
                    last_seen = idx
                    current_gap = 0
                else:
                    current_gap += 1
            
            # Current gap
            current_gap = len(self.data) - last_seen - 1 if last_seen != -1 else len(self.data)
            
            if gaps:
                avg_gap = np.mean(gaps)
                max_gap = max(gaps)
                min_gap = min(gaps)
                gap_std = np.std(gaps)
            else:
                avg_gap = max_gap = min_gap = gap_std = len(self.data)
            
            # Gap score: smaller current gap = higher score
            gap_score = max(0, 100 - (current_gap * 10))
            
            self.analyzed_digits[digit]['gap_current'] = current_gap
            self.analyzed_digits[digit]['gap_average'] = avg_gap
            self.analyzed_digits[digit]['gap_score'] = gap_score
            self.analyzed_digits[digit]['gap_volatility'] = gap_std
    
    def layer_3_trend_analysis(self):
        """Layer 3: Multi-Period Trend Analysis"""
        periods = {
            'short': 5,
            'medium': 10,
            'long': 20
        }
        
        for digit in self.analyzed_digits:
            digit_trends = {}
            
            for period_name, period in periods.items():
                if len(self.data) >= period:
                    recent = self.data[-period:]
                    trend_score = (sum(1 for nums in recent if digit in nums) / period) * 100
                else:
                    trend_score = 0
                
                digit_trends[f'trend_{period_name}'] = trend_score
            
            # Trend momentum
            if len(self.data) >= 10:
                last_5 = self.data[-5:] if len(self.data) >= 5 else self.data
                prev_5 = self.data[-10:-5] if len(self.data) >= 10 else []
                
                trend_current = sum(1 for nums in last_5 if digit in nums) / len(last_5) if last_5 else 0
                trend_previous = sum(1 for nums in prev_5 if digit in nums) / len(prev_5) if prev_5 else 0
                
                momentum = ((trend_current - trend_previous) / (trend_previous + 0.01)) * 100
            else:
                momentum = 0
            
            self.analyzed_digits[digit].update(digit_trends)
            self.analyzed_digits[digit]['trend_momentum'] = momentum
    
    def layer_4_pattern_recognition(self):
        """Layer 4: Pattern and Cycle Detection"""
        for digit in self.analyzed_digits:
            appearance_pattern = []
            
            for numbers in self.data:
                appearance_pattern.append(1 if digit in numbers else 0)
            
            # Detect cycles of length 2-7
            cycle_scores = {}
            for cycle_len in range(2, 8):
                if len(appearance_pattern) >= cycle_len * 2:
                    pattern_score = 0
                    for i in range(len(appearance_pattern) - cycle_len):
                        pattern = appearance_pattern[i:i+cycle_len]
                        repeat_count = 0
                        for j in range(i+cycle_len, len(appearance_pattern)-cycle_len, cycle_len):
                            if appearance_pattern[j:j+cycle_len] == pattern:
                                repeat_count += 1
                        pattern_score += repeat_count
                    cycle_scores[cycle_len] = pattern_score
                else:
                    cycle_scores[cycle_len] = 0
            
            # Best cycle score
            if cycle_scores:
                best_cycle = max(cycle_scores.items(), key=lambda x: x[1])
                cycle_strength = min(best_cycle[1] * 10, 100)
                cycle_length = best_cycle[0]
            else:
                cycle_strength = 0
                cycle_length = 0
            
            self.analyzed_digits[digit]['cycle_strength'] = cycle_strength
            self.analyzed_digits[digit]['cycle_length'] = cycle_length
    
    def layer_5_cooccurrence_analysis(self):
        """Layer 5: Digit Co-occurrence Network"""
        cooccurrence_matrix = np.zeros((10, 10))
        
        for numbers in self.data:
            digits_in_draw = [int(d) for d in str(numbers)]
            for i in range(len(digits_in_draw)):
                for j in range(i+1, len(digits_in_draw)):
                    d1, d2 = digits_in_draw[i], digits_in_draw[j]
                    cooccurrence_matrix[d1][d2] += 1
                    cooccurrence_matrix[d2][d1] += 1
        
        for digit in self.analyzed_digits:
            d = int(digit)
            cooccurrence_score = np.sum(cooccurrence_matrix[d]) / len(self.data) * 10
            self.analyzed_digits[digit]['cooccurrence_score'] = min(cooccurrence_score, 100)
    
    def layer_6_entropy_analysis(self):
        """Layer 6: Information Entropy Analysis"""
        position_distribution = {str(i): [0, 0, 0, 0, 0] for i in range(10)}
        
        for numbers in self.data:
            num_str = str(numbers)
            for pos, char in enumerate(num_str):
                if char in position_distribution:
                    position_distribution[char][pos] += 1
        
        for digit in self.analyzed_digits:
            distribution = position_distribution.get(digit, [0]*5)
            total = sum(distribution)
            
            if total > 0:
                # Calculate entropy
                probs = [count/total for count in distribution]
                entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
                max_entropy = np.log2(5)
                normalized_entropy = (entropy / max_entropy) * 100
            else:
                normalized_entropy = 0
            
            # Position bias score (lower bias = better)
            if total > 0:
                position_bias = np.std(distribution) / (total/5 + 1e-10)
                bias_score = max(0, 100 - (position_bias * 200))
            else:
                bias_score = 50
            
            self.analyzed_digits[digit]['entropy_score'] = normalized_entropy
            self.analyzed_digits[digit]['position_bias_score'] = bias_score
    
    def layer_7_signal_processing(self):
        """Layer 7: Signal Processing & Noise Reduction"""
        for digit in self.analyzed_digits:
            # Combine all scores with intelligent weighting
            scores = {
                'frequency': self.analyzed_digits[digit].get('frequency_weighted', 0) * 0.25,
                'gap': self.analyzed_digits[digit].get('gap_score', 0) * 0.20,
                'trend': self.analyzed_digits[digit].get('trend_short', 0) * 0.15,
                'momentum': min(max(self.analyzed_digits[digit].get('trend_momentum', 0) + 50, 0), 100) * 0.10,
                'cycle': self.analyzed_digits[digit].get('cycle_strength', 0) * 0.10,
                'cooccurrence': self.analyzed_digits[digit].get('cooccurrence_score', 0) * 0.10,
                'entropy': self.analyzed_digits[digit].get('entropy_score', 0) * 0.05,
                'position_bias': self.analyzed_digits[digit].get('position_bias_score', 0) * 0.05
            }
            
            final_score = sum(scores.values())
            
            # Apply confidence adjustment based on data size
            data_size_factor = min(len(self.data) / 50, 1.0)
            confidence_adjusted_score = final_score * data_size_factor
            
            self.analyzed_digits[digit]['final_score'] = min(confidence_adjusted_score, 99.9)
            
            # Determine signal strength
            if confidence_adjusted_score >= 75:
                signal = "🔥 MẠNH"
                signal_color = "#FF3131"
            elif confidence_adjusted_score >= 60:
                signal = "⚡ KHÁ"
                signal_color = "#00FFC2"
            elif confidence_adjusted_score >= 45:
                signal = "🟡 TRUNG BÌNH"
                signal_color = "#FFD700"
            else:
                signal = "⚪ CHỜ"
                signal_color = "#AAAAAA"
            
            self.analyzed_digits[digit]['signal'] = signal
            self.analyzed_digits[digit]['signal_color'] = signal_color
    
    def analyze_all(self):
        """Run all 7 layers of analysis"""
        if len(self.data) < 3:
            return None
        
        self.layer_1_frequency_analysis()
        self.layer_2_gap_analysis()
        self.layer_3_trend_analysis()
        self.layer_4_pattern_recognition()
        self.layer_5_cooccurrence_analysis()
        self.layer_6_entropy_analysis()
        self.layer_7_signal_processing()
        
        return self.analyzed_digits

# ==================== HELPER FUNCTIONS ====================
def parse_input_data(input_text):
    """Parse and clean input data"""
    if not input_text:
        return []
    
    # Find all 5-digit numbers
    matches = re.findall(r'\b\d{5}\b', input_text)
    return matches

def create_visualization(analysis_results, top_number):
    """Create interactive visualization"""
    digits = list(analysis_results.keys())
    scores = [analysis_results[d]['final_score'] for d in digits]
    
    fig = go.Figure()
    
    # Bar chart
    colors = ['#FF3131' if str(i) == str(top_number) else 
              '#00FFC2' if score >= 60 else 
              '#FFD700' if score >= 45 else 
              '#666666' 
              for i, score in enumerate(scores)]
    
    fig.add_trace(go.Bar(
        x=digits,
        y=scores,
        marker_color=colors,
        text=[f"{score:.1f}%" for score in scores],
        textposition='auto',
        name="Điểm số"
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>PHÂN TÍCH CHI TIẾT 0-9</b>",
            font=dict(size=20, color='#FFFFFF', family='Orbitron')
        ),
        xaxis=dict(
            title="SỐ",
            tickfont=dict(size=14, color='#FFFFFF'),
            titlefont=dict(size=16, color='#00FFC2')
        ),
        yaxis=dict(
            title="ĐIỂM TỔNG HỢP (%)",
            range=[0, 100],
            tickfont=dict(size=12, color='#FFFFFF'),
            titlefont=dict(size=16, color='#00FFC2')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        showlegend=False,
        height=400
    )
    
    return fig

def update_statistics(is_correct):
    """Update prediction statistics"""
    st.session_state.statistics['total_predictions'] += 1
    if is_correct:
        st.session_state.statistics['correct_predictions'] += 1
        st.session_state.statistics['streak_current'] += 1
        if st.session_state.statistics['streak_current'] > st.session_state.statistics['streak_max']:
            st.session_state.statistics['streak_max'] = st.session_state.statistics['streak_current']
    else:
        st.session_state.statistics['streak_current'] = 0
    
    if st.session_state.statistics['total_predictions'] > 0:
        st.session_state.statistics['accuracy_rate'] = (
            st.session_state.statistics['correct_predictions'] / 
            st.session_state.statistics['total_predictions']
        ) * 100
    
    st.session_state.statistics['last_update'] = datetime.now()

# ==================== MAIN APPLICATION ====================
def main():
    """Main Streamlit Application"""
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #00FFC2; font-family: Orbitron, sans-serif; margin-bottom: 5px;'>
                💎 LOTOBET ELITE v3.0
            </h1>
            <p style='color: #AAAAAA; font-family: Roboto Mono, monospace;'>
                Hệ Thống Phân Tích & Dự Đoán 7 Tầng Thuật Toán
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Statistics
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ HỆ THỐNG")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="TỔNG DỰ ĐOÁN",
                value=st.session_state.statistics['total_predictions'],
                delta=None
            )
        with col2:
            st.metric(
                label="TỶ LỆ CHÍNH XÁC",
                value=f"{st.session_state.statistics['accuracy_rate']:.1f}%",
                delta=None
            )
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="CHUỖI HIỆN TẠI",
                value=st.session_state.statistics['streak_current'],
                delta=None
            )
        with col4:
            st.metric(
                label="CHUỖI CAO NHẤT",
                value=st.session_state.statistics['streak_max'],
                delta=None
            )
        
        st.markdown("---")
        st.markdown("### ⚙️ CÀI ĐẶT")
        
        auto_refresh = st.checkbox("Tự động làm mới", value=True)
        show_details = st.checkbox("Hiển thị chi tiết", value=True)
        
        if st.button("🔄 Đặt lại thống kê", type="secondary"):
            st.session_state.statistics = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy_rate': 0.0,
                'streak_current': 0,
                'streak_max': 0,
                'last_update': datetime.now()
            }
            st.rerun()
    
    # Main Content Area
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Data Input Section
        st.markdown("""
            <div class='elite-card'>
                <h3 style='color: #00FFC2; text-align: center; margin-bottom: 20px;'>
                    🔌 NẠP DỮ LIỆU KẾT QUẢ
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Chọn phương thức nhập:",
            ["Dán chuỗi số", "Nhập từng số", "Import từ file"],
            horizontal=True
        )
        
        if input_method == "Dán chuỗi số":
            input_data = st.text_area(
                "Dán chuỗi số từ bất kỳ nguồn nào:",
                height=150,
                placeholder="""Ví dụ:
84729 10293 88273 45612 78901
23456 89012 34567 90123 45678
..."""
            )
        elif input_method == "Nhập từng số":
            single_input = st.text_input("Nhập số 5 chữ số:", max_chars=5)
            if single_input and len(single_input) == 5 and single_input.isdigit():
                input_data = single_input
            else:
                input_data = ""
        else:
            uploaded_file = st.file_uploader("Chọn file TXT/CSV", type=['txt', 'csv'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                input_data = content
            else:
                input_data = ""
        
        if st.button("🚀 KÍCH HOẠT PHÂN TÍCH", use_container_width=True):
            if input_data:
                parsed_data = parse_input_data(input_data)
                if parsed_data:
                    st.session_state.raw_data = parsed_data
                    st.success(f"✅ Đã nạp {len(parsed_data)} số hợp lệ!")
                    st.rerun()
                else:
                    st.error("❌ Không tìm thấy số hợp lệ. Vui lòng kiểm tra định dạng!")
            else:
                st.warning("⚠️ Vui lòng nhập dữ liệu trước!")
        
        # Display loaded data
        if st.session_state.raw_data:
            st.markdown("### 📋 DỮ LIỆU ĐÃ NẠP")
            data_df = pd.DataFrame({
                'STT': range(1, len(st.session_state.raw_data) + 1),
                'SỐ': st.session_state.raw_data
            })
            st.dataframe(data_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Xóa dữ liệu", type="secondary"):
                st.session_state.raw_data = []
                st.rerun()
    
    with col_right:
        # Analysis Section
        if st.session_state.raw_data:
            # Run analysis
            analyzer = EliteAnalyzerV3(st.session_state.raw_data)
            analysis_results = analyzer.analyze_all()
            
            if analysis_results:
                # Find top number
                sorted_results = sorted(
                    analysis_results.items(),
                    key=lambda x: x[1]['final_score'],
                    reverse=True
                )
                
                top_digit, top_info = sorted_results[0]
                
                # Main Prediction Card
                st.markdown(f"""
                    <div class='elite-card'>
                        <div style='text-align: center;'>
                            <div style='color: #00FFC2; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>
                                ⭐ DỰ ĐOÁN ƯU TIÊN HÀNG ĐẦU
                            </div>
                            <div class='main-number'>{top_digit}</div>
                            <div style='margin: 20px 0;'>
                                <div style='color: #FFFFFF; font-size: 18px; margin-bottom: 5px;'>
                                    ĐỘ TIN CẬY: <span style='color: #FF3131; font-weight: bold;'>{top_info['final_score']:.1f}%</span>
                                </div>
                                <div style='color: {top_info['signal_color']}; font-size: 16px; font-weight: bold;'>
                                    {top_info['signal']}
                                </div>
                            </div>
                            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 10px;'>
                                <div style='font-size: 14px; color: #AAAAAA;'>
                                    📊 Dữ liệu: {len(st.session_state.raw_data)} số | 
                                    ⏱️ Trễ: {top_info['gap_current']} kỳ |
                                    📈 Tần suất: {top_info['frequency_total']:.1f}%
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed Analysis Table
                st.markdown("### 📊 BẢNG PHÂN TÍCH CHI TIẾT")
                
                table_data = []
                for digit, info in sorted_results:
                    table_data.append({
                        'SỐ': digit,
                        'ĐIỂM': f"{info['final_score']:.1f}%",
                        'TRỄ': info['gap_current'],
                        'TẦN SUẤT': f"{info['frequency_total']:.1f}%",
                        'XU HƯỚNG': f"{info['trend_short']:.1f}%",
                        'TÍN HIỆU': info['signal']
                    })
                
                df_table = pd.DataFrame(table_data)
                st.dataframe(df_table, use_container_width=True, hide_index=True)
                
                # Visualization
                st.markdown("### 📈 BIỂU ĐỒ PHÂN TÍCH")
                fig = create_visualization(analysis_results, top_digit)
                st.plotly_chart(fig, use_container_width=True)
                
                # Verification System
                st.markdown("---")
                st.markdown("### 🎯 KIỂM TRA KẾT QUẢ")
                
                col_verify1, col_verify2 = st.columns([2, 1])
                with col_verify1:
                    actual_result = st.text_input(
                        "Nhập kết quả thực tế:",
                        placeholder="Ví dụ: 12345",
                        key="verification_input"
                    )
                
                with col_verify2:
                    if st.button("✅ XÁC NHẬN", use_container_width=True):
                        if actual_result and len(actual_result) == 5 and actual_result.isdigit():
                            is_correct = top_digit in actual_result
                            update_statistics(is_correct)
                            
                            # Add to history
                            st.session_state.history.insert(0, {
                                'Thời gian': datetime.now().strftime("%H:%M %d/%m"),
                                'Dự đoán': top_digit,
                                'Kết quả': actual_result,
                                'Trạng thái': '✅ ĐÚNG' if is_correct else '❌ SAI',
                                'Điểm số': f"{top_info['final_score']:.1f}%"
                            })
                            
                            # Keep only last 10 records
                            if len(st.session_state.history) > 10:
                                st.session_state.history = st.session_state.history[:10]
                            
                            st.rerun()
                        else:
                            st.error("Vui lòng nhập số 5 chữ số hợp lệ!")
                
                # Prediction History
                if st.session_state.history:
                    st.markdown("### 📜 LỊCH SỬ KIỂM TRA")
                    history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        else:
            # Welcome/Instructions Card
            st.markdown("""
                <div class='elite-card'>
                    <div style='text-align: center; padding: 40px 20px;'>
                        <div style='font-size: 48px; margin-bottom: 20px;'>💎</div>
                        <h3 style='color: #00FFC2; margin-bottom: 15px;'>
                            LOTOBET ELITE v3.0
                        </h3>
                        <p style='color: #AAAAAA; line-height: 1.6;'>
                            Hệ thống phân tích 7 tầng thuật toán tiên tiến nhất.<br>
                            Nhập dữ liệu ở cột bên trái để bắt đầu phân tích.
                        </p>
                        <div style='margin-top: 30px; padding: 15px; background: rgba(0, 255, 194, 0.1); border-radius: 10px;'>
                            <p style='color: #00FFC2; font-size: 14px; margin: 0;'>
                                🎯 <strong>Cách sử dụng:</strong><br>
                                1. Nạp dữ liệu kết quả trước đó<br>
                                2. Hệ thống tự động phân tích<br>
                                3. Nhận dự đoán số ưu tiên<br>
                                4. Kiểm tra và cập nhật kết quả
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666666; font-size: 12px; padding: 20px;'>
            <p>LOTOBET ELITE v3.0 | 7-Layer Algorithm System | Optimized for Mobile</p>
            <p>⚠️ Hệ thống phân tích hỗ trợ quyết định, không đảm bảo kết quả 100%</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
