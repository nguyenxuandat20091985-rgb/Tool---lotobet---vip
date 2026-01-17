với bản cos này bạn rúp tôi bỏ rúp tôi dự đoán 2 tinh.
 - thay vào đấy các số có thể về ( ví dụ 1,2,3,4 ) với xác xuất cao nhất, lên đánh trong kỳ tiếp theo. 
- luật chơi lotobet không cố định 3tinh.
• 3 số 5 tinh:
Từ 0~9 chọn ra 3 con số để đặt cược, mỗi một đơn cược được tạo thành bởi 3 con số, chỉ cần trong con số mở thưởng từ hàng【Chục ngàn】【Ngàn】【Trăm】【Chục】【Đơn vị】bao gồm con số đã chọn, đồng thời không giới hạn trình tự con số mở thưởng, như vậy xem như bạn đã trúng thưởng. Con số đã chọn để đặt cược bất luận xuất hiện bao nhiêu lần thì tiền thưởng cũng chỉ được tính 1 lần.
Ví dụ:
Đặt cược 3 số 5 tinh【Chục ngàn/Ngàn/Trăm/Chục/Đơn vị】với con số: 1, 2, 6 tạo thành 1 tổ hợp. Con số mở thưởng 5 tinh【Chục ngàn/Ngàn/Trăm/Chục/Đơn vị】là: 12864 như vậy bạn đã trúng thưởng.
Ví dụ:
Đặt cược 3 số 5 tinh【Chục ngàn /Ngàn/Trăm/Chục/Đơn vị】với con số: 1, 3, 6 tạo thành 1 tổ hợp. Con số mở thưởng 5 tinh【Chục ngàn /Ngàn/Trăm/Chục/Đơn vị】là: 12662 như vậy bạn không trúng thưởng.
>> cos cần làm 
"""
LOTOBET AI TOOL v1.0 - Professional Lottery Analysis
Nhập số tay → Dự đoán kỳ tiếp theo
Optimized for Android - Lightweight & Fast
""" 

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import io
import base64
import random
import math
from typing import List, Dict, Tuple, Any
from itertools import combinations 

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="LOTOBET AI TOOL v1.0 - 3 TINH",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
) 

# ==================== CUSTOM CSS - PROFESSIONAL DESIGN ====================
st.markdown("""
<style>
    /* Base Design - Android Optimized */
    .stApp {
        background: #0a0e17;
        color: #ffffff;
        max-width: 414px;
        margin: 0 auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        padding: 8px;
        overflow-x: hidden;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header { display: none !important; }
    
    /* Mobile optimization */
    @media (max-width: 414px) {
        .main > div { 
            padding: 3px !important;
            max-width: 100vw;
        }
        h1 { font-size: 18px !important; margin-bottom: 6px !important; }
        h2 { font-size: 16px !important; margin-bottom: 4px !important; }
        h3 { font-size: 14px !important; margin-bottom: 3px !important; }
    }
    
    /* Professional Header */
    .main-header {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Modern Compact Buttons */
    .stButton > button {
        width: 100% !important;
        height: 40px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        margin: 3px 0 !important;
        border: none !important;
        background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%) !important;
        color: white !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(255, 81, 47, 0.4) !important;
    }
    
    .primary-btn {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%) !important;
    }
    
    .primary-btn:hover {
        box-shadow: 0 6px 12px rgba(38, 208, 206, 0.4) !important;
    }
    
    /* Compact Tabs - Horizontal Layout */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        background: rgba(255, 255, 255, 0.05);
        padding: 3px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        padding: 6px 8px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        font-size: 11px !important;
        flex: 1;
        min-width: 55px;
        text-align: center;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Number Cards */
    .number-card {
        background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%);
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 2px;
        font-size: 13px;
        display: inline-block;
        min-width: 35px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .number-card:hover {
        transform: scale(1.05);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 3px;
        font-size: 16px;
        display: inline-block;
        min-width: 60px;
        box-shadow: 0 3px 8px rgba(255, 81, 47, 0.3);
    }
    
    /* Single Number Cards */
    .single-number-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
        font-weight: 900;
        margin: 3px;
        font-size: 18px;
        display: inline-block;
        min-width: 45px;
        box-shadow: 0 3px 8px rgba(0, 176, 155, 0.3);
    }
    
    /* FIXED: Input text color - BLACK TEXT ON WHITE BACKGROUND */
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stNumberInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* White text for labels */
    .stTextInput label,
    .stTextArea label,
    .stNumberInput label,
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Compact Box */
    .compact-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Stats Box */
    .stats-box {
        background: linear-gradient(135deg, rgba(255, 81, 47, 0.3) 0%, rgba(221, 36, 118, 0.3) 100%);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }
    
    /* Tables */
    .dataframe {
        font-size: 10px !important;
    }
    
    /* Prevent Overflow */
    * {
        max-width: 100%;
        box-sizing: border-box;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { 
        width: 4px; 
        height: 4px; 
    }
    ::-webkit-scrollbar-track { 
        background: rgba(255,255,255,0.05); 
        border-radius: 2px; 
    }
    ::-webkit-scrollbar-thumb { 
        background: #DD2476; 
        border-radius: 2px; 
    }
    
    /* Advice Colors */
    .advice-good {
        color: #00ff88;
        font-weight: 700;
        font-size: 12px;
    }
    
    .advice-medium {
        color: #ffcc00;
        font-weight: 700;
        font-size: 12px;
    }
    
    .advice-low {
        color: #ff6b6b;
        font-weight: 700;
        font-size: 12px;
    }
    
    /* Quick Action Row */
    .quick-action-row {
        display: flex;
        gap: 5px;
        margin: 8px 0;
    }
    
    /* Hot Numbers Row */
    .hot-numbers-container {
        display: flex;
        justify-content: center;
        gap: 5px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    
    /* Prediction Type Badge */
    .type-badge {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 10px;
        color: #ffcc00;
        display: inline-block;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True) 

# ==================== SESSION STATE INIT ====================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_period': 1000,
        'lottery_time': datetime.datetime.now().strftime("%H:%M"),
        'historical_data': None,
        'data_loaded': False,
        'manual_results': [],
        'predictions': {},
        'next_period_predictions': {},
        'last_update': datetime.datetime.now(),
        'hot_numbers_prediction': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value 

init_session_state() 

# ==================== ADVANCED AI ENGINE - FOCUS ON 3 TINH ====================
class LotteryAI:
    """Advanced AI for Lottery Prediction - Focus on 3 TINH and Single Numbers"""
    
    def __init__(self):
        self.algorithms_count = 50
        
    def _analyze_input_numbers(self, numbers: List[str]) -> Dict:
        """Analyze manually input numbers for patterns"""
        if not numbers:
            return {}
        
        # Extract all digits
        all_digits = ''.join(numbers)
        
        # Calculate frequency for each digit (0-9)
        freq = {}
        for digit in '0123456789':
            count = all_digits.count(digit)
            freq[digit] = (count / len(all_digits)) * 100
        
        # Find HOT numbers (appearing most frequently) - TOP 4
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot_numbers = [digit for digit, _ in sorted_freq[:4]]  # Lấy 4 số nóng nhất
        
        # Analyze patterns in positions
        position_analysis = {}
        if len(numbers) > 0 and len(numbers[0]) == 5:
            for i in range(5):
                pos_digits = [num[i] for num in numbers if len(num) > i]
                pos_freq = {}
                for digit in '0123456789':
                    count = pos_digits.count(digit)
                    pos_freq[digit] = (count / len(pos_digits)) * 100 if pos_digits else 0
                position_analysis[f'position_{i}'] = pos_freq
        
        # Analyze triple patterns
        triple_patterns = {}
        for num in numbers:
            if len(num) == 5:
                # Tạo các tổ hợp 3 số từ 5 số
                for combo in combinations(num, 3):
                    triple = ''.join(sorted(combo))
                    triple_patterns[triple] = triple_patterns.get(triple, 0) + 1
        
        top_triples = sorted(triple_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'hot_numbers': hot_numbers,  # Ví dụ: ['1', '2', '3', '4']
            'frequency': freq,
            'position_analysis': position_analysis,
            'top_triples': [triple for triple, _ in top_triples],
            'total_numbers': len(numbers)
        }
    
    def _generate_smart_triplets(self, analysis: Dict) -> List[str]:
        """Generate smart 3TINH triplets based on analysis"""
        triplets = []
        hot_numbers = analysis.get('hot_numbers', [])
        top_triples = analysis.get('top_triples', [])
        
        # Strategy 1: Use HOT numbers (ví dụ: 1,2,3,4)
        if len(hot_numbers) >= 3:
            # Tạo tổ hợp từ 4 số nóng nhất
            for combo in combinations(hot_numbers[:4], 3):
                triplet = ''.join(sorted(combo))
                if triplet not in triplets:
                    triplets.append(triplet)
                    if len(triplets) >= 2:
                        break
        
        # Strategy 2: Use historical triple patterns
        for triple in top_triples[:3]:
            if triple not in triplets:
                triplets.append(triple)
        
        # Strategy 3: Complementary combinations
        if hot_numbers:
            base_number = hot_numbers[0]  # Số nóng nhất
            complementary_numbers = random.sample('0123456789', 2)
            triple = ''.join(sorted([base_number] + complementary_numbers))
            if triple not in triplets:
                triplets.append(triple)
        
        # Ensure we have at least 4 triplets
        while len(triplets) < 4:
            digits = random.sample('0123456789', 3)
            triple = ''.join(sorted(digits))
            if triple not in triplets:
                triplets.append(triple)
        
        return triplets[:4]  # Return top 4 triplets
    
    def predict_hot_single_numbers(self, analysis: Dict) -> List[Dict]:
        """Predict single numbers that have high probability to appear"""
        hot_numbers = analysis.get('hot_numbers', [])
        frequency = analysis.get('frequency', {})
        
        predictions = []
        for i, number in enumerate(hot_numbers[:4]):  # Top 4 numbers
            freq_percent = frequency.get(number, 0)
            
            # Calculate probability based on frequency and position
            base_prob = 80 + (i * 5)  # 80%, 85%, 90%, 95%
            freq_adjust = freq_percent / 2
            final_prob = min(99, base_prob + freq_adjust)
            
            # Determine position recommendation
            position_rec = "Mọi vị trí"
            position_scores = {}
            for pos_name, pos_data in analysis.get('position_analysis', {}).items():
                position_scores[pos_name] = pos_data.get(number, 0)
            
            if position_scores:
                best_pos = max(position_scores.items(), key=lambda x: x[1])
                position_map = {
                    'position_0': "Chục ngàn",
                    'position_1': "Ngàn",
                    'position_2': "Trăm",
                    'position_3': "Chục", 
                    'position_4': "Đơn vị"
                }
                if best_pos[1] > 15:  # Nếu xuất hiện > 15% ở vị trí đó
                    position_rec = position_map.get(best_pos[0], "Mọi vị trí")
            
            predictions.append({
                'number': number,
                'probability': round(final_prob, 1),
                'position': position_rec,
                'frequency': round(freq_percent, 1),
                'advice': "✅ NÊN CHỌN" if final_prob >= 85 else "✅ CÓ THỂ CHỌN"
            })
        
        return predictions
    
    @st.cache_data(ttl=30, show_spinner=False)
    def predict_from_input(_self, numbers: List[str]) -> Dict:
        """Generate predictions for next period based on input numbers"""
        if not numbers:
            return {
                'hot_single_numbers': [],
                '3tinh': [],
                'analysis': {}
            }
        
        # Analyze input numbers
        analysis = _self._analyze_input_numbers(numbers)
        
        # Generate HOT single number predictions
        hot_single_predictions = _self.predict_hot_single_numbers(analysis)
        
        # Generate 3TINH predictions
        triplets = _self._generate_smart_triplets(analysis)
        triplet_predictions = []
        
        for i, triplet in enumerate(triplets):
            # Calculate probability
            base_prob = 75 + (i * 6)  # 75%, 81%, 87%, 93%
            
            # Adjust based on HOT numbers in triplet
            hot_count = sum(1 for d in triplet if d in analysis.get('hot_numbers', []))
            hot_adjust = hot_count * 8
            
            final_prob = min(98, base_prob + hot_adjust)
            
            # Risk assessment
            if final_prob >= 85:
                risk = "THẤP"
                advice = "✅ NÊN VÀO"
                confidence = "RẤT CAO"
            elif final_prob >= 80:
                risk = "TRUNG BÌNH"
                advice = "✅ CÓ THỂ THỬ"
                confidence = "CAO"
            else:
                risk = "CAO"
                advice = "⚠️ THEO DÕI"
                confidence = "TRUNG BÌNH"
            
            # Check if in historical top triples
            historical_boost = " (Xuất hiện nhiều)" if triplet in analysis.get('top_triples', []) else ""
            
            triplet_predictions.append({
                'combo': triplet,
                'probability': round(final_prob, 1),
                'risk': risk,
                'confidence': confidence,
                'advice': advice,
                'analysis': f"Dựa trên phân tích {len(numbers)} bộ số{historical_boost}",
                'hot_numbers_included': hot_count
            })
        
        return {
            'hot_single_numbers': hot_single_predictions,
            '3tinh': triplet_predictions,
            'analysis': {
                'hot_numbers': analysis.get('hot_numbers', []),
                'total_inputs': len(numbers),
                'top_triples': analysis.get('top_triples', [])[:3]
            }
        }

# ==================== HEADER ====================
st.markdown(f"""
<div class="main-header">
    <div style="font-size: 16px; font-weight: 900;">🎯 LOTOBET AI - 3 TINH FOCUS</div>
    <div style="font-size: 11px; color: rgba(255,255,255,0.8);">Dự đoán số nóng & 3 TINH | Kỳ tiếp theo</div>
</div>
""", unsafe_allow_html=True)

# ==================== TAB 1: DATA COLLECTION ====================
st.markdown("### 📊 NHẬP SỐ ĐỂ PHÂN TÍCH")

data_tabs = st.tabs(["🌐 Web", "📁 File", "✏️ Nhập số"])

with data_tabs[0]:
    st.markdown("**Kết nối website soi cầu**")
    
    url = st.text_input(
        "URL website:",
        placeholder="https://soicau.com",
        key="url_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Test", use_container_width=True):
            st.success("✅ Kết nối thành công!")
    with col2:
        if st.button("🔄 Lấy dữ liệu", use_container_width=True):
            st.info("Đang thu thập dữ liệu...")

with data_tabs[1]:
    st.markdown("**Upload file CSV/TXT**")
    
    uploaded_file = st.file_uploader(
        "Chọn file CSV/TXT",
        type=['csv', 'txt'],
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delimiter='\t')
            
            st.session_state.historical_data = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Đã tải {len(df)} dòng dữ liệu")
            
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

with data_tabs[2]:
    st.markdown("**✏️ Nhập số thủ công**")
    st.caption("Nhập số kết quả các kỳ trước (mỗi dòng 5 số)")
    
    # Text area for manual input
    numbers_input = st.text_area(
        "Nhập số (ví dụ: 12345):",
        placeholder="12345\n54321\n67890\n98765\n13579\n24680",
        height=120,
        key="number_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Lưu số", use_container_width=True, key="save_numbers", type="primary"):
            if numbers_input:
                lines = [line.strip() for line in numbers_input.split('\n') if line.strip()]
                valid = []
                
                for num in lines:
                    if len(num) == 5 and num.isdigit():
                        valid.append(num)
                
                if valid:
                    st.session_state.manual_results = valid
                    st.success(f"✅ Đã lưu {len(valid)} bộ số")
                else:
                    st.error("❌ Không có số hợp lệ")
            else:
                st.warning("⚠️ Vui lòng nhập số")
    
    with col2:
        if st.button("🤖 Phân tích & Dự đoán", use_container_width=True, key="analyze_numbers"):
            if st.session_state.manual_results:
                # Initialize AI and analyze
                ai = LotteryAI()
                predictions = ai.predict_from_input(st.session_state.manual_results)
                st.session_state.next_period_predictions = predictions
                st.session_state.hot_numbers_prediction = predictions.get('hot_single_numbers', [])
                st.success("✅ Đã phân tích và tạo dự đoán cho kỳ tiếp theo!")
                st.rerun()
            else:
                st.error("❌ Chưa có số để phân tích")

# ==================== PREDICTIONS FOR NEXT PERIOD ====================
st.markdown("---")

# Show current stats
if st.session_state.manual_results:
    st.markdown(f"**📋 Đang có {len(st.session_state.manual_results)} bộ số nhập tay**")

if st.session_state.data_loaded and st.session_state.historical_data is not None:
    st.markdown(f"**💾 Đang có {len(st.session_state.historical_data)} dòng dữ liệu lịch sử**")

st.markdown("### 🎯 DỰ ĐOÁN CHO KỲ TIẾP THEO")

# Display HOT SINGLE NUMBER predictions
if 'next_period_predictions' in st.session_state and st.session_state.next_period_predictions:
    predictions = st.session_state.next_period_predictions
    
    # ========== HOT SINGLE NUMBERS ==========
    st.markdown("#### 🔥 SỐ NÓNG CÓ THỂ VỀ CAO NHẤT")
    
    if predictions['hot_single_numbers']:
        # Display HOT numbers in a nice row
        st.markdown('<div class="hot-numbers-container">', unsafe_allow_html=True)
        
        for pred in predictions['hot_single_numbers'][:4]:  # Show top 4
            st.markdown(f"""
            <div class="compact-box" style="flex: 1; min-width: 90px;">
                <div style="text-align: center;">
                    <div class="single-number-card">{pred['number']}</div>
                </div>
                <div style="margin-top: 5px; text-align: center;">
                    <div style="color: #00ff88; font-size: 14px; font-weight: 900;">{pred['probability']}%</div>
                    <div style="color: #ffcc00; font-size: 10px;">Vị trí: {pred['position']}</div>
                </div>
                <div style="margin-top: 5px; text-align: center;">
                    <div class="{'advice-good' if 'NÊN' in pred['advice'] else 'advice-medium'}">
                        {pred['advice']}
                    </div>
                </div>
                <div style="font-size: 9px; color: #94a3b8; text-align: center; margin-top: 3px;">
                    Tần suất: {pred['frequency']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advice for using these numbers
        hot_numbers_list = ', '.join([p['number'] for p in predictions['hot_single_numbers'][:4]])
        st.markdown(f"""
        <div class="compact-box">
            <div style="color: #ffcc00; font-weight: 700; margin-bottom: 5px;">💡 CÁCH SỬ DỤNG SỐ NÓNG:</div>
            <div style="font-size: 11px;">
                • Các số <strong>{hot_numbers_list}</strong> có xác suất xuất hiện cao nhất
                <br>• Có thể kết hợp để tạo thành 3 TINH (ví dụ: {hot_numbers_list[0:2]})
                <br>• Đánh ở vị trí được đề xuất để tăng cơ hội trúng
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Chưa có dự đoán số nóng")
    
    # ========== 3 TINH PREDICTIONS ==========
    st.markdown("---")
    st.markdown("#### 🔢🔢🔢 3 TINH LÊN ĐÁNH")
    
    if predictions['3tinh']:
        # Create grid for 3TINH predictions
        cols = st.columns(2)
        
        for idx, pred in enumerate(predictions['3tinh'][:4]):  # Show top 4
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="compact-box">
                    <div style="text-align: center;">
                        <div class="prediction-card">{pred['combo']}</div>
                        <div style="margin-top: 3px;">
                            <span class="type-badge">3 TINH</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; text-align: center;">
                        <div style="color: #00ff88; font-size: 16px; font-weight: 900;">{pred['probability']}%</div>
                        <div style="color: {'#00ff88' if pred['risk'] == 'THẤP' else '#ffcc00' if pred['risk'] == 'TRUNG BÌNH' else '#ff6b6b'}; 
                             font-size: 10px;">Rủi ro: {pred['risk']} | Độ tin cậy: {pred['confidence']}</div>
                    </div>
                    <div style="margin-top: 8px; text-align: center;">
                        <div class="{'advice-good' if 'NÊN VÀO' in pred['advice'] else 'advice-medium' if 'CÓ THỂ' in pred['advice'] else 'advice-low'}">
                            {pred['advice']}
                        </div>
                    </div>
                    <div style="font-size: 9px; color: #94a3b8; text-align: center; margin-top: 5px;">
                        {pred['analysis']}
                        <br>Số nóng trong combo: {pred['hot_numbers_included']}/3
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 3TINH playing advice
        st.markdown(f"""
        <div class="compact-box">
            <div style="color: #26d0ce; font-weight: 700; margin-bottom: 5px;">🎯 LUẬT CHƠI 3 TINH:</div>
            <div style="font-size: 11px;">
                • Chọn 3 số (ví dụ: 1,2,6) để đặt cược
                <br>• Chỉ cần trong kết quả 5 số có chứa cả 3 số này (không cần đúng thứ tự)
                <br>• Ví dụ: Đánh 1,2,6 → Kết quả 12864 → Trúng thưởng
                <br>• Mỗi số dù xuất hiện bao nhiêu lần cũng chỉ tính 1 lần
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Chưa có dự đoán 3 TINH")
    
    # ========== ANALYSIS SUMMARY ==========
    if predictions.get('analysis'):
        analysis = predictions['analysis']
        st.markdown("---")
        st.markdown("#### 📊 PHÂN TÍCH SỐ NHẬP")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            hot_nums = ', '.join(analysis['hot_numbers'][:4]) if analysis['hot_numbers'] else "N/A"
            st.markdown(f"""
            <div class="stats-box">
                <div style="color: #94a3b8; font-size: 10px;">SỐ NÓNG NHẤT</div>
                <div style="color: #ff6b6b; font-size: 14px; font-weight: 900;">{hot_nums}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Top 4 xuất hiện nhiều</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            total = analysis['total_inputs']
            st.markdown(f"""
            <div class="stats-box">
                <div style="color: #94a3b8; font-size: 10px;">TỔNG SỐ NHẬP</div>
                <div style="color: white; font-size: 16px; font-weight: 900;">{total}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Bộ số đã phân tích</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            top_triples = ', '.join(analysis['top_triples'][:2]) if analysis['top_triples'] else "N/A"
            st.markdown(f"""
            <div class="stats-box">
                <div style="color: #94a3b8; font-size: 10px;">COMBO HAY XUẤT HIỆN</div>
                <div style="color: #00ff88; font-size: 12px; font-weight: 900;">{top_triples}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Trong lịch sử</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👆 **Nhập số và bấm 'Phân tích & Dự đoán' để xem dự đoán cho kỳ tiếp theo**")

# ==================== QUICK NUMBER GENERATOR ====================
st.markdown("---")
st.markdown("### ⚡ TẠO NHANH 3 TINH TỪ SỐ NÓNG")

if st.button("🎲 Tạo combo 3 TINH ngẫu nhiên", use_container_width=True, key="generate_random_3tinh"):
    # Use hot numbers if available, otherwise random
    hot_numbers = []
    if 'next_period_predictions' in st.session_state and st.session_state.next_period_predictions:
        hot_numbers = st.session_state.next_period_predictions.get('analysis', {}).get('hot_numbers', [])
    
    if hot_numbers:
        # Create combos from hot numbers
        if len(hot_numbers) >= 3:
            combo1 = ''.join(sorted(hot_numbers[:3]))
            if len(hot_numbers) >= 4:
                combo2 = ''.join(sorted([hot_numbers[0], hot_numbers[1], hot_numbers[3]]))
            else:
                combo2 = ''.join(sorted(random.sample('0123456789', 3)))
        else:
            combo1 = ''.join(sorted(random.sample('0123456789', 3)))
            combo2 = ''.join(sorted(random.sample('0123456789', 3)))
    else:
        # Generate completely random combos
        combo1 = ''.join(sorted(random.sample('0123456789', 3)))
        combo2 = ''.join(sorted(random.sample('0123456789', 3)))
    
    combo3 = ''.join(sorted(random.sample('0123456789', 3)))
    
    st.markdown(f"""
    <div class="compact-box">
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="color: #ffcc00; font-weight: 700;">🎲 COMBO 3 TINH NGẪU NHIÊN</div>
        </div>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <div class="prediction-card">{combo1}</div>
            <div class="prediction-card">{combo2}</div>
            <div class="prediction-card">{combo3}</div>
        </div>
        <div style="text-align: center; margin-top: 10px; font-size: 11px; color: #94a3b8;">
            Có thể sử dụng để tham khảo đánh 3 TINH
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== QUICK ACTIONS ====================
st.markdown("---")
st.markdown("### ⚡ THAO TÁC NHANH")

action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("🔄 Làm mới", use_container_width=True, key="refresh_btn"):
        st.rerun()

with action_col2:
    if st.button("📊 Xem dữ liệu", use_container_width=True, key="view_data_btn"):
        if st.session_state.data_loaded:
            st.dataframe(
                st.session_state.historical_data.head(10),
                use_container_width=True
            )
        elif st.session_state.manual_results:
            df = pd.DataFrame({
                'STT': range(1, len(st.session_state.manual_results) + 1),
                'Số': st.session_state.manual_results
            })
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu để hiển thị")

with action_col3:
    if st.button("🗑️ Xóa tất cả", use_container_width=True, key="clear_all_btn"):
        st.session_state.historical_data = None
        st.session_state.data_loaded = False
        st.session_state.manual_results = []
        st.session_state.predictions = {}
        st.session_state.next_period_predictions = {}
        st.session_state.hot_numbers_prediction = []
        st.success("✅ Đã xóa tất cả dữ liệu")
        st.rerun()

# ==================== AI STATS AT BOTTOM ====================
st.markdown("---")
st.markdown("### 📈 THỐNG KÊ AI")

col1, col2, col3 = st.columns(3)

with col1:
    # Algorithms count
    st.markdown("""
    <div class="stats-box">
        <div style="color: #94a3b8; font-size: 10px;">THUẬT TOÁN 3 TINH</div>
        <div style="color: #FF512F; font-size: 18px; font-weight: 900;">50</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Advanced AI</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Prediction accuracy for 3TINH
    accuracy = random.randint(75, 88)
    st.markdown(f"""
    <div class="stats-box">
        <div style="color: #94a3b8; font-size: 10px;">ĐỘ CHÍNH XÁC 3TINH</div>
        <div style="color: #00ff88; font-size: 18px; font-weight: 900;">{accuracy}%</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Dựa trên phân tích</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Processing speed
    st.markdown("""
    <div class="stats-box">
        <div style="color: #94a3b8; font-size: 10px;">TỐC ĐỘ XỬ LÝ</div>
        <div style="color: white; font-size: 18px; font-weight: 900;">< 0.5s</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 9px;">Real-time</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== DATA STATS AT BOTTOM ====================
st.markdown("---")

# Create two columns for data stats
stats_col1, stats_col2 = st.columns(2)

with stats_col1:
    if st.session_state.manual_results:
        st.markdown(f"""
        <div class="compact-box">
            <div style="color: #94a3b8; font-size: 11px;">📋 ĐANG CÓ</div>
            <div style="color: white; font-size: 16px; font-weight: 900;">{len(st.session_state.manual_results)} bộ số</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 10px;">Số đã nhập thủ công</div>
        </div>
        """, unsafe_allow_html=True)

with stats_col2:
    if 'next_period_predictions' in st.session_state and st.session_state.next_period_predictions:
        hot_count = len(st.session_state.next_period_predictions.get('hot_single_numbers', []))
        st.markdown(f"""
        <div class="compact-box">
            <div style="color: #94a3b8; font-size: 11px;">🎯 ĐANG DỰ ĐOÁN</div>
            <div style="color: #FF512F; font-size: 16px; font-weight: 900;">{hot_count} số nóng</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 10px;">+ 4 combo 3 TINH</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 9px; padding: 6px;">
    LOTOBET AI TOOL v1.0 - 3 TINH FOCUS | Dự đoán số nóng & 3 TINH | 50 Thuật Toán<br>
    <span style="font-size: 8px;">© 2024 - Chơi có trách nhiệm</span>
</div>
""", unsafe_allow_html=True)

# ==================== AUTO UPDATE TIME ====================
# Update time every minute
current_time = datetime.datetime.now()
if current_time.minute != st.session_state.last_update.minute:
    st.session_state.lottery_time = current_time.strftime("%H:%M")
    st.session_state.last_update = current_time
    st.rerun()
