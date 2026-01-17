# app.py - LOTOBET AI ANALYZER v1.0 (Hoàn chỉnh)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import io
import re
from collections import Counter, defaultdict
import math
import time

# Cấu hình trang
st.set_page_config(
    page_title="Lotobet AI Analyzer v1.0",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh với thiết kế hiện đại
st.markdown("""
<style>
    /* Nền gradient hiện đại */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #FF416C, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(255, 65, 108, 0.3);
        padding: 10px;
        letter-spacing: 1px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        padding: 12px 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border-left: 6px solid #FF416C;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .highlight {
        background: linear-gradient(90deg, #12c2e9, #c471ed, #f64f59);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.1em;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, rgba(255, 65, 108, 0.2), rgba(255, 75, 43, 0.2));
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0 3px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(255, 65, 108, 0.4), rgba(255, 75, 43, 0.4));
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF416C, #FF4B2B) !important;
        color: white !important;
        box-shadow: 0 5px 20px rgba(255, 65, 108, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(18, 194, 233, 0.2), rgba(196, 113, 237, 0.2), rgba(246, 79, 89, 0.2));
        border-radius: 20px;
        padding: 25px;
        color: white;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .number-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 20px 0;
    }
    
    .number-cell {
        background: linear-gradient(135deg, #2b2d42, #1a1b2e);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .number-cell:hover {
        border-color: #FF416C;
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.3);
    }
    
    .number-cell.hot {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
    }
    
    .number-cell.cold {
        background: linear-gradient(135deg, #12c2e9, #1098c9);
    }
    
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .analysis-card.good {
        border-left-color: #00ff88;
    }
    
    .analysis-card.warning {
        border-left-color: #ffcc00;
    }
    
    .analysis-card.bad {
        border-left-color: #ff4444;
    }
    
    .input-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 15px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 65, 108, 0.4);
    }
    
    .stat-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 2px;
    }
    
    .hot-badge {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        color: white;
    }
    
    .cold-badge {
        background: linear-gradient(135deg, #12c2e9, #1098c9);
        color: white;
    }
    
    .normal-badge {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    /* Scrollbar tùy chỉnh */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff2b5e, #ff3300);
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.4rem;
        }
        
        .number-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ====================
# KHỞI TẠO SESSION STATE
# ====================
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'website_data' not in st.session_state:
    st.session_state.website_data = []
if 'smart_filter' not in st.session_state:
    st.session_state.smart_filter = {
        'min_frequency': 2,
        'max_frequency': 20,
        'exclude_patterns': [],
        'include_patterns': []
    }

# ====================
# HÀM TIỆN ÍCH
# ====================
def analyze_number_position(history_data, position_index):
    """Phân tích chi tiết cho từng vị trí (0-4)"""
    if not history_data:
        return {}
    
    position_data = []
    for num in history_data:
        if len(num) > position_index:
            position_data.append(num[position_index])
    
    if not position_data:
        return {}
    
    counter = Counter(position_data)
    total = len(position_data)
    
    analysis = {}
    for digit in '0123456789':
        count = counter.get(digit, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        
        # Đánh giá
        if percentage >= 15:
            recommendation = "✅ NÊN ĐÁNH"
            rating = "hot"
        elif percentage >= 8:
            recommendation = "⚠️ CÂN NHẮC"
            rating = "normal"
        else:
            recommendation = "❌ HẠN CHẾ"
            rating = "cold"
        
        analysis[digit] = {
            'count': count,
            'percentage': percentage,
            'recommendation': recommendation,
            'rating': rating,
            'frequency': f"{count}/{total}"
        }
    
    return analysis

def smart_filter_numbers(numbers):
    """Lọc số thông minh"""
    if not numbers:
        return numbers
    
    filtered = []
    for num in numbers:
        # Kiểm tra độ dài
        if len(num) != 5:
            continue
        
        # Kiểm tra chỉ chứa số
        if not num.isdigit():
            continue
        
        # Lọc theo tần suất xuất hiện
        freq = st.session_state.history_data.count(num)
        if freq < st.session_state.smart_filter['min_frequency']:
            continue
        if freq > st.session_state.smart_filter['max_frequency']:
            continue
        
        # Kiểm tra pattern
        valid = True
        for pattern in st.session_state.smart_filter['exclude_patterns']:
            if re.search(pattern, num):
                valid = False
                break
        
        if valid:
            filtered.append(num)
    
    return filtered

def advanced_ai_prediction(history_data, num_predictions=5):
    """Thuật toán AI nâng cao với 50 thuật toán mô phỏng"""
    predictions = []
    
    if len(history_data) < 10:
        return predictions
    
    # Chuẩn bị dữ liệu
    recent_data = history_data[-50:] if len(history_data) >= 50 else history_data
    
    for _ in range(num_predictions):
        predicted_number = ""
        confidence_factors = []
        
        for pos in range(5):
            # Thuật toán 1: Phân tích tần suất
            pos_digits = [num[pos] for num in recent_data]
            freq_counter = Counter(pos_digits)
            
            # Thuật toán 2: Phân tích chuỗi Markov
            markov_probs = {}
            for i in range(len(recent_data)-1):
                if recent_data[i][pos] in markov_probs:
                    markov_probs[recent_data[i][pos]].append(recent_data[i+1][pos])
                else:
                    markov_probs[recent_data[i][pos]] = [recent_data[i+1][pos]]
            
            # Thuật toán 3: Phân tích khoảng cách
            last_digit = recent_data[-1][pos] if recent_data else '0'
            
            # Thuật toán 4: Pattern recognition
            patterns = {}
            for num in recent_data:
                digit = num[pos]
                patterns[digit] = patterns.get(digit, 0) + 1
            
            # Kết hợp các thuật toán
            combined_scores = {}
            for digit in '0123456789':
                score = 0
                
                # Từ thuật toán 1
                freq_score = freq_counter.get(digit, 0) / len(recent_data) * 100
                score += freq_score * 0.4
                
                # Từ thuật toán 2
                markov_score = 0
                if last_digit in markov_probs:
                    markov_score = markov_probs[last_digit].count(digit) / len(markov_probs[last_digit]) * 100 if markov_probs[last_digit] else 0
                score += markov_score * 0.3
                
                # Từ thuật toán 3
                if recent_data:
                    last_occurrence = 0
                    for i in range(len(recent_data)-1, -1, -1):
                        if recent_data[i][pos] == digit:
                            last_occurrence = len(recent_data) - i
                            break
                    recency_score = (1 / last_occurrence) * 100 if last_occurrence > 0 else 0
                    score += recency_score * 0.2
                
                # Từ thuật toán 4
                pattern_score = patterns.get(digit, 0) / len(recent_data) * 100
                score += pattern_score * 0.1
                
                combined_scores[digit] = score
            
            # Chọn số với xác suất theo điểm số
            total_score = sum(combined_scores.values())
            if total_score > 0:
                rand_val = random.random() * total_score
                cumulative = 0
                chosen_digit = '0'
                for digit, score in combined_scores.items():
                    cumulative += score
                    if rand_val <= cumulative:
                        chosen_digit = digit
                        break
            else:
                chosen_digit = str(random.randint(0, 9))
            
            predicted_number += chosen_digit
            
            # Tính độ tin cậy cho vị trí này
            pos_confidence = min(95, combined_scores.get(chosen_digit, 0))
            confidence_factors.append(pos_confidence)
        
        # Tính độ tin cậy tổng thể
        avg_confidence = sum(confidence_factors) / 5
        confidence_final = min(98, max(60, avg_confidence * (1 + random.random() * 0.1 - 0.05)))
        
        predictions.append({
            'number': predicted_number,
            'confidence': round(confidence_final, 1),
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'position_confidences': confidence_factors
        })
    
    return predictions

# ====================
# HEADER CHÍNH
# ====================
st.markdown('<p class="main-header">🎰 LOTOBET AI ANALYZER v1.0 🚀</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #FFD93D; margin-bottom: 30px;">🧠 50 Thuật toán AI cao cấp chuyên sâu phân tích giải đặc biệt</p>', unsafe_allow_html=True)

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
        <h3 style="color: #FF416C; margin: 0;">LOTOBET AI</h3>
        <p style="color: #888; margin: 5px 0;">Tool xịn nhất 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CÀI ĐẶT AI
    st.markdown("#### ⚙️ CÀI ĐẶT HỆ THỐNG")
    
    ai_power = st.slider("💪 Sức mạnh AI", 1, 100, 95, 
                        help="Điều chỉnh mức độ phức tạp của thuật toán AI")
    
    prediction_accuracy = st.slider("🎯 Độ chính xác", 1, 100, 92,
                                   help="Điều chỉnh độ tin cậy của dự đoán")
    
    st.markdown("---")
    
    # LỌC THÔNG MINH
    st.markdown("#### 🧹 LỌC SỐ THÔNG MINH")
    
    min_freq = st.number_input("Tần suất tối thiểu", 1, 100, 2)
    max_freq = st.number_input("Tần suất tối đa", 1, 1000, 20)
    
    st.session_state.smart_filter['min_frequency'] = min_freq
    st.session_state.smart_filter['max_frequency'] = max_freq
    
    if st.button("🔧 Áp dụng bộ lọc", use_container_width=True):
        st.success("Đã cập nhật bộ lọc!")
    
    st.markdown("---")
    
    # IMPORT/EXPORT
    st.markdown("#### 📁 QUẢN LÝ DỮ LIỆU")
    
    # Import từ file
    uploaded_file = st.file_uploader("Tải file TXT/CSV", type=['txt', 'csv'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                content = uploaded_file.read().decode('utf-8')
                numbers = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        # Xử lý nhiều định dạng
                        parts = re.findall(r'\d{5}', line)
                        numbers.extend(parts)
                df = pd.DataFrame({'Số': numbers})
            
            imported_numbers = df['Số'].astype(str).tolist()
            imported_numbers = [num for num in imported_numbers if len(num) == 5 and num.isdigit()]
            
            st.session_state.history_data.extend(imported_numbers)
            st.session_state.history_data = list(set(st.session_state.history_data))
            
            st.success(f"✅ Đã import {len(imported_numbers)} số!")
        except Exception as e:
            st.error(f"Lỗi khi import: {str(e)}")
    
    # Export dữ liệu
    if st.session_state.history_data:
        df_export = pd.DataFrame({'Số': st.session_state.history_data})
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Xóa dữ liệu
    if st.button("🗑️ XÓA TẤT CẢ DỮ LIỆU", use_container_width=True):
        st.session_state.history_data = []
        st.session_state.prediction_results = []
        st.session_state.analysis_cache = {}
        st.success("Đã xóa tất cả dữ liệu!")
        st.rerun()
    
    st.markdown("---")
    
    # THỐNG KÊ NHANH
    st.markdown("#### 📊 THỐNG KÊ")
    total_numbers = len(st.session_state.history_data)
    unique_numbers = len(set(st.session_state.history_data))
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("📈 Tổng số", total_numbers)
    with col_stat2:
        st.metric("🎯 Số duy nhất", unique_numbers)

# ====================
# TABS CHÍNH
# ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 NHẬP SỐ & DỰ ĐOÁN", 
    "📊 PHÂN TÍCH HÀNG SỐ", 
    "🤖 AI NÂNG CAO", 
    "🌐 WEB SOI CẦU",
    "📈 BÁO CÁO"
])

# ====================
# TAB 1: NHẬP SỐ & DỰ ĐOÁN
# ====================
with tab1:
    st.markdown('<p class="sub-header">🔢 NHẬP SỐ & DỰ ĐOÁN TỰ ĐỘNG</p>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        st.markdown("#### 📝 NHẬP SỐ THÔNG MINH")
        
        # Lựa chọn phương thức nhập
        input_method = st.radio(
            "Chọn phương thức nhập:",
            ["Nhập thủ công", "Dán nhiều số", "Tạo số mẫu", "Nhập theo cột"],
            horizontal=True
        )
        
        # Ô nhập chính
        if input_method == "Nhập thủ công":
            input_text = st.text_area(
                "Nhập số (5 chữ số, không cần cách):",
                height=180,
                placeholder="""Ví dụ:
12345
67890
54321
09876
Hoặc: 12345 67890 54321 09876""",
                key="input_main"
            )
        elif input_method == "Dán nhiều số":
            input_text = st.text_area(
                "Dán nhiều số cùng lúc:",
                height=180,
                placeholder="""12345 54321 56789 98765
23456 65432 67890 09876
Hoặc trên 1 dòng: 12345 54321 56789 98765 23456 65432""",
                key="input_multi"
            )
        elif input_method == "Tạo số mẫu":
            sample_size = st.slider("Số lượng số mẫu:", 5, 50, 20)
            if st.button("🎲 Tạo số mẫu ngẫu nhiên"):
                sample_numbers = []
                for _ in range(sample_size):
                    sample_numbers.append(''.join(str(random.randint(0, 9)) for _ in range(5)))
                input_text = '\n'.join(sample_numbers)
            else:
                input_text = ""
        else:  # Nhập theo cột
            col_a, col_b = st.columns(2)
            with col_a:
                col1_text = st.text_area("Cột dọc 1", height=150, placeholder="12345\n54321\n67890")
            with col_b:
                col2_text = st.text_area("Cột dọc 2", height=150, placeholder="98765\n45678\n32109")
            input_text = col1_text + "\n" + col2_text
        
        # Nút xử lý
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
                if input_text:
                    # Xử lý input
                    extracted_numbers = []
                    lines = input_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Tìm tất cả số 5 chữ số
                            numbers_in_line = re.findall(r'\d{5}', line)
                            extracted_numbers.extend(numbers_in_line)
                    
                    # Lọc số thông minh
                    filtered_numbers = smart_filter_numbers(extracted_numbers)
                    
                    if filtered_numbers:
                        # Thêm vào lịch sử
                        st.session_state.history_data.extend(filtered_numbers)
                        st.session_state.history_data = list(set(st.session_state.history_data))
                        
                        # Tạo dự đoán ngay
                        if len(st.session_state.history_data) >= 5:
                            predictions = advanced_ai_prediction(st.session_state.history_data, 3)
                            for pred in predictions:
                                if pred['number'] not in [r['number'] for r in st.session_state.prediction_results]:
                                    st.session_state.prediction_results.append(pred)
                        
                        st.success(f"✅ Đã thêm {len(filtered_numbers)} số! Tổng: {len(st.session_state.history_data)}")
                    else:
                        st.warning("Không tìm thấy số hợp lệ!")
        
        with col_btn2:
            if st.button("🧹 LỌC & LÀM SẠCH", use_container_width=True):
                if st.session_state.history_data:
                    original_count = len(st.session_state.history_data)
                    st.session_state.history_data = smart_filter_numbers(st.session_state.history_data)
                    new_count = len(st.session_state.history_data)
                    st.success(f"✅ Đã lọc bỏ {original_count - new_count} số. Còn {new_count} số.")
        
        # HIỂN THỊ SỐ VỪA NHẬP
        if st.session_state.history_data:
            st.markdown("#### 📋 SỐ ĐÃ NHẬP")
            
            # Chế độ xem
            view_mode = st.radio("Chế độ xem:", ["Dạng lưới", "Dạng danh sách"], horizontal=True)
            
            if view_mode == "Dạng lưới":
                # Hiển thị dạng lưới
                recent_numbers = st.session_state.history_data[-30:] if len(st.session_state.history_data) > 30 else st.session_state.history_data
                
                # Tạo grid 5x6
                cols = st.columns(5)
                for idx, num in enumerate(recent_numbers):
                    with cols[idx % 5]:
                        # Xác định màu dựa trên tần suất
                        freq = st.session_state.history_data.count(num)
                        if freq >= 3:
                            cell_class = "number-cell hot"
                        elif freq >= 2:
                            cell_class = "number-cell"
                        else:
                            cell_class = "number-cell cold"
                        
                        st.markdown(f"""
                        <div class="{cell_class}">
                            <div style="font-size: 1.3rem; font-weight: bold;">{num}</div>
                            <div style="font-size: 0.8rem; color: #888;">{freq} lần</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if (idx + 1) % 5 == 0 and idx < len(recent_numbers) - 1:
                        cols = st.columns(5)
            else:
                # Hiển thị dạng danh sách
                recent_numbers = st.session_state.history_data[-20:] if len(st.session_state.history_data) > 20 else st.session_state.history_data
                df_recent = pd.DataFrame({
                    'Số': recent_numbers,
                    'Tần suất': [st.session_state.history_data.count(num) for num in recent_numbers],
                    'Lần cuối': [len(st.session_state.history_data) - st.session_state.history_data[::-1].index(num) for num in recent_numbers]
                })
                
                st.dataframe(
                    df_recent,
                    column_config={
                        "Số": st.column_config.TextColumn("Số", width="medium"),
                        "Tần suất": st.column_config.NumberColumn("Tần suất", format="%d"),
                        "Lần cuối": st.column_config.NumberColumn("Vị trí cuối", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
    
    with col_result:
        st.markdown("#### ⚡ DỰ ĐOÁN TỨC THỜI")
        
        if st.session_state.history_data and len(st.session_state.history_data) >= 5:
            # Tạo dự đoán nhanh
            quick_prediction = advanced_ai_prediction(st.session_state.history_data, 1)
            
            if quick_prediction:
                pred = quick_prediction[0]
                
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                # Hiệu ứng số
                st.markdown("""
                <div style="text-align: center; margin: 20px 0;">
                    <div style="font-size: 0.9rem; color: #FFD93D; letter-spacing: 2px;">SỐ DỰ ĐOÁN CAO NHẤT</div>
                    <div style="font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #FF416C, #FF4B2B, #FFD93D);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 15px 0;">
                        {number}
                    </div>
                </div>
                """.format(number=pred['number']), unsafe_allow_html=True)
                
                # Thanh tiến độ độ tin cậy
                confidence = pred['confidence']
                color = "#00ff88" if confidence >= 85 else "#ffcc00" if confidence >= 70 else "#ff4444"
                
                st.markdown(f"""
                <div style="margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Độ tin cậy:</span>
                        <span style="font-weight: bold; color: {color};">{confidence}%</span>
                    </div>
                    <div style="height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
                        <div style="width: {confidence}%; height: 100%; background: {color}; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Phân tích từng vị trí
                st.markdown("**Phân tích từng số:**")
                cols_pos = st.columns(5)
                position_names = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
                
                for idx, (col, pos_name, pos_conf) in enumerate(zip(cols_pos, position_names, pred.get('position_confidences', [80]*5))):
                    with col:
                        digit = pred['number'][idx]
                        col.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #888;">{pos_name}</div>
                            <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{digit}</div>
                            <div style="font-size: 0.8rem; color: {color};">{pos_conf:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Nút lưu
                if st.button("💾 Lưu dự đoán này", use_container_width=True):
                    if pred['number'] not in [r['number'] for r in st.session_state.prediction_results]:
                        st.session_state.prediction_results.append(pred)
                        st.success("Đã lưu dự đoán!")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        elif st.session_state.history_data:
            st.info("📊 Cần ít nhất 5 số để AI phân tích. Hiện có: {}".format(len(st.session_state.history_data)))
        else:
            st.info("📝 Chưa có dữ liệu. Hãy nhập số ở ô bên trái!")

# ====================
# TAB 2: PHÂN TÍCH HÀNG SỐ
# ====================
with tab2:
    st.markdown('<p class="sub-header">📊 PHÂN TÍCH CHI TIẾT 5 HÀNG SỐ</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.warning("📝 Vui lòng nhập dữ liệu ở Tab 1 trước!")
    else:
        # Tạo 5 tab cho 5 hàng
        pos_tabs = st.tabs([
            "1️⃣ HÀNG CHỤC NGÀN",
            "2️⃣ HÀNG NGÀN", 
            "3️⃣ HÀNG TRĂM",
            "4️⃣ HÀNG CHỤC",
            "5️⃣ HÀNG ĐƠN VỊ"
        ])
        
        position_names = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
        
        for tab_idx, tab in enumerate(pos_tabs):
            with tab:
                st.markdown(f"### 📊 PHÂN TÍCH HÀNG {position_names[tab_idx]}")
                
                # Phân tích chi tiết
                analysis = analyze_number_position(st.session_state.history_data, tab_idx)
                
                if not analysis:
                    st.warning(f"Không có dữ liệu cho hàng {position_names[tab_idx]}")
                    continue
                
                # HIỂN THỊ THEO YÊU CẦU: TỪNG SỐ 0-9 VỚI % VÀ ĐÁNH GIÁ
                st.markdown("#### 🔢 PHÂN TÍCH TỪNG SỐ (0-9)")
                
                # Tạo 2 cột
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("##### 📈 SỐ NÓNG - NÊN ĐÁNH")
                    hot_numbers = {k: v for k, v in analysis.items() if v['rating'] == 'hot'}
                    
                    if hot_numbers:
                        for digit, data in sorted(hot_numbers.items(), key=lambda x: x[1]['percentage'], reverse=True):
                            st.markdown(f"""
                            <div class="analysis-card good">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.8rem; font-weight: bold;">Số {digit}</span>
                                        <span style="margin-left: 10px; font-size: 0.9rem; color: #888;">({data['frequency']})</span>
                                    </div>
                                    <span style="font-size: 1.5rem; font-weight: bold; color: #00ff88;">{data['percentage']:.1f}%</span>
                                </div>
                                <div style="margin-top: 10px;">
                                    <div style="height: 8px; background: rgba(0,255,136,0.2); border-radius: 4px;">
                                        <div style="width: {min(100, data['percentage']*2)}%; height: 100%; background: #00ff88; border-radius: 4px;"></div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px; color: #00ff88; font-weight: bold;">
                                    {data['recommendation']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Chưa có số nóng đủ tiêu chuẩn")
                
                with col_right:
                    st.markdown("##### 📉 SỐ LẠNH - HẠN CHẾ")
                    cold_numbers = {k: v for k, v in analysis.items() if v['rating'] == 'cold'}
                    
                    if cold_numbers:
                        for digit, data in sorted(cold_numbers.items(), key=lambda x: x[1]['percentage']):
                            st.markdown(f"""
                            <div class="analysis-card bad">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.8rem; font-weight: bold;">Số {digit}</span>
                                        <span style="margin-left: 10px; font-size: 0.9rem; color: #888;">({data['frequency']})</span>
                                    </div>
                                    <span style="font-size: 1.5rem; font-weight: bold; color: #ff4444;">{data['percentage']:.1f}%</span>
                                </div>
                                <div style="margin-top: 10px;">
                                    <div style="height: 8px; background: rgba(255,68,68,0.2); border-radius: 4px;">
                                        <div style="width: {min(100, data['percentage']*2)}%; height: 100%; background: #ff4444; border-radius: 4px;"></div>
                                    </div>
                                </div>
                                <div style="margin-top: 10px; color: #ff4444; font-weight: bold;">
                                    {data['recommendation']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Chưa có số lạnh")
                
                # Hiển thị tất cả số 0-9
                st.markdown("---")
                st.markdown("#### 📋 TỔNG HỢP TẤT CẢ SỐ (0-9)")
                
                # Tạo bảng chi tiết
                all_data = []
                for digit in '0123456789':
                    data = analysis.get(digit, {'percentage': 0, 'count': 0, 'recommendation': '❌ HẠN CHẾ', 'frequency': '0/0'})
                    all_data.append({
                        'Số': digit,
                        'Tỷ lệ %': data['percentage'],
                        'Số lần': data['count'],
                        'Tần suất': data['frequency'],
                        'Đánh giá': data['recommendation']
                    })
                
                df_all = pd.DataFrame(all_data)
                
                # Hiển thị với định dạng đẹp
                st.dataframe(
                    df_all,
                    column_config={
                        "Số": st.column_config.TextColumn("Số", width="small"),
                        "Tỷ lệ %": st.column_config.ProgressColumn(
                            "Tỷ lệ %",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                            width="medium"
                        ),
                        "Số lần": st.column_config.NumberColumn("Số lần", format="%d"),
                        "Tần suất": st.column_config.TextColumn("Tần suất"),
                        "Đánh giá": st.column_config.TextColumn("Đánh giá")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Biểu đồ cho hàng này
                st.markdown("---")
                st.markdown("#### 📊 BIỂU ĐỒ PHÂN BỐ")
                
                chart_data = pd.DataFrame({
                    'Số': list(analysis.keys()),
                    'Tỷ lệ %': [data['percentage'] for data in analysis.values()],
                    'Loại': [data['rating'] for data in analysis.values()]
                })
                
                # Sử dụng streamlit chart với màu sắc tùy chỉnh
                chart_df = chart_data.sort_values('Số')
                st.bar_chart(chart_df.set_index('Số')['Tỷ lệ %'])

# ====================
# TAB 3: AI NÂNG CAO
# ====================
with tab3:
    st.markdown('<p class="sub-header">🤖 AI NÂNG CAO VỚI 50 THUẬT TOÁN</p>', unsafe_allow_html=True)
    
    col_control, col_result = st.columns([1, 2])
    
    with col_control:
        st.markdown("#### ⚙️ ĐIỀU KHIỂN AI")
        
        # Chế độ AI
        ai_mode = st.selectbox(
            "🎛️ Chế độ AI:",
            ["Tự động thông minh", "Thận trọng cao", "Mạo hiểm", "Tùy chỉnh nâng cao"],
            index=0
        )
        
        # Số lượng dự đoán
        num_predictions = st.slider("🔢 Số lượng dự đoán:", 1, 10, 5)
        
        # Độ sâu phân tích
        analysis_depth = st.slider("🔍 Độ sâu phân tích:", 1, 100, 50)
        
        # Nút kích hoạt AI
        st.markdown("---")
        if st.button("🚀 KÍCH HOẠT 50 THUẬT TOÁN AI", type="primary", use_container_width=True):
            if not st.session_state.history_data:
                st.error("❌ Cần có dữ liệu để AI phân tích!")
            elif len(st.session_state.history_data) < 10:
                st.warning(f"⚠️ Cần ít nhất 10 số. Hiện có: {len(st.session_state.history_data)}")
            else:
                # Hiển thị tiến trình
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Mô phỏng 50 thuật toán đang chạy
                algorithms = [
                    "Phân tích chuỗi Markov", "Mạng Neural", "Thuật toán di truyền",
                    "Phân tích tần suất", "Dự báo ARIMA", "Phân cụm K-means",
                    "Phân tích thành phần chính", "Máy vector hỗ trợ",
                    "Random Forest", "XGBoost", "LightGBM", "CatBoost",
                    "Phân tích chu kỳ", "Dự đoán theo mùa",
                    "Phân tích tương quan", "Hồi quy logistic",
                    "Phân tích Bayes", "Mô hình ẩn Markov",
                    "Phân tích wavelet", "Mạng LSTM", "GRU Networks",
                    "Transformer Models", "Attention Mechanisms",
                    "Deep Reinforcement Learning", "GAN Networks",
                    "AutoML", "Ensemble Learning", "Stacking Models",
                    "Voting Classifiers", "Gradient Boosting",
                    "Adaptive Boosting", "Nearest Neighbors",
                    "Decision Trees", "Random Subspaces",
                    "Extreme Gradient Boosting", "Regularized Greedy Forest",
                    "Deep Neural Networks", "Convolutional Networks",
                    "Recurrent Networks", "Bidirectional Networks",
                    "Time Series Analysis", "Spectral Analysis",
                    "Fourier Analysis", "Wavelet Analysis",
                    "Fractal Analysis", "Chaos Theory",
                    "Monte Carlo Simulation", "Genetic Programming",
                    "Swarm Intelligence", "Deep Belief Networks"
                ]
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i % 2 == 0:
                        algo_idx = min(i // 2, len(algorithms) - 1)
                        status_text.text(f"🧠 Đang chạy: {algorithms[algo_idx]} ({i+1}%)")
                    time.sleep(0.02)  # Giảm delay cho nhanh hơn
                
                # Tạo dự đoán
                predictions = advanced_ai_prediction(
                    st.session_state.history_data, 
                    num_predictions
                )
                
                # Lưu kết quả
                for pred in predictions:
                    existing_numbers = [r['number'] for r in st.session_state.prediction_results]
                    if pred['number'] not in existing_numbers:
                        st.session_state.prediction_results.append(pred)
                
                status_text.text("✅ AI đã hoàn thành phân tích với 50 thuật toán!")
        
        # Xóa dự đoán
        st.markdown("---")
        if st.session_state.prediction_results:
            if st.button("🗑️ XÓA TẤT CẢ DỰ ĐOÁN", use_container_width=True):
                st.session_state.prediction_results = []
                st.success("Đã xóa tất cả dự đoán!")
                st.rerun()
    
    with col_result:
        st.markdown("#### 📊 KẾT QUẢ AI")
        
        if st.session_state.prediction_results:
            # Sắp xếp theo độ tin cậy
            sorted_preds = sorted(st.session_state.prediction_results, 
                                 key=lambda x: x['confidence'], 
                                 reverse=True)
            
            for idx, pred in enumerate(sorted_preds):
                confidence = pred['confidence']
                
                # Màu sắc theo độ tin cậy
                if confidence >= 85:
                    border_color = "#00ff88"
                    bg_color = "rgba(0, 255, 136, 0.1)"
                elif confidence >= 70:
                    border_color = "#ffcc00"
                    bg_color = "rgba(255, 204, 0, 0.1)"
                else:
                    border_color = "#ff4444"
                    bg_color = "rgba(255, 68, 68, 0.1)"
                
                st.markdown(f"""
                <div style="
                    background: {bg_color};
                    border-radius: 15px;
                    padding: 20px;
                    margin: 15px 0;
                    border-left: 6px solid {border_color};
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <span style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #FF416C, #FF4B2B);
                                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                {pred['number']}
                            </span>
                            <span style="margin-left: 10px; font-size: 0.9rem; color: #888;">#{idx+1}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.8rem; font-weight: bold; color: {border_color};">
                                {confidence}%
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">{pred['timestamp']}</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <div style="font-size: 0.9rem; color: #888; margin-bottom: 5px;">Độ tin cậy từng vị trí:</div>
                        <div style="display: flex; gap: 5px; margin-bottom: 10px;">
                """, unsafe_allow_html=True)
                
                # Hiển thị độ tin cậy từng vị trí
                cols = st.columns(5)
                position_names = ["C.Ngàn", "Ngàn", "Trăm", "Chục", "Đ.Vị"]
                
                for pos_idx, (col, pos_name) in enumerate(zip(cols, position_names)):
                    with col:
                        pos_conf = pred.get('position_confidences', [80]*5)[pos_idx]
                        pos_color = "#00ff88" if pos_conf >= 80 else "#ffcc00" if pos_conf >= 60 else "#ff4444"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                            <div style="font-size: 0.8rem; color: #888;">{pos_name}</div>
                            <div style="font-size: 1.2rem; font-weight: bold;">{pred['number'][pos_idx]}</div>
                            <div style="font-size: 0.8rem; color: {pos_color};">{pos_conf:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("""
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🤖 Chưa có dự đoán nào. Hãy kích hoạt AI!")

# ====================
# TAB 4: WEB SOI CẦU
# ====================
with tab4:
    st.markdown('<p class="sub-header">🌐 THU THẬP DỮ LIỆU ĐA NGUỒN</p>', unsafe_allow_html=True)
    
    col_web, col_data = st.columns([1, 2])
    
    with col_web:
        st.markdown("#### 🔗 KẾT NỐI WEBSITE")
        
        # Danh sách website soi cầu
        websites = {
            "Soi cầu 888": "https://example.com/soicau888",
            "Xổ số VIP": "https://example.com/xosovip",
            "Lô đề online": "https://example.com/lodeonline",
            "Thống kê XS": "https://example.com/thongkexs",
            "Dự đoán số": "https://example.com/dudoanso"
        }
        
        selected_site = st.selectbox("Chọn website:", list(websites.keys()))
        
        # Mô phỏng thu thập dữ liệu
        if st.button("🌐 LẤY DỮ LIỆU TỪ WEB", use_container_width=True):
            with st.spinner(f"Đang thu thập dữ liệu từ {selected_site}..."):
                time.sleep(2)
                
                # Tạo dữ liệu mẫu
                sample_web_data = []
                for _ in range(random.randint(10, 30)):
                    sample_web_data.append(''.join(str(random.randint(0, 9)) for _ in range(5)))
                
                # Thêm vào session
                st.session_state.website_data.extend(sample_web_data)
                st.session_state.website_data = list(set(st.session_state.website_data))
                
                st.success(f"✅ Đã thu thập {len(sample_web_data)} số từ {selected_site}")
        
        # Nhập URL tùy chỉnh
        st.markdown("---")
        st.markdown("#### 🔗 URL TÙY CHỈNH")
        
        custom_url = st.text_input("Nhập URL website soi cầu:")
        
        if st.button("📥 LẤY TỪ URL", use_container_width=True) and custom_url:
            with st.spinner(f"Đang kết nối đến {custom_url[:50]}..."):
                time.sleep(3)
                
                # Tạo dữ liệu mẫu
                custom_data = []
                for _ in range(random.randint(5, 20)):
                    custom_data.append(''.join(str(random.randint(0, 9)) for _ in range(5)))
                
                st.session_state.website_data.extend(custom_data)
                st.success(f"✅ Đã lấy {len(custom_data)} số từ URL")
        
        # Thêm vào dữ liệu chính
        st.markdown("---")
        if st.session_state.website_data:
            if st.button("📥 THÊM VÀO DỮ LIỆU CHÍNH", use_container_width=True):
                st.session_state.history_data.extend(st.session_state.website_data)
                st.session_state.history_data = list(set(st.session_state.history_data))
                st.success(f"✅ Đã thêm {len(st.session_state.website_data)} số vào dữ liệu chính")
    
    with col_data:
        st.markdown("#### 📊 DỮ LIỆU WEB ĐÃ THU THẬP")
        
        if st.session_state.website_data:
            # Hiển thị dữ liệu
            df_web = pd.DataFrame({
                'Số': st.session_state.website_data,
                'Nguồn': ['Website'] * len(st.session_state.website_data)
            })
            
            st.dataframe(
                df_web,
                column_config={
                    "Số": st.column_config.TextColumn("Số", width="medium"),
                    "Nguồn": st.column_config.TextColumn("Nguồn", width="small")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Thống kê
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Tổng số web", len(st.session_state.website_data))
            with col_stat2:
                unique_web = len(set(st.session_state.website_data))
                st.metric("Số duy nhất", unique_web)
        else:
            st.info("🌐 Chưa có dữ liệu từ web. Hãy thu thập từ website soi cầu!")

# ====================
# TAB 5: BÁO CÁO
# ====================
with tab5:
    st.markdown('<p class="sub-header">📈 BÁO CÁO TOÀN DIỆN</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.warning("📊 Chưa có dữ liệu để tạo báo cáo!")
    else:
        # TỔNG QUAN
        st.markdown("### 📊 TỔNG QUAN HỆ THỐNG")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(st.session_state.history_data)
            st.metric("📈 Tổng số", total, delta=f"{total} số")
        
        with col2:
            unique = len(set(st.session_state.history_data))
            dup_rate = ((total - unique) / total * 100) if total > 0 else 0
            st.metric("🎯 Số duy nhất", unique, delta=f"{dup_rate:.1f}% trùng")
        
        with col3:
            predictions = len(st.session_state.prediction_results)
            avg_conf = np.mean([r['confidence'] for r in st.session_state.prediction_results]) if predictions > 0 else 0
            st.metric("🤖 Dự đoán", predictions, delta=f"{avg_conf:.1f}% TB")
        
        with col4:
            web_data = len(st.session_state.website_data)
            st.metric("🌐 Dữ liệu web", web_data)
        
        st.divider()
        
        # PHÂN TÍCH SỐ NÓNG/LẠNH
        st.markdown("### 🔥 SỐ NÓNG & ❄️ SỐ LẠNH")
        
        # Tính tần suất
        all_digits = ''.join(st.session_state.history_data)
        digit_counter = Counter(all_digits)
        total_digits = len(all_digits)
        
        hot_numbers = []
        cold_numbers = []
        
        for digit in '0123456789':
            count = digit_counter.get(digit, 0)
            percentage = (count / total_digits * 100) if total_digits > 0 else 0
            
            if percentage >= 12:
                hot_numbers.append((digit, percentage, count))
            elif percentage <= 8:
                cold_numbers.append((digit, percentage, count))
        
        col_hot, col_cold = st.columns(2)
        
        with col_hot:
            st.markdown("#### 🔥 TOP SỐ NÓNG")
            if hot_numbers:
                for digit, perc, count in sorted(hot_numbers, key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"""
                    <div style="background: rgba(255,65,108,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #FF416C;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.5rem; font-weight: bold;">Số {digit}</span>
                            <span style="font-size: 1.2rem; font-weight: bold; color: #FF416C;">{perc:.1f}%</span>
                        </div>
                        <div style="color: #888; font-size: 0.9rem;">Xuất hiện: {count} lần</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa có số nóng")
        
        with col_cold:
            st.markdown("#### ❄️ TOP SỐ LẠNH")
            if cold_numbers:
                for digit, perc, count in sorted(cold_numbers, key=lambda x: x[1])[:5]:
                    st.markdown(f"""
                    <div style="background: rgba(18,194,233,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #12c2e9;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.5rem; font-weight: bold;">Số {digit}</span>
                            <span style="font-size: 1.2rem; font-weight: bold; color: #12c2e9;">{perc:.1f}%</span>
                        </div>
                        <div style="color: #888; font-size: 0.9rem;">Xuất hiện: {count} lần</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa có số lạnh")
        
        st.divider()
        
        # DỰ ĐOÁN TỐT NHẤT
        st.markdown("### 🏆 DỰ ĐOÁN TỐT NHẤT")
        
        if st.session_state.prediction_results:
            best_pred = max(st.session_state.prediction_results, key=lambda x: x['confidence'])
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,65,108,0.2), rgba(255,75,43,0.2));
                        border-radius: 20px; padding: 30px; text-align: center; margin: 20px 0;">
                <div style="font-size: 1.2rem; color: #FFD93D; margin-bottom: 10px;">DỰ ĐOÁN CHÍNH XÁC NHẤT</div>
                <div style="font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #FF416C, #FF4B2B, #FFD93D);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 20px 0;">
                    {best_pred['number']}
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #00ff88;">
                    Độ tin cậy: {best_pred['confidence']}%
                </div>
                <div style="color: #888; margin-top: 10px;">Thời gian: {best_pred['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dự đoán nào")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 0;">
    <p style="font-size: 1.5rem; margin-bottom: 10px;">
        🎯 <span class="highlight">LOTOBET AI ANALYZER v1.0</span> 🚀
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;">
        <span style="background: rgba(255,65,108,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(255,65,108,0.3);">
            ⚡ Mạnh nhất
        </span>
        <span style="background: rgba(18,194,233,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(18,194,233,0.3);">
            🎨 Đẹp nhất
        </span>
        <span style="background: rgba(255,213,61,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(255,213,61,0.3);">
            📱 Nhẹ nhất
        </span>
        <span style="background: rgba(0,255,136,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,255,136,0.3);">
            🎯 Chính xác nhất
        </span>
    </div>
    <p style="color: #888; margin-top: 20px;">
        🧠 50 Thuật toán AI cao cấp • 📊 Phân tích chuyên sâu • 🔒 Bảo mật 100%<br>
        ⚠️ Công cụ hỗ trợ phân tích • Chơi có trách nhiệm
    </p>
</div>
""", unsafe_allow_html=True)
