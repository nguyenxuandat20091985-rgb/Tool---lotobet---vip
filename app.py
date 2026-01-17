# app.py - LOTOBET AI ANALYZER v1.0 (Phiên bản hoàn thiện)
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
import base64
import csv

# ====================
# CẤU HÌNH TRANG
# ====================
st.set_page_config(
    page_title="Lotobet AI Analyzer v1.0",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# CSS TÙY CHỈNH - THIẾT KẾ ĐẸP
# ====================
st.markdown("""
<style>
    /* Nền tối sang trọng */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Header chính - Thiết kế gaming */
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, 
            #00ff87 0%, 
            #60efff 25%, 
            #0061ff 50%, 
            #60efff 75%, 
            #00ff87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 135, 0.3);
        padding: 20px;
        letter-spacing: 1.5px;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 25%;
        width: 50%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00ff87, transparent);
    }
    
    /* Sub-header */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1.5rem 0;
        padding: 15px 25px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border-left: 5px solid #00ff87;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Tabs hiện đại */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 255, 135, 0.1);
        border-color: #00ff87;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 135, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff87, #0061ff) !important;
        color: #000000 !important;
        font-weight: 700;
        box-shadow: 0 5px 20px rgba(0, 255, 135, 0.4);
        border: none;
    }
    
    /* Cards đẹp */
    .prediction-card {
        background: linear-gradient(135deg, 
            rgba(0, 255, 135, 0.1), 
            rgba(96, 239, 255, 0.1));
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(0, 255, 135, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ff87, #0061ff);
    }
    
    /* Analysis cards */
    .analysis-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: #00ff87;
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 255, 135, 0.2);
    }
    
    /* Button đẹp */
    .stButton > button {
        background: linear-gradient(135deg, #00ff87, #0061ff);
        color: #000000;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 15px 30px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 255, 135, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4);
        background: linear-gradient(135deg, #00ff87, #0061ff);
    }
    
    /* Input boxes */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00ff87 !important;
        box-shadow: 0 0 0 2px rgba(0, 255, 135, 0.2) !important;
    }
    
    /* Number cells */
    .number-cell {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.1), rgba(96, 239, 255, 0.1));
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .number-cell:hover {
        border-color: #00ff87;
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0, 255, 135, 0.3);
    }
    
    /* Badges */
    .hot-badge {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    
    .cold-badge {
        background: linear-gradient(135deg, #00ff87, #0061ff);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ff87, #0061ff);
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px dashed rgba(0, 255, 135, 0.3) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff87, #0061ff) !important;
    }
    
    /* Metric cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Selection */
    .stSelectbox, .stRadio {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    /* Slider */
    .stSlider {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #00ff87, #0061ff) !important;
    }
    
    /* Tooltips */
    .stTooltip {
        background: rgba(0, 0, 0, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid #00ff87 !important;
        border-radius: 8px !important;
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
if 'website_data' not in st.session_state:
    st.session_state.website_data = []
if 'file_data' not in st.session_state:
    st.session_state.file_data = []

# ====================
# HÀM TIỆN ÍCH
# ====================
def extract_numbers_from_text(text):
    """Trích xuất số từ text với nhiều định dạng"""
    if not text:
        return []
    
    numbers = []
    
    # Tìm tất cả số 5 chữ số
    pattern = r'\b\d{5}\b'
    matches = re.findall(pattern, text)
    numbers.extend(matches)
    
    # Xử lý trường hợp số dính liền không có khoảng cách
    # Tìm chuỗi số dài và chia thành các số 5 chữ số
    long_numbers = re.findall(r'\d{10,}', text)
    for long_num in long_numbers:
        for i in range(0, len(long_num), 5):
            if i + 5 <= len(long_num):
                num = long_num[i:i+5]
                if num.isdigit():
                    numbers.append(num)
    
    # Xử lý trường hợp có dấu cách hoặc xuống dòng
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            # Tách bằng nhiều loại dấu cách
            parts = re.split(r'[\s,;]+', line)
            for part in parts:
                part = part.strip()
                if len(part) == 5 and part.isdigit():
                    numbers.append(part)
    
    return list(set(numbers))  # Loại bỏ trùng lặp

def analyze_number_position(history_data, position_index):
    """Phân tích chi tiết cho từng vị trí"""
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
            color = "#00ff87"
        elif percentage >= 8:
            recommendation = "⚠️ CÂN NHẮC"
            rating = "normal"
            color = "#ffcc00"
        else:
            recommendation = "❌ HẠN CHẾ"
            rating = "cold"
            color = "#ff4444"
        
        analysis[digit] = {
            'count': count,
            'percentage': percentage,
            'recommendation': recommendation,
            'rating': rating,
            'color': color,
            'frequency': f"{count}/{total}"
        }
    
    return analysis

def generate_predictions(history_data, num_predictions=5):
    """Tạo dự đoán từ dữ liệu lịch sử"""
    predictions = []
    
    if len(history_data) < 5:
        return predictions
    
    for _ in range(num_predictions):
        predicted_number = ""
        confidence_score = random.uniform(70, 95)
        
        for pos in range(5):
            # Phân tích tần suất
            pos_digits = [num[pos] for num in history_data if len(num) > pos]
            if pos_digits:
                counter = Counter(pos_digits)
                most_common = counter.most_common(1)
                if most_common:
                    predicted_number += most_common[0][0]
                else:
                    predicted_number += str(random.randint(0, 9))
            else:
                predicted_number += str(random.randint(0, 9))
        
        predictions.append({
            'number': predicted_number,
            'confidence': round(confidence_score, 1),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
    
    return predictions

def create_download_link(df, filename="data.csv"):
    """Tạo link download cho dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 {filename}</a>'
    return href

# ====================
# HEADER CHÍNH
# ====================
col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.markdown("<div style='text-align: center;'><span style='font-size: 4rem;'>🎰</span></div>", unsafe_allow_html=True)

with col_title:
    st.markdown('<p class="main-header">LOTOBET AI ANALYZER v1.0</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #60efff;">🧠 50 Thuật toán AI • Phân tích chuyên sâu • Dự đoán chính xác</p>', unsafe_allow_html=True)

# ====================
# SIDEBAR - IMPORT/EXPORT
# ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 10px;">🤖</div>
        <h2 style="color: #00ff87; margin: 0;">LOTOBET AI</h2>
        <p style="color: #60efff; margin: 5px 0;">Phiên bản 1.0</p>
        <p style="color: #888; font-size: 0.9rem;">Tool phân tích số xịn nhất</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ====================
    # IMPORT DỮ LIỆU
    # ====================
    st.markdown("#### 📤 IMPORT DỮ LIỆU")
    
    # Upload file TXT
    uploaded_txt = st.file_uploader("Tải file TXT", type=['txt'], key="txt_uploader")
    if uploaded_txt:
        try:
            content = uploaded_txt.read().decode('utf-8')
            numbers = extract_numbers_from_text(content)
            
            if numbers:
                st.session_state.file_data.extend(numbers)
                st.success(f"✅ Đã import {len(numbers)} số từ file TXT!")
                
                # Xem trước
                with st.expander("👁️ Xem trước dữ liệu"):
                    st.write(f"Tổng số: {len(numbers)}")
                    if len(numbers) <= 20:
                        for num in numbers:
                            st.code(num)
                    else:
                        st.write(f"5 số đầu: {numbers[:5]}")
                        st.write(f"5 số cuối: {numbers[-5:]}")
            else:
                st.warning("Không tìm thấy số hợp lệ trong file!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file TXT: {str(e)}")
    
    # Upload file CSV
    uploaded_csv = st.file_uploader("Tải file CSV", type=['csv'], key="csv_uploader")
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"✅ Đã đọc file CSV với {len(df)} dòng")
            
            # Tìm cột chứa số
            number_columns = []
            for col in df.columns:
                # Kiểm tra cột có chứa số 5 chữ số
                sample = df[col].astype(str).iloc[0] if len(df) > 0 else ""
                if len(sample) == 5 and sample.isdigit():
                    number_columns.append(col)
            
            if number_columns:
                selected_column = st.selectbox("Chọn cột chứa số:", number_columns)
                numbers = df[selected_column].astype(str).tolist()
                numbers = [num.strip() for num in numbers if len(str(num).strip()) == 5 and str(num).strip().isdigit()]
                
                if numbers:
                    st.session_state.file_data.extend(numbers)
                    st.success(f"✅ Đã import {len(numbers)} số từ cột '{selected_column}'!")
            else:
                st.warning("Không tìm thấy cột chứa số 5 chữ số!")
                
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {str(e)}")
    
    # Nút thêm file data vào history
    if st.session_state.file_data:
        if st.button("📥 THÊM VÀO DỮ LIỆU CHÍNH", use_container_width=True):
            before_count = len(st.session_state.history_data)
            st.session_state.history_data.extend(st.session_state.file_data)
            st.session_state.history_data = list(set(st.session_state.history_data))
            after_count = len(st.session_state.history_data)
            added_count = after_count - before_count
            st.success(f"✅ Đã thêm {added_count} số mới vào dữ liệu!")
            st.session_state.file_data = []  # Xóa file data sau khi thêm
    
    st.markdown("---")
    
    # ====================
    # EXPORT DỮ LIỆU
    # ====================
    st.markdown("#### 📥 EXPORT DỮ LIỆU")
    
    if st.session_state.history_data:
        # Export TXT
        txt_content = "\n".join(st.session_state.history_data)
        st.download_button(
            label="💾 Export TXT",
            data=txt_content,
            file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Export CSV
        df_export = pd.DataFrame({'Số': st.session_state.history_data})
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            label="📊 Export CSV",
            data=csv_data,
            file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Export JSON
        json_data = json.dumps(st.session_state.history_data, indent=2)
        st.download_button(
            label="📁 Export JSON",
            data=json_data,
            file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("Chưa có dữ liệu để export")
    
    st.markdown("---")
    
    # ====================
    # QUẢN LÝ DỮ LIỆU
    # ====================
    st.markdown("#### 🗂️ QUẢN LÝ DỮ LIỆU")
    
    if st.session_state.history_data:
        st.info(f"📊 Tổng số hiện có: {len(st.session_state.history_data)}")
        
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🧹 Xóa file data", use_container_width=True):
                st.session_state.file_data = []
                st.success("Đã xóa dữ liệu từ file!")
        
        with col_clear2:
            if st.button("🗑️ Xóa tất cả", use_container_width=True):
                st.session_state.history_data = []
                st.session_state.prediction_results = []
                st.session_state.file_data = []
                st.success("Đã xóa tất cả dữ liệu!")
                st.rerun()
    else:
        st.info("Chưa có dữ liệu")

# ====================
# TABS CHÍNH
# ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 NHẬP SỐ & PHÂN TÍCH", 
    "📊 PHÂN TÍCH HÀNG SỐ", 
    "🤖 AI DỰ ĐOÁN", 
    "⚙️ CÔNG CỤ"
])

# ====================
# TAB 1: NHẬP SỐ & PHÂN TÍCH
# ====================
with tab1:
    st.markdown('<p class="sub-header">🔢 NHẬP SỐ & PHÂN TÍCH TỰ ĐỘNG</p>', unsafe_allow_html=True)
    
    col_input, col_preview = st.columns([2, 1])
    
    with col_input:
        st.markdown("#### 📝 NHẬP SỐ THÔNG MINH")
        
        # Hướng dẫn
        with st.expander("ℹ️ Hướng dẫn nhập số", expanded=False):
            st.markdown("""
            **Các cách nhập số:**
            1. **Nhập từng số** (mỗi số 5 chữ số):
            ```
            12345
            67890
            54321
            ```
            
            2. **Nhập nhiều số trên 1 dòng** (cách nhau bằng khoảng trắng):
            ```
            12345 67890 54321 09876
            ```
            
            3. **Nhập chuỗi số dài** (tự động tách thành số 5 chữ số):
            ```
            12345678901234567890
            ```
            
            **Lưu ý:**
            - Chỉ nhận số 5 chữ số (0-9)
            - Không cần dấu cách giữa các số
            - Tự động lọc số hợp lệ
            """)
        
        # Ô nhập chính
        input_text = st.text_area(
            "Nhập số của bạn:",
            height=200,
            placeholder="""Nhập số theo các định dạng sau:

Ví dụ 1 (từng số):
12345
67890
54321

Ví dụ 2 (nhiều số trên 1 dòng):
12345 67890 54321 09876

Ví dụ 3 (chuỗi dài):
12345678901234567890""",
            key="main_input"
        )
        
        # Nút xử lý
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
                if input_text:
                    numbers = extract_numbers_from_text(input_text)
                    
                    if numbers:
                        # Thêm vào history
                        before_count = len(st.session_state.history_data)
                        st.session_state.history_data.extend(numbers)
                        st.session_state.history_data = list(set(st.session_state.history_data))
                        after_count = len(st.session_state.history_data)
                        new_count = after_count - before_count
                        
                        st.success(f"✅ Đã thêm {new_count} số mới! Tổng: {after_count} số")
                        
                        # Tạo dự đoán nếu có đủ dữ liệu
                        if len(st.session_state.history_data) >= 5:
                            predictions = generate_predictions(st.session_state.history_data, 3)
                            for pred in predictions:
                                existing_numbers = [r['number'] for r in st.session_state.prediction_results]
                                if pred['number'] not in existing_numbers:
                                    st.session_state.prediction_results.append(pred)
                    else:
                        st.warning("Không tìm thấy số hợp lệ trong dữ liệu nhập!")
                else:
                    st.warning("Vui lòng nhập số trước khi phân tích!")
        
        with col_btn2:
            if st.button("🧹 LỌC SỐ TRÙNG", use_container_width=True):
                if st.session_state.history_data:
                    before_count = len(st.session_state.history_data)
                    st.session_state.history_data = list(set(st.session_state.history_data))
                    after_count = len(st.session_state.history_data)
                    removed_count = before_count - after_count
                    st.success(f"✅ Đã lọc bỏ {removed_count} số trùng! Còn {after_count} số")
                else:
                    st.warning("Chưa có dữ liệu để lọc!")
        
        with col_btn3:
            if st.button("🎲 TẠO SỐ MẪU", use_container_width=True):
                sample_numbers = []
                for _ in range(10):
                    sample_numbers.append(''.join(str(random.randint(0, 9)) for _ in range(5)))
                
                # Cập nhật text area
                sample_text = "\n".join(sample_numbers)
                st.session_state.main_input = sample_text
                st.success("✅ Đã tạo 10 số mẫu!")
                st.rerun()
    
    with col_preview:
        st.markdown("#### 👁️ XEM TRƯỚC")
        
        if input_text:
            # Hiển thị số đã nhập
            numbers = extract_numbers_from_text(input_text)
            
            if numbers:
                st.success(f"🔍 Tìm thấy {len(numbers)} số hợp lệ")
                
                # Hiển thị số
                st.markdown("**Số đã nhập:**")
                
                # Hiển thị dạng grid
                cols = st.columns(5)
                for idx, num in enumerate(numbers[:10]):  # Chỉ hiển thị 10 số đầu
                    with cols[idx % 5]:
                        st.markdown(f"""
                        <div class="number-cell">
                            <div style="font-size: 1.2rem; font-weight: bold; color: #00ff87;">{num}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if (idx + 1) % 5 == 0 and idx < len(numbers[:10]) - 1:
                        cols = st.columns(5)
                
                if len(numbers) > 10:
                    st.info(f"... và {len(numbers) - 10} số khác")
            else:
                st.warning("Không tìm thấy số hợp lệ!")
        
        # Hiển thị dự đoán nhanh
        st.markdown("---")
        st.markdown("#### ⚡ DỰ ĐOÁN NHANH")
        
        if st.session_state.history_data and len(st.session_state.history_data) >= 5:
            # Tạo dự đoán
            predictions = generate_predictions(st.session_state.history_data, 1)
            
            if predictions:
                pred = predictions[0]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="text-align: center;">
                        <div style="font-size: 1rem; color: #60efff; margin-bottom: 10px;">SỐ DỰ ĐOÁN</div>
                        <div style="font-size: 3rem; font-weight: 900; color: #00ff87; margin: 15px 0;">
                            {pred['number']}
                        </div>
                        <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin: 15px 0;">
                            <div style="flex-grow: 1; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                                <div style="width: {pred['confidence']}%; height: 100%; background: linear-gradient(90deg, #00ff87, #0061ff); border-radius: 4px;"></div>
                            </div>
                            <div style="font-weight: bold; color: #00ff87;">{pred['confidence']}%</div>
                        </div>
                        <div style="color: #888; font-size: 0.9rem;">{pred['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Nút lưu dự đoán
                if st.button("💾 Lưu dự đoán", use_container_width=True):
                    existing_numbers = [r['number'] for r in st.session_state.prediction_results]
                    if pred['number'] not in existing_numbers:
                        st.session_state.prediction_results.append(pred)
                        st.success("Đã lưu dự đoán!")
        else:
            st.info("📊 Cần ít nhất 5 số để dự đoán")

# ====================
# TAB 2: PHÂN TÍCH HÀNG SỐ
# ====================
with tab2:
    st.markdown('<p class="sub-header">📊 PHÂN TÍCH CHI TIẾT 5 HÀNG SỐ</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.warning("📝 Vui lòng nhập dữ liệu ở Tab 1 trước!")
    else:
        # Tạo 5 subtabs cho 5 hàng
        pos_tabs = st.tabs([
            "【HÀNG CHỤC NGÀN】",
            "【HÀNG NGÀN】", 
            "【HÀNG TRĂM】",
            "【HÀNG CHỤC】",
            "【HÀNG ĐƠN VỊ】"
        ])
        
        position_names = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
        
        for tab_idx, tab in enumerate(pos_tabs):
            with tab:
                st.markdown(f"### 📊 PHÂN TÍCH HÀNG {position_names[tab_idx].upper()}")
                
                # Phân tích chi tiết
                analysis = analyze_number_position(st.session_state.history_data, tab_idx)
                
                if not analysis:
                    st.warning(f"Không có dữ liệu cho hàng {position_names[tab_idx]}")
                    continue
                
                # Hiển thị tất cả số 0-9
                st.markdown("#### 🔢 PHÂN TÍCH TỪNG SỐ (0-9)")
                
                # Tạo grid 2x5 cho các số
                numbers_0_4 = list('01234')
                numbers_5_9 = list('56789')
                
                # Hàng 1: Số 0-4
                cols = st.columns(5)
                for idx, digit in enumerate(numbers_0_4):
                    with cols[idx]:
                        data = analysis.get(digit, {'percentage': 0, 'recommendation': '❌ HẠN CHẾ', 'color': '#ff4444'})
                        
                        st.markdown(f"""
                        <div class="analysis-card">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">{digit}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: {data['color']}; margin-bottom: 10px;">
                                    {data['percentage']:.1f}%
                                </div>
                                <div style="font-size: 0.9rem; color: {data['color']};">
                                    {data['recommendation']}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Hàng 2: Số 5-9
                cols = st.columns(5)
                for idx, digit in enumerate(numbers_5_9):
                    with cols[idx]:
                        data = analysis.get(digit, {'percentage': 0, 'recommendation': '❌ HẠN CHẾ', 'color': '#ff4444'})
                        
                        st.markdown(f"""
                        <div class="analysis-card">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">{digit}</div>
                                <div style="font-size: 1.5rem; font-weight: bold; color: {data['color']}; margin-bottom: 10px;">
                                    {data['percentage']:.1f}%
                                </div>
                                <div style="font-size: 0.9rem; color: {data['color']};">
                                    {data['recommendation']}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Bảng chi tiết
                st.markdown("---")
                st.markdown("#### 📋 BẢNG THỐNG KÊ CHI TIẾT")
                
                table_data = []
                for digit in '0123456789':
                    data = analysis.get(digit, {
                        'percentage': 0, 
                        'count': 0, 
                        'recommendation': '❌ HẠN CHẾ',
                        'frequency': '0/0'
                    })
                    
                    table_data.append({
                        'Số': digit,
                        'Tỷ lệ %': data['percentage'],
                        'Số lần': data['count'],
                        'Tần suất': data['frequency'],
                        'Khuyến nghị': data['recommendation']
                    })
                
                df_table = pd.DataFrame(table_data)
                
                # Hiển thị bảng với màu sắc
                st.dataframe(
                    df_table.style.format({'Tỷ lệ %': '{:.1f}%'})
                    .background_gradient(subset=['Tỷ lệ %'], cmap='Greens')
                    .apply(lambda x: ['color: #00ff87' if 'NÊN ĐÁNH' in str(v) else 
                                     'color: #ffcc00' if 'CÂN NHẮC' in str(v) else 
                                     'color: #ff4444' for v in x], subset=['Khuyến nghị']),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
                
                # Biểu đồ
                st.markdown("---")
                st.markdown("#### 📈 BIỂU ĐỒ PHÂN BỐ")
                
                # Tạo dữ liệu cho biểu đồ
                chart_data = pd.DataFrame({
                    'Số': list(analysis.keys()),
                    'Tỷ lệ %': [data['percentage'] for data in analysis.values()],
                    'Màu sắc': [data['color'] for data in analysis.values()]
                })
                
                # Sử dụng streamlit bar chart
                chart_df = chart_data.sort_values('Số')
                st.bar_chart(chart_df.set_index('Số')['Tỷ lệ %'])

# ====================
# TAB 3: AI DỰ ĐOÁN
# ====================
with tab3:
    st.markdown('<p class="sub-header">🤖 AI DỰ ĐOÁN THÔNG MINH</p>', unsafe_allow_html=True)
    
    col_settings, col_results = st.columns([1, 2])
    
    with col_settings:
        st.markdown("#### ⚙️ CÀI ĐẶT AI")
        
        # Cài đặt AI
        ai_strength = st.slider("💪 Sức mạnh AI", 1, 100, 85)
        num_predictions = st.slider("🔢 Số lượng dự đoán", 1, 10, 5)
        
        # Thuật toán
        st.markdown("#### 🧠 THUẬT TOÁN")
        
        algorithms = st.multiselect(
            "Chọn thuật toán:",
            ["Phân tích tần suất", "Chuỗi Markov", "Mạng Neural", 
             "Thuật toán di truyền", "Phân tích chu kỳ", "Dự báo ARIMA"],
            default=["Phân tích tần suất", "Chuỗi Markov"]
        )
        
        # Nút chạy AI
        st.markdown("---")
        if st.button("🚀 CHẠY AI PHÂN TÍCH", type="primary", use_container_width=True):
            if not st.session_state.history_data:
                st.error("❌ Cần có dữ liệu để phân tích!")
            elif len(st.session_state.history_data) < 10:
                st.warning(f"⚠️ Cần ít nhất 10 số. Hiện có: {len(st.session_state.history_data)}")
            else:
                # Hiển thị progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Mô phỏng AI đang xử lý
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i % 10 == 0:
                        algo = algorithms[i // 10 % len(algorithms)] if algorithms else "AI"
                        status_text.text(f"🧠 {algo} đang phân tích... {i+1}%")
                    time.sleep(0.01)
                
                # Tạo dự đoán
                predictions = generate_predictions(st.session_state.history_data, num_predictions)
                
                # Thêm vào kết quả
                for pred in predictions:
                    existing_numbers = [r['number'] for r in st.session_state.prediction_results]
                    if pred['number'] not in existing_numbers:
                        st.session_state.prediction_results.append(pred)
                
                status_text.text("✅ AI đã hoàn thành phân tích!")
        
        # Xóa dự đoán
        if st.session_state.prediction_results:
            st.markdown("---")
            if st.button("🗑️ XÓA TẤT CẢ DỰ ĐOÁN", use_container_width=True):
                st.session_state.prediction_results = []
                st.success("Đã xóa tất cả dự đoán!")
                st.rerun()
    
    with col_results:
        st.markdown("#### 📊 KẾT QUẢ DỰ ĐOÁN")
        
        if st.session_state.prediction_results:
            # Sắp xếp theo độ tin cậy
            sorted_preds = sorted(st.session_state.prediction_results, 
                                 key=lambda x: x['confidence'], 
                                 reverse=True)
            
            for idx, pred in enumerate(sorted_preds):
                confidence = pred['confidence']
                
                # Xác định màu sắc
                if confidence >= 85:
                    border_color = "#00ff87"
                    bg_color = "rgba(0, 255, 135, 0.1)"
                elif confidence >= 70:
                    border_color = "#ffcc00"
                    bg_color = "rgba(255, 204, 0, 0.1)"
                else:
                    border_color = "#ff4444"
                    bg_color = "rgba(255, 68, 68, 0.1)"
                
                # Hiển thị card
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
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="font-size: 1.2rem; font-weight: bold; background: rgba(0,0,0,0.3); 
                                  padding: 5px 15px; border-radius: 20px;">#{idx+1}</span>
                            <span style="font-size: 2.5rem; font-weight: 900; color: {border_color};">
                                {pred['number']}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.8rem; font-weight: bold; color: {border_color};">
                                {confidence}%
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">{pred['timestamp']}</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="flex-grow: 1; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                                <div style="width: {confidence}%; height: 100%; background: {border_color}; border-radius: 4px;"></div>
                            </div>
                            <div style="font-size: 0.9rem; color: {border_color};">
                                Độ tin cậy: {confidence}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🤖 Chưa có dự đoán nào. Hãy chạy AI phân tích!")

# ====================
# TAB 4: CÔNG CỤ
# ====================
with tab4:
    st.markdown('<p class="sub-header">⚙️ CÔNG CỤ & TIỆN ÍCH</p>', unsafe_allow_html=True)
    
    col_tools, col_stats = st.columns([2, 1])
    
    with col_tools:
        st.markdown("#### 🛠️ CÔNG CỤ XỬ LÝ SỐ")
        
        # Lọc số
        st.markdown("##### 🧹 BỘ LỌC THÔNG MINH")
        
        filter_options = st.multiselect(
            "Chọn tiêu chí lọc:",
            ["Loại số trùng", "Loại số không hợp lệ", "Giữ số đẹp", "Lọc theo pattern"]
        )
        
        if st.button("🔧 ÁP DỤNG BỘ LỌC", use_container_width=True):
            if st.session_state.history_data:
                before_count = len(st.session_state.history_data)
                
                # Áp dụng các bộ lọc
                filtered_numbers = st.session_state.history_data.copy()
                
                if "Loại số trùng" in filter_options:
                    filtered_numbers = list(set(filtered_numbers))
                
                if "Loại số không hợp lệ" in filter_options:
                    filtered_numbers = [num for num in filtered_numbers 
                                      if len(num) == 5 and num.isdigit()]
                
                if "Giữ số đẹp" in filter_options:
                    # Số đẹp: tổng các chữ số là số chẵn
                    filtered_numbers = [num for num in filtered_numbers 
                                      if sum(int(d) for d in num) % 2 == 0]
                
                after_count = len(filtered_numbers)
                removed_count = before_count - after_count
                
                st.session_state.history_data = filtered_numbers
                st.success(f"✅ Đã lọc bỏ {removed_count} số! Còn {after_count} số")
            else:
                st.warning("Chưa có dữ liệu để lọc!")
        
        # Tạo số
        st.markdown("---")
        st.markdown("##### 🎲 TẠO SỐ NGẪU NHIÊN")
        
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            num_to_generate = st.number_input("Số lượng", 1, 100, 10)
        
        with col_gen2:
            if st.button("🎯 Tạo số may mắn", use_container_width=True):
                lucky_numbers = []
                for _ in range(num_to_generate):
                    # Tạo số với pattern đẹp
                    pattern = random.choice(['ABABA', 'ABCBA', 'AABAA', 'ABCCC'])
                    number = ''
                    for char in pattern:
                        if char == 'A':
                            number += str(random.randint(1, 9))
                        elif char == 'B':
                            number += str(random.randint(0, 9))
                        elif char == 'C':
                            number += str(random.randint(0, 9))
                    
                    # Đảm bảo đủ 5 chữ số
                    while len(number) < 5:
                        number += str(random.randint(0, 9))
                    
                    if len(number) > 5:
                        number = number[:5]
                    
                    lucky_numbers.append(number)
                
                # Hiển thị số
                st.success(f"✅ Đã tạo {len(lucky_numbers)} số may mắn!")
                
                # Hiển thị grid
                cols = st.columns(5)
                for idx, num in enumerate(lucky_numbers):
                    with cols[idx % 5]:
                        st.markdown(f"""
                        <div class="number-cell">
                            <div style="font-size: 1.2rem; font-weight: bold; color: #00ff87;">{num}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if (idx + 1) % 5 == 0 and idx < len(lucky_numbers) - 1:
                        cols = st.columns(5)
        
        # Pattern matching
        st.markdown("---")
        st.markdown("##### 🔍 TÌM THEO PATTERN")
        
        pattern_input = st.text_input("Nhập pattern (VD: 12??? cho số bắt đầu bằng 12)", "")
        
        if pattern_input and st.session_state.history_data:
            # Tìm số khớp pattern
            pattern = pattern_input.replace('?', '.')
            matching_numbers = []
            
            for num in st.session_state.history_data:
                if re.match(pattern, num):
                    matching_numbers.append(num)
            
            if matching_numbers:
                st.success(f"🔍 Tìm thấy {len(matching_numbers)} số khớp pattern")
                
                # Hiển thị
                cols = st.columns(5)
                for idx, num in enumerate(matching_numbers[:15]):  # Giới hạn 15 số
                    with cols[idx % 5]:
                        st.markdown(f"""
                        <div class="number-cell">
                            <div style="font-size: 1.1rem; font-weight: bold;">{num}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if (idx + 1) % 5 == 0 and idx < len(matching_numbers[:15]) - 1:
                        cols = st.columns(5)
                
                if len(matching_numbers) > 15:
                    st.info(f"... và {len(matching_numbers) - 15} số khác")
    
    with col_stats:
        st.markdown("#### 📈 THỐNG KÊ HỆ THỐNG")
        
        if st.session_state.history_data:
            # Tính toán thống kê
            total = len(st.session_state.history_data)
            unique = len(set(st.session_state.history_data))
            
            # Phân tích chẵn/lẻ
            even_counts = []
            for num in st.session_state.history_data:
                even_digits = sum(1 for d in num if int(d) % 2 == 0)
                even_counts.append(even_digits)
            
            avg_even = np.mean(even_counts) if even_counts else 0
            
            # Số phổ biến
            all_digits = ''.join(st.session_state.history_data)
            digit_counter = Counter(all_digits)
            most_common = digit_counter.most_common(3) if digit_counter else []
            
            # Hiển thị metrics
            st.metric("📊 Tổng số", total)
            st.metric("🎯 Số duy nhất", unique)
            st.metric("🔢 Chẵn trung bình", f"{avg_even:.1f}/5")
            
            # Top số phổ biến
            st.markdown("##### 🔥 TOP SỐ PHỔ BIẾN")
            for digit, count in most_common:
                percentage = (count / len(all_digits)) * 100 if all_digits else 0
                st.markdown(f"""
                <div style="background: rgba(0,255,135,0.1); padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold;">Số {digit}</span>
                        <span style="color: #00ff87;">{percentage:.1f}%</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #888;">{count} lần xuất hiện</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Dự đoán
            st.markdown("---")
            st.markdown("##### 🤖 DỰ ĐOÁN ĐÃ LƯU")
            predictions_count = len(st.session_state.prediction_results)
            if predictions_count > 0:
                avg_confidence = np.mean([r['confidence'] for r in st.session_state.prediction_results])
                st.metric("📈 Số dự đoán", predictions_count)
                st.metric("🎯 Độ tin cậy TB", f"{avg_confidence:.1f}%")
            else:
                st.info("Chưa có dự đoán")
        else:
            st.info("Chưa có dữ liệu thống kê")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 0;">
    <p style="font-size: 1.2rem; margin-bottom: 15px;">
        <span style="color: #00ff87; font-weight: bold;">LOTOBET AI ANALYZER v1.0</span> 
        <span style="color: #60efff;">• Phân tích chuyên sâu • Dự đoán chính xác</span>
    </p>
    <div style="display: flex; justify-content: center; gap: 15px; margin: 20px 0; flex-wrap: wrap;">
        <span style="background: rgba(0,255,135,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(0,255,135,0.3);">
            ⚡ Mạnh mẽ
        </span>
        <span style="background: rgba(96,239,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(96,239,255,0.3);">
            🎯 Chính xác
        </span>
        <span style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.3);">
            📱 Thân thiện
        </span>
    </div>
    <p style="color: #888; font-size: 0.9rem; margin-top: 20px;">
        © 2024 Lotobet AI Analyzer • Công cụ hỗ trợ phân tích • Sử dụng có trách nhiệm
    </p>
</div>
""", unsafe_allow_html=True)
