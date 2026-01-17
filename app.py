# app.py - LOTOBET AI ANALYZER v1.0 (Tối ưu giao diện)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json
import re
from collections import Counter

# ====================
# CẤU HÌNH TRANG
# ====================
st.set_page_config(
    page_title="Lotobet AI Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# CSS TÙY CHỈNH - ĐẸP & NHẸ
# ====================
st.markdown("""
<style>
    /* Reset và nền cơ bản */
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }
    
    /* Header chính */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        padding: 0.5rem;
    }
    
    /* Sub-header */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        border-left: 4px solid #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
    }
    
    /* Tabs đơn giản */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #1e293b;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        color: #cbd5e1;
        font-weight: 600;
        border-radius: 8px;
        margin: 0 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }
    
    /* Input boxes */
    .stTextArea textarea {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 2px solid #475569 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-family: monospace;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Number grid */
    .number-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin: 15px 0;
    }
    
    .number-cell {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        transition: all 0.2s;
    }
    
    .number-cell:hover {
        border-color: #3b82f6;
        transform: scale(1.05);
    }
    
    /* Analysis cards */
    .pos-card {
        background: #1e293b;
        border: 2px solid #475569;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Progress bars */
    .progress-bar {
        height: 8px;
        background: #334155;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    /* File uploader */
    .uploadedFile {
        background: #1e293b !important;
        border: 2px dashed #475569 !important;
        border-radius: 10px !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: #1e293b !important;
        border-radius: 10px !important;
    }
    
    /* Metric */
    .stMetric {
        background: #1e293b !important;
        border-radius: 10px !important;
        padding: 15px !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# KHỞI TẠO SESSION STATE
# ====================
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# ====================
# HÀM TIỆN ÍCH
# ====================
def extract_numbers(text):
    """Trích xuất số 5 chữ số từ text"""
    if not text:
        return []
    
    # Tìm tất cả số 5 chữ số
    numbers = re.findall(r'\d{5}', text)
    
    # Xử lý chuỗi dài không có khoảng cách
    long_nums = re.findall(r'\d{6,}', text)
    for num in long_nums:
        for i in range(0, len(num), 5):
            if i + 5 <= len(num):
                numbers.append(num[i:i+5])
    
    # Lọc số hợp lệ
    valid_numbers = []
    for num in numbers:
        if len(num) == 5 and num.isdigit():
            valid_numbers.append(num)
    
    return list(set(valid_numbers))

def analyze_position(data, pos_idx):
    """Phân tích một vị trí cụ thể"""
    if not data:
        return {}
    
    digits = []
    for num in data:
        if len(num) > pos_idx:
            digits.append(num[pos_idx])
    
    if not digits:
        return {}
    
    counter = Counter(digits)
    total = len(digits)
    
    result = {}
    for digit in '0123456789':
        count = counter.get(digit, 0)
        percent = (count / total) * 100
        
        # Đánh giá
        if percent >= 15:
            rating = "hot"
            color = "#10b981"  # Xanh lá
        elif percent >= 8:
            rating = "normal"
            color = "#f59e0b"  # Vàng
        else:
            rating = "cold"
            color = "#ef4444"  # Đỏ
        
        result[digit] = {
            'count': count,
            'percent': percent,
            'rating': rating,
            'color': color
        }
    
    return result

def generate_top_predictions(data, n=2):
    """Tạo 2 dự đoán có tỷ lệ thắng cao nhất"""
    if len(data) < 5:
        return []
    
    predictions = []
    
    for _ in range(n):
        # Phân tích từng vị trí
        number = ""
        confidence = 0
        
        for pos in range(5):
            pos_digits = [num[pos] for num in data[-20:] if len(num) > pos]
            if pos_digits:
                counter = Counter(pos_digits)
                most_common = counter.most_common(1)
                if most_common:
                    digit = most_common[0][0]
                    freq = most_common[0][1]
                    number += digit
                    confidence += (freq / len(pos_digits)) * 20
                else:
                    number += str(random.randint(0, 9))
            else:
                number += str(random.randint(0, 9))
        
        confidence = min(95, confidence)
        predictions.append({
            'number': number,
            'confidence': round(confidence, 1)
        })
    
    return predictions

# ====================
# HEADER
# ====================
st.markdown('<p class="main-header">🎯 LOTOBET AI ANALYZER</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#94a3b8">Tool phân tích số nhẹ & đẹp</p>', unsafe_allow_html=True)

# ====================
# SIDEBAR - IMPORT/EXPORT
# ====================
with st.sidebar:
    st.markdown("### 📁 IMPORT/EXPORT")
    
    # Upload file TXT
    uploaded_txt = st.file_uploader("Chọn file TXT", type=['txt'])
    if uploaded_txt:
        try:
            content = uploaded_txt.read().decode('utf-8')
            numbers = extract_numbers(content)
            
            if numbers:
                # Hiển thị preview
                with st.expander(f"Preview ({len(numbers)} số)"):
                    cols = st.columns(3)
                    for idx, num in enumerate(numbers[:9]):
                        with cols[idx % 3]:
                            st.markdown(f"`{num}`")
                    
                    if len(numbers) > 9:
                        st.caption(f"... và {len(numbers) - 9} số khác")
                
                # Nút import
                if st.button("📥 IMPORT VÀO HỆ THỐNG", use_container_width=True):
                    old_count = len(st.session_state.history_data)
                    st.session_state.history_data.extend(numbers)
                    st.session_state.history_data = list(set(st.session_state.history_data))
                    new_count = len(st.session_state.history_data)
                    added = new_count - old_count
                    
                    st.success(f"✅ Đã thêm {added} số mới")
                    st.rerun()
            else:
                st.warning("Không tìm thấy số hợp lệ trong file!")
        except:
            st.error("Lỗi khi đọc file!")
    
    # Upload file CSV
    uploaded_csv = st.file_uploader("Chọn file CSV", type=['csv'])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"Đọc được {len(df)} dòng")
            
            # Tìm cột chứa số
            for col in df.columns:
                sample = str(df[col].iloc[0]) if len(df) > 0 else ""
                if len(sample) == 5 and sample.isdigit():
                    numbers = df[col].astype(str).tolist()
                    numbers = [n.strip() for n in numbers if len(str(n).strip()) == 5]
                    
                    with st.expander(f"Cột '{col}' ({len(numbers)} số)"):
                        st.write(f"5 số đầu: {numbers[:5]}")
                    
                    if st.button(f"📥 IMPORT TỪ '{col}'", use_container_width=True):
                        old_count = len(st.session_state.history_data)
                        st.session_state.history_data.extend(numbers)
                        st.session_state.history_data = list(set(st.session_state.history_data))
                        st.success(f"✅ Đã thêm {len(st.session_state.history_data) - old_count} số")
                        st.rerun()
                    
                    break
        except:
            st.error("Lỗi khi đọc CSV!")
    
    st.markdown("---")
    
    # Export dữ liệu
    if st.session_state.history_data:
        st.markdown("#### 📤 EXPORT DỮ LIỆU")
        
        # Export TXT
        txt_content = "\n".join(st.session_state.history_data)
        st.download_button(
            label="💾 Export TXT",
            data=txt_content,
            file_name=f"lotobet_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Export CSV
        df = pd.DataFrame({'Số': st.session_state.history_data})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Export CSV",
            data=csv,
            file_name=f"lotobet_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Quản lý dữ liệu
    if st.session_state.history_data:
        st.markdown(f"**📊 Tổng số:** {len(st.session_state.history_data)}")
        
        if st.button("🗑️ XÓA TẤT CẢ DỮ LIỆU", type="secondary", use_container_width=True):
            st.session_state.history_data = []
            st.session_state.predictions = []
            st.success("Đã xóa dữ liệu!")
            st.rerun()

# ====================
# TABS CHÍNH
# ====================
tab1, tab2, tab3 = st.tabs(["📝 NHẬP SỐ", "📊 PHÂN TÍCH", "🤖 AI DỰ ĐOÁN"])

# ====================
# TAB 1: NHẬP SỐ
# ====================
with tab1:
    st.markdown('<p class="section-title">📝 NHẬP SỐ THÔNG MINH</p>', unsafe_allow_html=True)
    
    # Ô nhập số
    input_text = st.text_area(
        "Nhập số của bạn (nhiều định dạng đều được):",
        height=150,
        placeholder="""Các cách nhập:
• Từng số riêng: 12345
                67890
                54321
                
• Nhiều số trên 1 dòng: 12345 67890 54321
                
• Chuỗi dài: 123456789012345""",
        key="number_input"
    )
    
    # Nút xử lý
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
            if input_text:
                numbers = extract_numbers(input_text)
                
                if numbers:
                    old_count = len(st.session_state.history_data)
                    st.session_state.history_data.extend(numbers)
                    st.session_state.history_data = list(set(st.session_state.history_data))
                    new_count = len(st.session_state.history_data)
                    
                    # Tạo dự đoán
                    if new_count >= 5:
                        st.session_state.predictions = generate_top_predictions(st.session_state.history_data, 2)
                    
                    st.success(f"✅ Đã thêm {new_count - old_count} số mới! Tổng: {new_count} số")
                    st.rerun()
                else:
                    st.warning("Không tìm thấy số hợp lệ!")
            else:
                st.warning("Vui lòng nhập số!")
    
    with col2:
        if st.button("🎲 TẠO SỐ MẪU", use_container_width=True):
            sample = []
            for _ in range(10):
                sample.append(''.join(str(random.randint(0, 9)) for _ in range(5)))
            
            # Cập nhật ô nhập
            sample_text = "\n".join(sample)
            st.session_state.number_input = sample_text
            st.rerun()
    
    # Preview số đã nhập
    if input_text:
        numbers = extract_numbers(input_text)
        if numbers:
            st.markdown(f"**🔍 Tìm thấy {len(numbers)} số hợp lệ:**")
            
            # Hiển thị dạng grid
            st.markdown('<div class="number-grid">', unsafe_allow_html=True)
            cols = st.columns(5)
            for idx, num in enumerate(numbers[:20]):
                with cols[idx % 5]:
                    st.markdown(f'''
                    <div class="number-cell">
                        <div style="font-size:1.2rem;font-weight:bold;color:#60a5fa">{num}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                if (idx + 1) % 5 == 0 and idx < len(numbers[:20]) - 1:
                    cols = st.columns(5)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if len(numbers) > 20:
                st.caption(f"... và {len(numbers) - 20} số khác")

# ====================
# TAB 2: PHÂN TÍCH HÀNG SỐ
# ====================
with tab2:
    st.markdown('<p class="section-title">📊 PHÂN TÍCH CHI TIẾT 5 HÀNG SỐ</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.info("📝 Vui lòng nhập số ở Tab 1 trước!")
    else:
        # Tạo 5 subtabs cho 5 hàng
        pos_names = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
        pos_tabs = st.tabs([f"【{name}】" for name in pos_names])
        
        for tab_idx, tab in enumerate(pos_tabs):
            with tab:
                # Phân tích vị trí này
                analysis = analyze_position(st.session_state.history_data, tab_idx)
                
                if not analysis:
                    st.warning("Không có dữ liệu")
                    continue
                
                # 1. HIỂN THỊ SỐ 0-9 THEO HÀNG NGANG
                st.markdown("### Số 0-9:")
                
                # Tạo hàng số 0-9
                cols = st.columns(10)
                for i in range(10):
                    digit = str(i)
                    data = analysis.get(digit, {'percent': 0, 'color': '#6b7280'})
                    
                    with cols[i]:
                        # Card số
                        st.markdown(f'''
                        <div class="pos-card" style="border-color:{data['color']}">
                            <div style="text-align:center">
                                <div style="font-size:1.8rem;font-weight:bold;margin-bottom:8px">{digit}</div>
                                <div style="font-size:1.2rem;color:{data['color']};font-weight:bold">
                                    {data['percent']:.1f}%
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 2. PHÂN TÍCH TỈ LỆ % THEO DẠNG LƯỚI
                st.markdown("### 📈 Phân tích tỉ lệ %:")
                
                # Tạo 2x5 grid cho phân tích chi tiết
                row1_cols = st.columns(5)
                row2_cols = st.columns(5)
                
                digits_0_4 = list('01234')
                digits_5_9 = list('56789')
                
                # Hàng 1: Số 0-4
                for idx, digit in enumerate(digits_0_4):
                    data = analysis.get(digit, {'percent': 0, 'color': '#6b7280', 'count': 0})
                    
                    with row1_cols[idx]:
                        # Progress bar
                        st.markdown(f'''
                        <div style="margin:10px 0">
                            <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                                <span style="font-weight:bold">Số {digit}</span>
                                <span style="color:{data['color']}">{data['percent']:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{min(100, data['percent']*2)}%;background:{data['color']}"></div>
                            </div>
                            <div style="text-align:center;margin-top:5px;font-size:0.9rem;color:#94a3b8">
                                {data['count']} lần
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Hàng 2: Số 5-9
                for idx, digit in enumerate(digits_5_9):
                    data = analysis.get(digit, {'percent': 0, 'color': '#6b7280', 'count': 0})
                    
                    with row2_cols[idx]:
                        # Progress bar
                        st.markdown(f'''
                        <div style="margin:10px 0">
                            <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                                <span style="font-weight:bold">Số {digit}</span>
                                <span style="color:{data['color']}">{data['percent']:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:{min(100, data['percent']*2)}%;background:{data['color']}"></div>
                            </div>
                            <div style="text-align:center;margin-top:5px;font-size:0.9rem;color:#94a3b8">
                                {data['count']} lần
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # 3. ĐÁNH GIÁ TỔNG QUAN
                st.markdown("---")
                st.markdown("### 🎯 Đánh giá:")
                
                # Tìm số nóng nhất và lạnh nhất
                hot_digits = []
                cold_digits = []
                
                for digit in '0123456789':
                    data = analysis.get(digit, {'percent': 0})
                    if data['percent'] >= 15:
                        hot_digits.append((digit, data['percent']))
                    elif data['percent'] <= 5:
                        cold_digits.append((digit, data['percent']))
                
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    if hot_digits:
                        st.markdown("**🔥 Số nóng (Nên đánh):**")
                        for digit, percent in sorted(hot_digits, key=lambda x: x[1], reverse=True)[:3]:
                            st.markdown(f'<div style="color:#10b981">• Số {digit}: {percent:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("**📊 Số trung bình:**")
                        st.info("Chưa có số đủ nóng")
                
                with col_eval2:
                    if cold_digits:
                        st.markdown("**❄️ Số lạnh (Hạn chế):**")
                        for digit, percent in sorted(cold_digits, key=lambda x: x[1])[:3]:
                            st.markdown(f'<div style="color:#ef4444">• Số {digit}: {percent:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("**📊 Số trung bình:**")
                        st.info("Chưa có số quá lạnh")

# ====================
# TAB 3: AI DỰ ĐOÁN
# ====================
with tab3:
    st.markdown('<p class="section-title">🤖 AI DỰ ĐOÁN</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.info("📝 Cần nhập số để AI phân tích!")
    else:
        # Nút tạo dự đoán
        if st.button("🚀 AI PHÂN TÍCH & DỰ ĐOÁN", type="primary", use_container_width=True):
            if len(st.session_state.history_data) >= 10:
                with st.spinner("AI đang phân tích..."):
                    # Tạo dự đoán
                    predictions = generate_top_predictions(st.session_state.history_data, 2)
                    st.session_state.predictions = predictions
                    
                    st.success("✅ AI đã hoàn thành phân tích!")
            else:
                st.warning(f"Cần ít nhất 10 số. Hiện có: {len(st.session_state.history_data)}")
        
        # Hiển thị dự đoán
        if st.session_state.predictions:
            st.markdown("### 🏆 2 DỰ ĐOÁN TỐT NHẤT:")
            
            for idx, pred in enumerate(st.session_state.predictions):
                confidence = pred['confidence']
                
                # Màu theo độ tin cậy
                if confidence >= 80:
                    color = "#10b981"
                    bg_color = "rgba(16, 185, 129, 0.1)"
                elif confidence >= 60:
                    color = "#f59e0b"
                    bg_color = "rgba(245, 158, 11, 0.1)"
                else:
                    color = "#ef4444"
                    bg_color = "rgba(239, 68, 68, 0.1)"
                
                # Card dự đoán
                st.markdown(f'''
                <div style="background:{bg_color};border-radius:15px;padding:20px;margin:15px 0;border-left:5px solid {color}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px">
                        <div>
                            <span style="font-size:1.2rem;font-weight:bold;background:rgba(0,0,0,0.2);padding:5px 15px;border-radius:20px">
                                #{idx+1}
                            </span>
                            <span style="font-size:2.5rem;font-weight:900;margin-left:15px;color:{color}">
                                {pred['number']}
                            </span>
                        </div>
                        <div style="text-align:right">
                            <div style="font-size:1.5rem;font-weight:bold;color:{color}">
                                {pred['confidence']}%
                            </div>
                            <div style="font-size:0.9rem;color:#94a3b8">Tỷ lệ thắng</div>
                        </div>
                    </div>
                    
                    <div style="margin-top:15px">
                        <div style="display:flex;align-items:center;gap:10px">
                            <div style="flex-grow:1;background:rgba(255,255,255,0.1);height:8px;border-radius:4px">
                                <div style="width:{pred['confidence']}%;height:100%;background:{color};border-radius:4px"></div>
                            </div>
                            <div style="font-size:0.9rem;color:{color}">
                                Độ tin cậy: {pred['confidence']}%
                            </div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Thống kê AI
        if st.session_state.history_data:
            st.markdown("---")
            st.markdown("### 📈 THỐNG KÊ AI:")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                total = len(st.session_state.history_data)
                st.metric("📊 Tổng số", total)
            
            with col_stat2:
                unique = len(set(st.session_state.history_data))
                st.metric("🎯 Số duy nhất", unique)
            
            with col_stat3:
                if st.session_state.predictions:
                    avg_conf = np.mean([p['confidence'] for p in st.session_state.predictions])
                    st.metric("🤖 Độ tin cậy TB", f"{avg_conf:.1f}%")
                else:
                    st.metric("🤖 Độ tin cậy TB", "N/A")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:20px 0;color:#64748b">
    <p>🎯 <strong>LOTOBET AI ANALYZER v1.0</strong> • Tool nhẹ & đẹp</p>
    <p style="font-size:0.9rem">Công cụ hỗ trợ phân tích • Sử dụng có trách nhiệm</p>
</div>
""", unsafe_allow_html=True)
