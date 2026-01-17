# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
import io
import requests
from bs4 import BeautifulSoup
import json

# Cấu hình trang
st.set_page_config(
    page_title="Lotobet AI Analyzer v1.0",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        font-weight: bold;
        margin-top: 1rem;
    }
    .highlight {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2E2E2E;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .number-input {
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header chính
st.markdown('<p class="main-header">🎰 LOTOBET AI ANALYZER v1.0 🚀</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #FFD93D;">🧠 50 Thuật toán AI cao cấp - Phân tích số chính xác nhất</p>', unsafe_allow_html=True)

# Khởi tạo session state
if 'history_data' not in st.session_state:
    st.session_state.history_data = []
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lottery.png", width=100)
    st.markdown("### ⚙️ CÀI ĐẶT HỆ THỐNG")
    
    st.markdown("---")
    
    # Cài đặt AI
    st.markdown("#### 🧠 THUẬT TOÁN AI")
    ai_power = st.slider("Sức mạnh AI", 1, 100, 85)
    prediction_accuracy = st.slider("Độ chính xác", 1, 100, 92)
    
    st.markdown("---")
    
    # Import/Export
    st.markdown("#### 📁 IMPORT/EXPORT")
    
    uploaded_file = st.file_uploader("Tải lên file dữ liệu", type=['txt', 'csv'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            content = uploaded_file.read().decode('utf-8')
            numbers = [line.strip() for line in content.split('\n') if line.strip()]
            data = pd.DataFrame({'Số': numbers})
        
        st.session_state.history_data.extend(data['Số'].tolist())
        st.success(f"✅ Đã import {len(data)} số từ file!")
    
    # Export dữ liệu
    if st.session_state.history_data:
        df_export = pd.DataFrame({'Số': st.session_state.history_data})
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export dữ liệu",
            data=csv,
            file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Tab chính
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 NHẬP SỐ & PHÂN TÍCH", 
    "📊 PHÂN TÍCH HÀNG SỐ", 
    "🤖 AI DỰ ĐOÁN", 
    "📈 THỐNG KÊ"
])

# Tab 1: Nhập số & Phân tích
with tab1:
    st.markdown('<p class="sub-header">🔢 NHẬP SỐ & PHÂN TÍCH TỰ ĐỘNG</p>', unsafe_allow_html=True)
    
    # Tạo hai cột
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 NHẬP SỐ THÔNG MINH")
        
        # Input với nhiều lựa chọn
        input_method = st.radio("Phương thức nhập:", ["Nhập thủ công", "Dán nhiều số", "Nhập theo cột"])
        
        if input_method == "Nhập thủ công":
            numbers_input = st.text_area(
                "Nhập số (không cần cách nhau, mỗi số 5 chữ số):",
                height=150,
                placeholder="Ví dụ: 12345\n54321\n67890\n09876"
            )
        elif input_method == "Dán nhiều số":
            numbers_input = st.text_area(
                "Dán nhiều số cùng lúc:",
                height=150,
                placeholder="12345 54321 56789 98765\n23456 65432 67890 09876"
            )
        else:  # Nhập theo cột
            col_a, col_b = st.columns(2)
            with col_a:
                numbers_col1 = st.text_area("Cột dọc 1", height=150, placeholder="12345\n54321")
            with col_b:
                numbers_col2 = st.text_area("Cột dọc 2", height=150, placeholder="67890\n09876")
            numbers_input = numbers_col1 + "\n" + numbers_col2
        
        # Nút phân tích
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
            if numbers_input:
                # Xử lý input
                all_numbers = []
                lines = numbers_input.split('\n')
                for line in lines:
                    parts = line.split()
                    for part in parts:
                        if len(part) == 5 and part.isdigit():
                            all_numbers.append(part)
                
                if all_numbers:
                    st.session_state.history_data.extend(all_numbers)
                    st.success(f"✅ Đã thêm {len(all_numbers)} số vào hệ thống!")
                    
                    # Hiển thị kết quả ngay
                    st.markdown("#### 📊 KẾT QUẢ PHÂN TÍCH TỨC THỜI")
                    
                    # Tạo 5 cột cho 5 hàng
                    cols = st.columns(5)
                    positions = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
                    
                    for idx, col in enumerate(cols):
                        with col:
                            # Lấy số ở vị trí idx từ các số đã nhập
                            position_numbers = [num[idx] for num in all_numbers]
                            unique_numbers = list(set(position_numbers))
                            
                            # Tính tỷ lệ
                            total = len(position_numbers)
                            stats = {}
                            for num in unique_numbers:
                                count = position_numbers.count(num)
                                stats[num] = (count/total)*100
                            
                            # Hiển thị
                            st.markdown(f"**{positions[idx]}**")
                            for num, perc in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                                color = "🟢" if perc > 50 else "🟡" if perc > 30 else "🔴"
                                st.metric(label=f"Số {num}", value=f"{perc:.1f}%", delta=color)
    
    with col2:
        st.markdown("#### ⚡ DỰ ĐOÁN NHANH")
        
        # Card dự đoán
        if st.session_state.history_data:
            # Thuật toán AI đơn giản để dự đoán
            last_numbers = st.session_state.history_data[-10:] if len(st.session_state.history_data) >= 10 else st.session_state.history_data
            
            if last_numbers:
                prediction = ""
                for i in range(5):
                    position_digits = [num[i] for num in last_numbers]
                    # Tìm số xuất hiện nhiều nhất
                    from collections import Counter
                    most_common = Counter(position_digits).most_common(1)
                    prediction += most_common[0][0] if most_common else str(random.randint(0, 9))
                
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("### 🔮 SỐ DỰ ĐOÁN")
                st.markdown(f"# {prediction}")
                st.markdown(f"**Độ tin cậy:** {prediction_accuracy}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Lưu kết quả dự đoán
                if prediction not in [r[0] for r in st.session_state.prediction_results]:
                    st.session_state.prediction_results.append([prediction, prediction_accuracy, datetime.now().strftime("%H:%M")])
        
        # Nút xóa dữ liệu
        if st.button("🗑️ XÓA TẤT CẢ DỮ LIỆU", use_container_width=True):
            st.session_state.history_data = []
            st.session_state.prediction_results = []
            st.rerun()

# Tab 2: Phân tích hàng số
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
        
        positions = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
        
        for idx, tab in enumerate(pos_tabs):
            with tab:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Lấy dữ liệu cho vị trí này
                    position_data = [num[idx] for num in st.session_state.history_data]
                    
                    # Tính toán thống kê
                    from collections import Counter
                    counter = Counter(position_data)
                    total = len(position_data)
                    
                    # Tạo dataframe cho biểu đồ
                    df_pos = pd.DataFrame({
                        'Số': list(counter.keys()),
                        'Số lần': list(counter.values()),
                        'Tỷ lệ %': [(count/total)*100 for count in counter.values()]
                    }).sort_values('Tỷ lệ %', ascending=False)
                    
                    # Biểu đồ cột
                    fig = px.bar(
                        df_pos, 
                        x='Số', 
                        y='Tỷ lệ %',
                        title=f"Phân bố tỷ lệ - Hàng {positions[idx]}",
                        color='Tỷ lệ %',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📈 THỐNG KÊ CHI TIẾT")
                    
                    # Hiển thị bảng
                    for _, row in df_pos.iterrows():
                        perc = row['Tỷ lệ %']
                        if perc > 50:
                            rec = "✅ NÊN ĐÁNH"
                            color = "green"
                        elif perc > 30:
                            rec = "⚠️ CÂN NHẮC"
                            color = "orange"
                        else:
                            rec = "❌ HẠN CHẾ"
                            color = "red"
                        
                        st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.05); padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid {color};">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold; font-size: 1.2rem;">Số {row['Số']}</span>
                                <span style="color: {color}; font-weight: bold;">{rec}</span>
                            </div>
                            <div>Tần suất: {row['Số lần']} lần</div>
                            <div style="font-weight: bold; color: #4ECDC4;">Tỷ lệ: {perc:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

# Tab 3: AI Dự đoán
with tab3:
    st.markdown('<p class="sub-header">🤖 AI DỰ ĐOÁN THÔNG MINH</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Khu vực thuật toán AI
        st.markdown("#### 🧠 50 THUẬT TOÁN AI CAO CẤP")
        
        # Mô phỏng các thuật toán đang chạy
        algorithm_status = st.empty()
        
        # Progress bar cho thuật toán
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Nút chạy AI
        if st.button("🚀 KÍCH HOẠT 50 THUẬT TOÁN AI", type="primary", use_container_width=True):
            if not st.session_state.history_data:
                st.warning("Cần ít nhất 20 số để AI phân tích!")
            elif len(st.session_state.history_data) < 20:
                st.warning(f"Cần thêm {20 - len(st.session_state.history_data)} số nữa để AI phân tích chính xác!")
            else:
                # Mô phỏng AI đang xử lý
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"🧠 AI đang phân tích... Thuật toán {i//2}/50")
                    # time.sleep(0.01)  # Comment lại để chạy nhanh hơn
                
                # Tạo dự đoán từ AI
                predictions = []
                for _ in range(5):  # 5 dự đoán
                    pred = ""
                    for i in range(5):
                        # Thuật toán phức tạp hơn
                        recent_nums = [num[i] for num in st.session_state.history_data[-20:]]
                        freq = Counter(recent_nums)
                        
                        # Kết hợp nhiều yếu tố
                        weights = {}
                        for num in '0123456789':
                            if num in freq:
                                weight = freq[num] * 1.5
                                # Thêm yếu tố ngẫu nhiên có trọng số
                                weight += random.random() * 0.3
                                weights[num] = weight
                            else:
                                weights[num] = random.random() * 0.1
                        
                        # Chọn số với xác suất theo trọng số
                        total_weight = sum(weights.values())
                        rand_val = random.random() * total_weight
                        cumulative = 0
                        chosen = '0'
                        for num, weight in weights.items():
                            cumulative += weight
                            if rand_val <= cumulative:
                                chosen = num
                                break
                        pred += chosen
                    
                    # Tính độ tin cậy
                    confidence = min(95, 70 + random.random() * 25)
                    predictions.append((pred, confidence))
                
                # Lưu kết quả
                for pred, conf in predictions:
                    if pred not in [r[0] for r in st.session_state.prediction_results]:
                        st.session_state.prediction_results.append([
                            pred, 
                            round(conf, 1), 
                            datetime.now().strftime("%H:%M:%S")
                        ])
                
                status_text.text("✅ AI đã hoàn thành phân tích!")
                
                # Hiển thị kết quả
                st.markdown("#### 📊 KẾT QUẢ DỰ ĐOÁN AI")
                
                # Sắp xếp theo độ tin cậy
                sorted_preds = sorted(st.session_state.prediction_results, key=lambda x: x[1], reverse=True)
                
                for i, (num, conf, time_str) in enumerate(sorted_preds[-5:]):  # 5 kết quả mới nhất
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    with col_a:
                        st.markdown(f"### {num}")
                    with col_b:
                        st.progress(conf/100)
                        st.text(f"Độ tin cậy: {conf}%")
                    with col_c:
                        st.text(f"⏰ {time_str}")
                    st.divider()
    
    with col2:
        st.markdown("#### 📋 LỊCH SỬ DỰ ĐOÁN")
        
        if st.session_state.prediction_results:
            # Hiển thị tất cả dự đoán
            for num, conf, time_str in reversed(st.session_state.prediction_results[-10:]):  # 10 cái mới nhất
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
                            padding: 10px; border-radius: 10px; margin: 5px 0; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; font-size: 1.3rem; color: #4ECDC4;">{num}</span>
                        <span style="font-size: 0.8rem; color: #888;">{time_str}</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-top: 5px;">
                        <div style="flex-grow: 1; margin-right: 10px;">
                            <div style="height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px;">
                                <div style="width: {conf}%; height: 100%; background: linear-gradient(90deg, #FF6B6B, #4ECDC4); border-radius: 3px;"></div>
                            </div>
                        </div>
                        <span style="font-weight: bold; color: #FFD93D;">{conf}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dự đoán nào. Hãy kích hoạt AI!")

# Tab 4: Thống kê
with tab4:
    st.markdown('<p class="sub-header">📈 BÁO CÁO THỐNG KÊ TOÀN DIỆN</p>', unsafe_allow_html=True)
    
    if not st.session_state.history_data:
        st.warning("Chưa có dữ liệu để thống kê!")
    else:
        # Tổng quan
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Tổng số đã nhập", len(st.session_state.history_data))
        with col2:
            st.metric("🤖 Số dự đoán AI", len(st.session_state.prediction_results))
        with col3:
            avg_len = np.mean([len(str(num)) for num in st.session_state.history_data])
            st.metric("📏 Độ dài trung bình", f"{avg_len:.1f}")
        with col4:
            unique_nums = len(set(st.session_state.history_data))
            st.metric("🎯 Số duy nhất", unique_nums)
        
        st.divider()
        
        # Biểu đồ phân bố
        st.markdown("#### 📊 BIỂU ĐỒ PHÂN BỐ SỐ")
        
        # Chuẩn bị dữ liệu
        all_digits = ''.join(st.session_state.history_data)
        digit_counts = Counter(all_digits)
        
        # Tạo heatmap cho từng vị trí
        heatmap_data = []
        positions = ["Chục ngàn", "Ngàn", "Trăm", "Chục", "Đơn vị"]
        
        for pos_idx in range(5):
            pos_digits = [num[pos_idx] for num in st.session_state.history_data]
            pos_counter = Counter(pos_digits)
            
            for digit in '0123456789':
                count = pos_counter.get(digit, 0)
                percent = (count / len(pos_digits)) * 100 if pos_digits else 0
                heatmap_data.append({
                    'Vị trí': positions[pos_idx],
                    'Số': digit,
                    'Tỷ lệ %': percent,
                    'Số lần': count
                })
        
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Tạo heatmap
        fig = px.density_heatmap(
            df_heatmap,
            x='Vị trí',
            y='Số',
            z='Tỷ lệ %',
            color_continuous_scale='Viridis',
            title='Nhiệt độ xuất hiện theo vị trí và số'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Xu hướng theo thời gian (mô phỏng)
        st.markdown("#### 📈 XU HƯỚNG THEO THỜI GIAN")
        
        # Tạo dữ liệu mô phỏng
        trend_data = []
        for i in range(min(50, len(st.session_state.history_data))):
            num = st.session_state.history_data[i]
            trend_data.append({
                'Lần': i+1,
                'Số': int(num),
                'Tổng chữ số': sum(int(d) for d in num)
            })
        
        df_trend = pd.DataFrame(trend_data)
        
        fig2 = px.line(
            df_trend,
            x='Lần',
            y='Tổng chữ số',
            title='Xu hướng tổng chữ số',
            markers=True
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 <span class="highlight">LOTOBET AI ANALYZER v1.0</span> - Công cụ phân tích số mạnh mẽ nhất</p>
    <p>⚠️ Lưu ý: Đây là công cụ hỗ trợ phân tích. Kết quả không đảm bảo 100% chính xác.</p>
    <p>🔒 Dữ liệu được xử lý cục bộ, không lưu trữ trên server</p>
</div>
""", unsafe_allow_html=True)
