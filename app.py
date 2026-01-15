# ================= LOTOBET AI PRO – V10.2 OPTIMIZED LAYOUT =================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AI PRO V10.2",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Optimized CSS
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Compact table headers */
    .table-header {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    /* Prediction cards - compact */
    .prediction-card {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        text-align: center;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .prediction-card-current {
        border: 2px solid #3B82F6;
        background-color: #EFF6FF;
    }
    
    .prediction-card-next {
        border: 2px solid #94A3B8;
        background-color: #F8FAFC;
    }
    
    /* Number displays - compact */
    .compact-big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E40AF;
        margin: 3px 0;
    }
    
    .compact-small-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #475569;
        margin: 3px 0;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px 0;
    }
    
    .confidence-high { background-color: #10B981; color: white; }
    .confidence-medium { background-color: #F59E0B; color: white; }
    .confidence-low { background-color: #EF4444; color: white; }
    
    /* Recommendation icons */
    .rec-icon {
        font-size: 1.2rem;
        margin: 2px;
    }
    
    .rec-good { color: #10B981; }
    .rec-maybe { color: #F59E0B; }
    .rec-bad { color: #EF4444; }
    
    /* Horizontal analysis row */
    .analysis-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 8px;
        background-color: #F8FAFC;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .algo-item {
        text-align: center;
        padding: 5px;
        min-width: 40px;
    }
    
    .algo-number {
        font-size: 0.9rem;
        font-weight: bold;
        color: #475569;
    }
    
    .algo-progress {
        height: 4px;
        background-color: #E2E8F0;
        border-radius: 2px;
        margin: 3px 0;
        overflow: hidden;
    }
    
    .algo-progress-fill {
        height: 100%;
        background-color: #3B82F6;
        border-radius: 2px;
    }
    
    .algo-confidence {
        font-size: 0.7rem;
        color: #64748B;
    }
    
    /* Pattern and tip items */
    .pattern-item, .tip-item {
        display: inline-block;
        text-align: center;
        margin: 0 5px;
        padding: 4px 8px;
        background-color: white;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }
    
    /* Capital management table */
    .capital-table {
        width: 100%;
        font-size: 0.9rem;
    }
    
    .capital-table td {
        padding: 6px 8px;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .capital-bar {
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .capital-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    /* Notification box */
    .notification-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    
    /* Input area */
    .compact-textarea {
        font-size: 0.9rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .compact-big-number { font-size: 1.8rem; }
        .compact-small-number { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================= SIMULATED DATA & FUNCTIONS =================

def tai_xiu(tong):
    return "TÀI" if tong >= 23 else "XỈU"

def le_chan(tong):
    return "LẺ" if tong % 2 else "CHẴN"

def format_tien(tien):
    return f"{tien:,.0f}₫"

def get_confidence_color(confidence):
    if confidence >= 75:
        return "confidence-high"
    elif confidence >= 65:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_recommendation_icon(confidence, threshold=65):
    if confidence >= threshold:
        return "✅"
    elif confidence >= threshold - 10:
        return "⚠️"
    else:
        return "❌"

# Simulated current predictions
current_predictions = {
    'ky': '116043',
    '2_so': {'number': '68', 'confidence': 75, 'recommendation': '✅'},
    '3_so': {'number': '168', 'confidence': 72, 'recommendation': '✅'},
    'tai_xiu': {'prediction': 'TÀI', 'confidence': 68, 'recommendation': '✅', 'should_bet': True},
    'le_chan': {'prediction': 'LẺ', 'confidence': 65, 'recommendation': '⚠️', 'should_bet': False},
    'de_numbers': ['56', '78', '65', '89', '68'],
    'de_confidence': 70
}

# Simulated next predictions
next_predictions = {
    'ky': '116044',
    '2_so': {'number': '79', 'confidence': 70, 'recommendation': '⚠️'},
    '3_so': {'number': '279', 'confidence': 68, 'recommendation': '⚠️'},
    'tai_xiu': {'prediction': 'XỈU', 'confidence': 65, 'recommendation': '⚠️', 'should_bet': False},
    'le_chan': {'prediction': 'CHẴN', 'confidence': 62, 'recommendation': '⚠️', 'should_bet': False},
    'de_numbers': ['89', '45', '67', '23', '34'],
    'de_confidence': 67
}

# Simulated algorithm analysis
algorithms = [
    {'id': 1, 'name': 'Basic Stats', 'confidence': 75, 'enabled': True},
    {'id': 2, 'name': 'Hot/Cold', 'confidence': 80, 'enabled': True},
    {'id': 3, 'name': 'Pattern Rec', 'confidence': 65, 'enabled': True},
    {'id': 4, 'name': 'Time Series', 'confidence': 78, 'enabled': True},
    {'id': 5, 'name': 'ML Predict', 'confidence': 62, 'enabled': True},
    {'id': 6, 'name': 'Cycle Anal', 'confidence': 85, 'enabled': True},
    {'id': 7, 'name': 'Probability', 'confidence': 58, 'enabled': True},
    {'id': 8, 'name': 'Cloud AI', 'confidence': 72, 'enabled': True}
]

# Simulated patterns
patterns = [
    {'id': 1, 'name': 'Straight', 'count': 3, 'active': True},
    {'id': 2, 'name': 'Wave', 'count': 2, 'active': True},
    {'id': 3, 'name': 'Mirror', 'count': 0, 'active': False},
    {'id': 4, 'name': 'Ladder', 'count': 1, 'active': True},
    {'id': 5, 'name': 'Repeat', 'count': 0, 'active': False}
]

# Simulated tips
tips = [
    {'id': 1, 'name': 'Bạc Nhớ', 'count': 4, 'applied': True},
    {'id': 2, 'name': 'Lô Gan', 'count': 3, 'applied': True},
    {'id': 3, 'name': 'Chạm Đầu', 'count': 2, 'applied': True},
    {'id': 4, 'name': 'Tổng Đề', 'count': 3, 'applied': True},
    {'id': 5, 'name': 'Bóng Số', 'count': 0, 'applied': False},
    {'id': 6, 'name': 'Kẹp Số', 'count': 3, 'applied': True}
]

# Capital management
capital_data = {
    'total': 1000000,
    'distribution': {
        '2_so': {'amount': 175000, 'percentage': 35},
        '3_so': {'amount': 150000, 'percentage': 30},
        'tai_xiu': {'amount': 100000, 'percentage': 20},
        'le_chan': {'amount': 75000, 'percentage': 15}
    },
    'sufficient': True
}

# ================= MAIN APP WITH NEW LAYOUT =================

def main():
    # Header - Compact
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown("### 🎰 LOTOBET AI PRO V10.2")
    
    with col_header2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"🕒 {current_time} | 📊 300 kỳ")
    
    st.markdown("---")
    
    # ========== BẢNG 1: THU THẬP DỮ LIỆU & ĐỒNG BỘ KỲ ==========
    st.markdown('<div class="table-header">📥 BẢNG 1: THU THẬP DỮ LIỆU & ĐỒNG BỘ KỲ</div>', unsafe_allow_html=True)
    
    col1_1, col1_2, col1_3 = st.columns([3, 2, 2])
    
    with col1_1:
        # Input area
        st.markdown("**Nhập dữ liệu:**")
        raw_data = st.text_area(
            "Dán kết quả hoặc nhập số:",
            height=80,
            placeholder="Mỗi dòng 1 số 5 chữ số\nVD:\n12345\n67890",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📥 Từ file", use_container_width=True):
                st.success("Chức năng upload file")
        with col_btn2:
            if st.button("💾 Lưu dữ liệu", type="primary", use_container_width=True):
                if raw_data:
                    st.success("✅ Đã lưu dữ liệu!")
    
    with col1_2:
        # Ky synchronization
        st.markdown("**Đồng bộ kỳ:**")
        
        current_ky = st.text_input(
            "Kỳ hiện tại:",
            value="116043",
            max_chars=6,
            label_visibility="collapsed"
        )
        
        st.markdown(f"**Trạng thái:** ✅ Đã đồng bộ")
        st.caption(f"Kỳ tiếp: **{int(current_ky) + 1}**")
        
        if st.button("🔄 Cập nhật kỳ", use_container_width=True):
            st.success("✅ Đã cập nhật kỳ mới!")
    
    with col1_3:
        # Database info
        st.markdown("**Database:**")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            st.metric("Tổng kỳ", "300")
        with col_db2:
            st.metric("Hôm nay", "15")
        
        if st.button("📋 Xem 10 kỳ gần nhất", use_container_width=True):
            st.info("Hiển thị 10 kỳ gần nhất...")
    
    st.markdown("---")
    
    # ========== BẢNG 2: KẾT LUẬN SỐ ĐÁNH KỲ HIỆN TẠI ==========
    st.markdown(f'<div class="table-header">🎯 BẢNG 2: KẾT LUẬN SỐ ĐÁNH KỲ {current_predictions["ky"]} (HIỆN TẠI)</div>', unsafe_allow_html=True)
    
    # Create 5 columns for predictions
    col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)
    
    with col2_1:
        # 2 Số
        pred = current_predictions['2_so']
        st.markdown('<div class="prediction-card prediction-card-current">', unsafe_allow_html=True)
        st.markdown("**🔥 2 SỐ**")
        st.markdown(f'<div class="compact-big-number">{pred["number"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_2:
        # 3 Số
        pred = current_predictions['3_so']
        st.markdown('<div class="prediction-card prediction-card-current">', unsafe_allow_html=True)
        st.markdown("**🔥 3 SỐ**")
        st.markdown(f'<div class="compact-big-number">{pred["number"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_3:
        # Tài/Xỉu
        pred = current_predictions['tai_xiu']
        st.markdown('<div class="prediction-card prediction-card-current">', unsafe_allow_html=True)
        st.markdown("**🎲 TÀI/XỈU**")
        st.markdown(f'<div class="compact-big-number">{pred["prediction"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_4:
        # Lẻ/Chẵn
        pred = current_predictions['le_chan']
        st.markdown('<div class="prediction-card prediction-card-current">', unsafe_allow_html=True)
        st.markdown("**🎲 LẺ/CHẴN**")
        st.markdown(f'<div class="compact-big-number">{pred["prediction"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2_5:
        # Số đề
        st.markdown('<div class="prediction-card prediction-card-current">', unsafe_allow_html=True)
        st.markdown("**🎯 SỐ ĐỀ**")
        de_nums = current_predictions['de_numbers'][:3]
        for num in de_nums:
            st.markdown(f'<div class="compact-small-number">{num}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(current_predictions["de_confidence"])} confidence-badge">{current_predictions["de_confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== BẢNG 3: DỰ ĐOÁN ĐÁNH KỲ TIẾP THEO ==========
    st.markdown(f'<div class="table-header">🔮 BẢNG 3: DỰ ĐOÁN ĐÁNH KỲ {next_predictions["ky"]} (TIẾP THEO)</div>', unsafe_allow_html=True)
    
    # Create 5 columns for next predictions
    col3_1, col3_2, col3_3, col3_4, col3_5 = st.columns(5)
    
    with col3_1:
        # 2 Số (Next)
        pred = next_predictions['2_so']
        st.markdown('<div class="prediction-card prediction-card-next">', unsafe_allow_html=True)
        st.markdown("**🔥 2 SỐ**")
        st.markdown(f'<div class="compact-big-number">{pred["number"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3_2:
        # 3 Số (Next)
        pred = next_predictions['3_so']
        st.markdown('<div class="prediction-card prediction-card-next">', unsafe_allow_html=True)
        st.markdown("**🔥 3 SỐ**")
        st.markdown(f'<div class="compact-big-number">{pred["number"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3_3:
        # Tài/Xỉu (Next)
        pred = next_predictions['tai_xiu']
        st.markdown('<div class="prediction-card prediction-card-next">', unsafe_allow_html=True)
        st.markdown("**🎲 TÀI/XỈU**")
        st.markdown(f'<div class="compact-big-number">{pred["prediction"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3_4:
        # Lẻ/Chẵn (Next)
        pred = next_predictions['le_chan']
        st.markdown('<div class="prediction-card prediction-card-next">', unsafe_allow_html=True)
        st.markdown("**🎲 LẺ/CHẴN**")
        st.markdown(f'<div class="compact-big-number">{pred["prediction"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(pred["confidence"])} confidence-badge">{pred["confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="rec-icon">{pred["recommendation"]}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3_5:
        # Số đề (Next)
        st.markdown('<div class="prediction-card prediction-card-next">', unsafe_allow_html=True)
        st.markdown("**🎯 SỐ ĐỀ**")
        de_nums = next_predictions['de_numbers'][:3]
        for num in de_nums:
            st.markdown(f'<div class="compact-small-number">{num}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="{get_confidence_color(next_predictions["de_confidence"])} confidence-badge">{next_predictions["de_confidence"]}%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== BẢNG 4: THÔNG BÁO ĐÁNH CÙNG KỲ ==========
    st.markdown(f'<div class="table-header">🔔 BẢNG 4: THÔNG BÁO ĐÁNH CÙNG KỲ {current_predictions["ky"]}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="notification-box">
    <strong>🎯 ĐÁNH NGAY CÙNG NHÀ CÁI:</strong><br>
    • <strong>2 Tinh:</strong> <code>68</code> (vào số <code>6</code> và <code>8</code>)<br>
    • <strong>3 Tinh:</strong> <code>168</code> (vào <code>1,6,8</code>) • <code>867</code> • <code>568</code><br>
    • <strong>Tài/Xỉu:</strong> ✅ <strong>NÊN ĐÁNH</strong> <code>TÀI</code> (68%)<br>
    • <strong>Số đề:</strong> <code>56</code>, <code>78</code>, <code>65</code>, <code>89</code>, <code>68</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== BẢNG 5: QUẢN LÝ VỐN THÔNG MINH ==========
    st.markdown('<div class="table-header">💰 BẢNG 5: QUẢN LÝ VỐN THÔNG MINH</div>', unsafe_allow_html=True)
    
    col5_1, col5_2 = st.columns([2, 3])
    
    with col5_1:
        st.markdown("**Tổng vốn hiện có:**")
        total_capital = st.number_input(
            "Nhập số vốn (VNĐ):",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000,
            label_visibility="collapsed"
        )
        
        risk_level = st.selectbox(
            "**Mức rủi ro:**",
            ["Thấp", "Trung bình", "Cao"],
            index=1
        )
        
        if st.button("🧮 TÍNH PHÂN BỔ", type="primary", use_container_width=True):
            st.success("✅ Đã tính toán phân bổ vốn!")
    
    with col5_2:
        st.markdown("**Phân bổ đề xuất:**")
        
        st.markdown("""
        <table class="capital-table">
        <tr>
            <td><strong>Loại cược</strong></td>
            <td><strong>Số tiền</strong></td>
            <td><strong>Tỷ lệ</strong></td>
            <td><strong>Tiến độ</strong></td>
        </tr>
        <tr>
            <td>2 Số</td>
            <td>175.000₫</td>
            <td>35%</td>
            <td><div class="capital-bar"><div class="capital-bar-fill" style="width:35%;background-color:#3B82F6"></div></div></td>
        </tr>
        <tr>
            <td>3 Số</td>
            <td>150.000₫</td>
            <td>30%</td>
            <td><div class="capital-bar"><div class="capital-bar-fill" style="width:30%;background-color:#10B981"></div></div></td>
        </tr>
        <tr>
            <td>Tài/Xỉu</td>
            <td>100.000₫</td>
            <td>20%</td>
            <td><div class="capital-bar"><div class="capital-bar-fill" style="width:20%;background-color:#F59E0B"></div></div></td>
        </tr>
        <tr>
            <td>Lẻ/Chẵn</td>
            <td>75.000₫</td>
            <td>15%</td>
            <td><div class="capital-bar"><div class="capital-bar-fill" style="width:15%;background-color:#EF4444"></div></div></td>
        </tr>
        <tr style="border-top:2px solid #CBD5E1;font-weight:bold;">
            <td><strong>Tổng</strong></td>
            <td><strong>500.000₫</strong></td>
            <td><strong>50%</strong></td>
            <td><div class="capital-bar"><div class="capital-bar-fill" style="width:50%;background-color:#6B7280"></div></div></td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.caption("💡 *Sử dụng tối đa 50% vốn cho mỗi kỳ, giữ lại 50% dự phòng*")
    
    st.markdown("---")
    
    # ========== BẢNG 6: PHÂN TÍCH HỆ THỐNG ==========
    st.markdown('<div class="table-header">🤖 BẢNG 6: PHÂN TÍCH HỆ THỐNG</div>', unsafe_allow_html=True)
    
    # Algorithms
    st.markdown("**📊 8 THUẬT TOÁN:**")
    algo_cols = st.columns(8)
    for i, algo in enumerate(algorithms):
        with algo_cols[i]:
            progress = algo['confidence'] / 100
            st.markdown(f'<div class="algo-item">', unsafe_allow_html=True)
            st.markdown(f'<div class="algo-number">A{algo["id"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="algo-progress"><div class="algo-progress-fill" style="width:{progress*100}%"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="algo-confidence">{algo["confidence"]}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Patterns
    st.markdown("**🌀 5 MẪU HÌNH:**")
    pattern_cols = st.columns(5)
    for i, pattern in enumerate(patterns):
        with pattern_cols[i]:
            badge = "🟢" if pattern['active'] else "⚫"
            st.markdown(f'<div class="pattern-item">', unsafe_allow_html=True)
            st.markdown(f'{badge} P{pattern["id"]}')
            st.markdown(f'<div style="font-size:0.8rem">{pattern["count"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips
    st.markdown("**💡 6 MẸO ĐÁNH:**")
    tip_cols = st.columns(6)
    for i, tip in enumerate(tips):
        with tip_cols[i]:
            badge = "✅" if tip['applied'] else "❌"
            st.markdown(f'<div class="tip-item">', unsafe_allow_html=True)
            st.markdown(f'{badge} T{tip["id"]}')
            st.markdown(f'<div style="font-size:0.8rem">{tip["count"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6B7280;font-size:0.8rem">
    <strong>LOTOBET AI PRO – V10.2 OPTIMIZED LAYOUT</strong><br>
    8 Thuật toán • 5 Mẫu hình • 6 Mẹo đánh • Quản lý vốn thông minh<br>
    ⚠️ Dành cho mục đích phân tích và nghiên cứu
    </div>
    """, unsafe_allow_html=True)

# ================= RUN APP =================
if __name__ == "__main__":
    main()
