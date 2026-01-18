import streamlit as st
import pandas as pd
import re
from collections import Counter

# --- 1. CẤU HÌNH HỆ THỐNG & GIAO DIỆN ---
st.set_page_config(page_title="AI SUPREME v4.5 ELITE", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTextArea textarea { background-color: #111; color: #00FF00; border: 1px solid #444; font-size: 18px !important; }
    .stButton>button { width: 100%; background: linear-gradient(to right, #ff4b2b, #ff416c); color: white; border: none; font-weight: bold; height: 50px; }
    .result-box { padding: 20px; border-radius: 15px; background: #1a1a1a; border: 2px solid #333; margin-top: 15px; }
    .highlight-score { font-size: 40px; font-weight: bold; color: #FF0000; text-align: center; }
    .label-custom { font-size: 14px; color: #888; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THUẬT TOÁN XỬ LÝ DỮ LIỆU THÔNG MINH (SMART ANALYTICS) ---
def clean_and_format_data(raw_input):
    """Lọc bỏ mã kỳ, ký tự lạ, chỉ giữ lại số đơn lẻ (0-9)"""
    # Khử các dãy số kỳ dài (ví dụ: 260118396)
    data_no_sessions = re.sub(r'\d{6,}', ' ', raw_input)
    # Lấy tất cả các chữ số từ 0-9
    numbers = re.findall(r'\d', data_no_sessions)
    return [int(n) for n in numbers]

def calculate_ai_logic(numbers):
    """Thuật toán Điểm Nổ chuyên sâu cho Sảnh A"""
    if not numbers: return None
    
    total_len = len(numbers)
    counts = Counter(numbers)
    
    # Tìm nhịp vắng (Gap) - Cực kỳ quan trọng cho Sảnh A
    last_positions = {i: -1 for i in range(10)}
    for idx, val in enumerate(numbers):
        last_positions[val] = idx
        
    results = []
    for num in range(10):
        freq = counts[num]
        # Khoảng cách từ lần cuối xuất hiện đến hiện tại
        gap = (total_len - 1) - last_positions[num]
        
        # Công thức Điểm Nổ v4.5: (Tần suất * Hệ số) + Thưởng nhịp vắng + Ưu tiên số 0
        score = (freq * 1.2) 
        if 3 <= gap <= 6: score += 12  # Nhịp vàng sảnh A
        if num == 0 and gap > 2: score += 5 # Ưu tiên số 0 (Cầu hồi)
        if gap > 12: score -= 15 # Cầu quá khan, nên bỏ
        
        results.append({'num': num, 'score': round(max(0, score), 2)})
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

# --- 3. GIAO DIỆN NGƯỜI DÙNG (USER INTERFACE) ---
st.title("🤖 AI SUPREME v4.5")
st.markdown("<p style='text-align: center; color: #888;'>SẢNH A ELITE | AUTO-FILTER | XIÊN 3</p>", unsafe_allow_html=True)

# Khung nạp dữ liệu
with st.container():
    st.markdown("<div class='label-custom'>📥 NẠP DỮ LIỆU (QUÉT S-PEN DỌC/NGANG):</div>", unsafe_allow_html=True)
    input_text = st.text_area("", placeholder="Dán kết quả tại đây...", height=120, label_visibility="collapsed")
    
    col_run, col_reset = st.columns([3, 1])
    with col_run:
        btn_active = st.button("🚀 KÍCH HOẠT HỆ THỐNG")
    with col_reset:
        if st.button("🔄"):
            st.rerun()

# --- 4. HIỂN THỊ KẾT QUẢ ---
if btn_active and input_text:
    data = clean_and_format_data(input_text)
    
    if len(data) < 5:
        st.error("Dữ liệu quá ngắn! Hãy quét thêm ít nhất 5-10 kỳ.")
    else:
        results = calculate_ai_logic(data)
        top1 = results[0]
        top2 = results[1]
        top3 = results[2]
        
        # Vùng hiển thị Bạch Thủ & Điểm Nổ
        st.markdown(f"""
            <div class='result-box'>
                <div style='text-align: center; color: #888;'>🎯 BẠCH THỦ TIỀM NĂNG</div>
                <div style='text-align: center; font-size: 60px; font-weight: bold; color: #00FF00;'>{top1['num']}</div>
                <div class='highlight-score'>ĐIỂM NỔ: {top1['score']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Vùng hiển thị Xiên
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div style='background: #111; padding: 15px; border-radius: 10px; border: 1px solid #444; text-align: center;'>
                    <div style='color: #888; font-size: 12px;'>✨ XIÊN 2</div>
                    <div style='font-size: 22px; font-weight: bold; color: #00CCFF;'>{top1['num']} - {top2['num']}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div style='background: #111; padding: 15px; border-radius: 10px; border: 1px solid #444; text-align: center;'>
                    <div style='color: #888; font-size: 12px;'>🏆 XIÊN 3</div>
                    <div style='font-size: 22px; font-weight: bold; color: #FFD700;'>{top1['num']} - {top2['num']} - {top3['num']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Lệnh thực chiến dựa trên Điểm Nổ
        st.markdown("<br>", unsafe_allow_html=True)
        if top1['score'] >= 20:
            st.success("🔥 LỆNH: VÀO TIỀN MẠNH (TỰ TIN >85%)")
        elif top1['score'] >= 12:
            st.warning("⚡ LỆNH: VÀO TIỀN VỪA PHẢI")
        else:
            st.info("⏳ LỆNH: ĐỢI NHỊP CẦU ĐẸP HƠN")
