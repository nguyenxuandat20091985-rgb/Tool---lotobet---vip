import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import Counter
import datetime

# --- CẤU HÌNH GIAO DIỆN SUPREME v4.3 (TỐI ƯU POP-UP) ---
st.set_page_config(page_title="AI SUPREME v4.3", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Giao diện Robot Supreme */
    .supreme-container {
        background: linear-gradient(145deg, #1a1a1a, #000000);
        border: 2px solid #FFD700; /* Màu vàng Gold đẳng cấp */
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.2);
    }
    
    .robot-head {
        width: 100px;
        filter: drop-shadow(0 0 15px #FFD700);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .status-bar {
        font-size: 10px;
        color: #00FFC2;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #333;
    }

    .main-number {
        font-size: 65px;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
        margin: 0;
    }

    .score-badge {
        background: #FF3131;
        color: white;
        padding: 2px 10px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Thu nhỏ các nút bấm */
    .stButton>button { height: 40px; border-radius: 20px; background: #FFD700; color: black; font-weight: bold; border: none; }
    header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- THUẬT TOÁN SUPREME ENGINE ---
def calculate_supreme(data):
    if len(data) < 5: return None
    matrix = [list(map(int, list(s))) for s in data]
    
    # 1. Trọng số theo thời gian (Số mới quan trọng hơn)
    weighted_counts = Counter()
    for i, row in enumerate(matrix):
        w = (i + 1) / len(matrix)
        for n in row: weighted_counts[n] += w
            
    # 2. Phân tích Nhịp (Gap)
    gaps = {n: 0 for n in range(10)}
    for n in range(10):
        for row in reversed(matrix):
            if n in row: break
            gaps[n] += 1
            
    # 3. Tính Điểm Nổ AI (AI Explosion Score)
    scores = {}
    for n in range(10):
        # Công thức: Tần suất + (Ưu tiên số vừa về hoặc về đều)
        scores[n] = round((weighted_counts[n] * 5) + (10 / (gaps[n] + 1)), 2)
        
    best_n = max(scores, key=scores.get)
    
    # Xiên 2 Correlation
    all_pairs = []
    for row in matrix: all_pairs.extend(combinations(sorted(row), 2))
    best_pair = Counter(all_pairs).most_common(1)[0]
    
    return best_n, scores[best_n], best_pair[0], gaps

# --- CHƯƠNG TRÌNH CHÍNH ---
if 'history' not in st.session_state: st.session_state.history = []

st.markdown("""
    <div class="supreme-container">
        <div class="status-bar">● CORE v4.3 SUPREME ACTIVE</div>
        <img src="https://cdn-icons-png.flaticon.com/512/6298/6298127.png" class="robot-head">
    </div>
    """, unsafe_allow_html=True)

# Nhập liệu cực gọn
with st.expander("📥 NẠP DỮ LIỆU"):
    raw_input = st.text_area("Dán chuỗi số:", height=60)
    if st.button("KÍCH HOẠT HỆ THỐNG"):
        clean = re.findall(r'\b\d{5}\b', raw_input)
        if clean:
            st.session_state.history = clean
            st.rerun()

if st.session_state.history:
    best_n, ai_score, pair, gaps = calculate_supreme(st.session_state.history)
    
    st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:12px; color:#aaa;">BẠCH THỦ TIỀM NĂNG</div>
            <p class="main-number">{best_n}</p>
            <span class="score-badge">ĐIỂM NỔ: {ai_score}</span>
        </div>
        """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='prediction-box' style='padding:10px;'><b>XIÊN 2</b><br><span style='color:#FFD700; font-size:20px;'>{pair[0]}-{pair[1]}</span></div>", unsafe_allow_html=True)
    with c2:
        # Cảnh báo dựa trên Điểm Nổ
        status = "VÀO TIỀN" if ai_score > 15 else "THEO DÕI"
        st.markdown(f"<div class='prediction-box' style='padding:10px;'><b>LỆNH</b><br><span style='color:#00FFC2; font-size:18px;'>{status}</span></div>", unsafe_allow_html=True)

    if st.button("🔄 RESET"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("🤖 Robot v4.3 đang chờ dữ liệu...")
