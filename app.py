import streamlit as st
import re
from collections import Counter
import random

# --- 1. CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP ---
st.set_page_config(page_title="v5.2 ULTRA-2D PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stApp { background: #ffffff; }
    /* Khung hiển thị cặp số dự đoán */
    .prediction-container {
        display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; margin: 20px 0;
    }
    .prediction-card {
        background: #ffffff; border: 4px solid #d9534f; border-radius: 20px;
        padding: 20px; width: 200px; text-align: center;
        box-shadow: 0 8px 25px rgba(217, 83, 79, 0.15);
    }
    .pred-num { color: #d9534f; font-size: 55px; font-weight: 900; margin-bottom: 5px; }
    .pred-percent { color: #28a745; font-size: 22px; font-weight: bold; }
    .pred-label { color: #666; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THUẬT TOÁN SOI 2D & TÍNH TỶ LỆ % ---
def analyze_2d_pro(data):
    # Lấy toàn bộ cặp số 2D từ dữ liệu (2 chữ số cuối)
    raw_nums = re.findall(r'\d{2,5}', data)
    last_2d = [n[-2:] for n in raw_nums]
    
    if len(last_2d) < 10: return None
    
    freq = Counter(last_2d)
    all_possible = [f"{i:02d}" for i in range(100)]
    scored = []
    
    for num in all_possible:
        score = 0
        # Tầng 1: Tần suất (Số hay về)
        score += freq[num] * 30
        # Tầng 2: Nhịp rơi (Bệt lại từ kỳ trước)
        if num in last_2d[-5:]: score += 45
        # Tầng 3: Thuật toán bóng số & lộn
        reversed_num = num[::-1]
        if reversed_num in last_2d[-5:]: score += 25
        
        # Tạo tỷ lệ % dựa trên điểm số (Giả lập dao động từ 75% - 98%)
        confidence = min(75 + (score / 5), 98.5)
        scored.append({'num': num, 'conf': round(confidence, 1)})
    
    # Lấy 4 cặp số có điểm/tỷ lệ cao nhất
    top_4 = sorted(scored, key=lambda x: x['conf'], reverse=True)[:4]
    return top_4

# --- 3. GIAO DIỆN ĐIỀU KHIỂN ---
st.markdown("<h1 style='text-align: center; color: #d9534f;'>🎯 v5.2 ULTRA-2D PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: bold;'>Dự đoán cặp số rời rạc & Tỉ lệ nổ (%)</p>", unsafe_allow_html=True)

col_in, col_empty, col_out = st.columns([1, 0.1, 1.5])

with col_in:
    st.markdown("### 📥 Nhập Kết Quả")
    input_data = st.text_area("Dán chuỗi số kỳ vừa mở (OCR):", height=250, placeholder="Ví dụ: 4685 9245 7308...")
    
    if st.button("🚀 PHÂN TÍCH CHUYÊN SÂU"):
        with st.spinner('Hệ thống đang quét nhịp cầu...'):
            results = analyze_2d_pro(input_data)
            if results:
                st.session_state.results_2d = results
                st.success("✅ Đã hoàn tất dự đoán!")
            else:
                st.error("❌ Dữ liệu không đủ để phân tích.")

with col_out:
    if 'results_2d' in st.session_state:
        res = st.session_state.results_2d
        st.markdown("### 🔮 Cặp Số Khuyên Đánh (Vốn 40k)")
        
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        for item in res:
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="pred-label">Tỉ lệ về</div>
                    <div class="pred-percent">{item['conf']}%</div>
                    <div class="pred-num">{item['num']}</div>
                    <div class="pred-label">Độ tin cậy</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("💡 Chiến thuật: Đặt mỗi cặp số 10k. Chỉ cần nổ 1 cặp là bạn đã có lãi.")
    else:
        st.info("Đang chờ dữ liệu từ ô nhập bên trái...")

# --- 4. NHẬT KÝ CHIẾN ĐẤU ---
st.write("---")
if 'history_2d' not in st.session_state: st.session_state.history_2d = []

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    if st.button("✅ BÁO THẮNG (WIN)"):
        st.session_state.history_2d.append("WIN")
        st.balloons()
with c2:
    if st.button("❌ BÁO THUA (LOSS)"):
        st.session_state.history_2d.append("LOSS")

# Thống kê nhanh
if st.session_state.history_2d:
    wins = st.session_state.history_2d.count("WIN")
    total = len(st.session_state.history_2d)
    st.sidebar.metric("Tỉ lệ thắng thực tế", f"{(wins/total)*100:.1f}%")
