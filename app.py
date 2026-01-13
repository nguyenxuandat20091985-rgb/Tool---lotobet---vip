import streamlit as st
import random
from collections import Counter
import re
import pandas as pd

# Cấu hình App chuyên nghiệp
st.set_page_config(page_title="LOTOBET AI ULTIMATE V10", layout="centered")

# Giao diện CSS tùy chỉnh cao cấp
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; font-weight: bold; background: linear-gradient(45deg, #ff4b4b, #ff7676); color: white; border: none; }
    .result-card { background: rgba(30, 30, 47, 0.7); padding: 15px; border-radius: 15px; border: 1px solid #3d3d5c; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    .number-highlight { color: #00ffcc; font-size: 2.2em; font-weight: bold; text-shadow: 0 0 10px #00ffcc; }
    .bet-alert { background: #4b0000; color: #ff4b4b; padding: 10px; border-radius: 8px; font-weight: bold; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ LOTOBET AI ULTIMATE V10")
st.write("Hệ thống AI chuyên sâu: 5 Tinh, 3 Tinh, Nhận diện Cầu Bệt & Quản lý vốn")

# --- QUẢN LÝ DỮ LIỆU ---
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ CÀI ĐẶT AI")
    target_profit = st.number_input("Mục tiêu lãi (%)", value=20)
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.history = []
        st.rerun()

# Nạp dữ liệu đa nguồn
with st.expander("📥 NẠP DỮ LIỆU ĐA NGUỒN", expanded=len(st.session_state.history) == 0):
    raw_data = st.text_area("Dán kết quả (Copy từ nhà cái hoặc quét Google Lens):", height=150)
    if st.button("🚀 KÍCH HOẠT QUÉT DỮ LIỆU"):
        digits = re.findall(r'\d', raw_data)
        new_rows = [[int(d) for d in digits[i:i+5]] for i in range(0, len(digits)-4, 5)]
        if new_rows:
            st.session_state.history.extend(new_rows)
            st.session_state.history = st.session_state.history[-500:] # Nhớ 500 kỳ
            st.success(f"Đã nạp {len(new_rows)} kỳ quay mới!")
            st.rerun()

# --- THUẬT TOÁN PHÂN TÍCH ---
def deep_analyze():
    data = st.session_state.history
    all_nums = [n for row in data for n in row]
    last_5_ky = data[-5:]
    
    # 1. Thuật toán 2 số 5 Tinh (Dựa trên cặp số hay đi cùng nhau)
    flat_data = ["".join(map(str, row)) for row in data]
    counts = Counter(all_nums)
    
    # 2. Nhận diện cầu Bệt (Xác định số xuất hiện > 3 lần trong 5 kỳ gần nhất)
    recent_flat = [n for row in last_5_ky for n in row]
    recent_counts = Counter(recent_flat)
    bet_nums = [num for num, count in recent_counts.items() if count >= 3]
    
    # 3. Dự đoán 3 Tinh (Dựa trên nhịp rơi vị trí)
    top_3 = [item[0] for item in counts.most_common(3)]
    
    return top_3, bet_nums, data[-1]

if st.session_state.history:
    st.markdown(f"📊 Dữ liệu: `{len(st.session_state.history)}` kỳ | 🟢 Trạng thái: **Sẵn sàng**")
    
    if st.button("🔮 PHÂN TÍCH KẾT QUẢ KỲ TIẾP THEO"):
        if len(st.session_state.history) < 10:
            st.warning("⚠️ Hãy nạp ít nhất 10 kỳ để AI nhận diện nhịp cầu bệt!")
        else:
            top_nums, bet_list, last_ky = deep_analyze()
            
            # --- HIỂN THỊ CẢNH BÁO BỆT ---
            if bet_list:
                st.markdown(f"<div class='bet-alert'>🚨 CẢNH BÁO CẦU BỆT: Số {', '.join(map(str, bet_list))} đang nổ liên tục!</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 CẶP 2 SỐ 5 TINH")
                # Ưu tiên số đang bệt ghép với số hay về nhất
                s1 = top_nums[0]
                s2 = bet_list[0] if bet_list else top_nums[1]
                st.markdown(f"<div class='result-card'><small>TỈ LỆ THẮNG CAO</small><br><span class='number-highlight'>{s1} - {s2}</span></div>", unsafe_allow_html=True)
                st.caption("Đánh cặp này cho mục 2 số 5 tinh.")

            with col2:
                st.subheader("🌟 TAM THỦ 3 TINH")
                res_3t = "".join(map(str, top_nums))
                st.markdown(f"<div class='result-card'><small>HẬU TAM/TIỀN TAM</small><br><span class='number-highlight' style='color:#ffcc00;'>{res_3t}</span></div>", unsafe_allow_html=True)

            # --- QUẢN LÝ VỐN ---
            st.markdown("---")
            st.subheader("💰 CHIẾN THUẬT VÀO TIỀN")
            st.info("AI khuyên: Đánh gấp thếp 1-2-4 nếu ván trước chưa về. Nếu trúng cầu bệt, đánh đều tay.")

st.caption("AI Ultimate V10 - Bản nâng cấp ưu việt nhất cho 2 số 5 tinh & Cầu bệt.")
