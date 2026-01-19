import streamlit as st
import re
from collections import Counter
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN SIÊU TƯƠNG PHẢN (MÀU ĐỎ TRẮNG) ---
st.set_page_config(page_title="v5.2 ULTRA-4 FINAL", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: bold; color: #333; }
    .stTabs [aria-selected="true"] { color: #d9534f !important; border-bottom: 4px solid #d9534f !important; }
    
    /* Khung hiển thị 4 số rời cực lớn */
    .solo-card {
        background: #ffffff; padding: 40px; border-radius: 25px;
        border: 10px solid #d9534f; box-shadow: 0 15px 40px rgba(217, 83, 79, 0.2);
        text-align: center; margin: 20px 0;
    }
    .solo-nums { color: #d9534f !important; font-size: 100px !important; font-weight: 900; letter-spacing: 20px; }
    .target-text { color: #555; font-size: 24px; font-weight: bold; margin-top: 15px; }
    
    /* Nút bấm báo cáo */
    .stButton>button { width: 100%; border-radius: 12px; font-weight: bold; height: 3em; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. HỆ THỐNG PHÂN TÍCH 6 TẦNG (CHUYÊN KUBET) ---
def analyze_engine(data):
    # Trích xuất tất cả số từ văn bản dán vào (OCR)
    nums = [int(x) for x in re.findall(r'\d', data)]
    if len(nums) < 5: return None
    
    last_kỳ = nums[-5:] # Lấy kết quả 5 hàng số gần nhất
    freq = Counter(nums[-40:]) # Tần suất trong 40 số gần đây
    shadow = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
    
    scores = {i: 0 for i in range(10)}
    for i in range(10):
        # T1: Nhịp Bệt (Số vừa ra)
        if i in last_kỳ: scores[i] += 40
        # T2: Bóng đối xứng
        if any(i == shadow.get(x) for x in last_kỳ): scores[i] += 35
        # T3: Tần suất (Hot numbers)
        if freq[i] >= 3: scores[i] += 25
        # T4: Cầu Sát kép (Tiến lùi)
        if any(i == (x+1)%10 or i == (x-1)%10 for x in last_kỳ): scores[i] += 20
        # T5: Loại trừ số Gan
        if i not in nums[-15:]: scores[i] -= 30
        # T6: Nhịp nghỉ Fibonacci
        if i not in last_kỳ and i in nums[-10:-5]: scores[i] += 15

    # Lấy 4 số điểm cao nhất
    res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
    return [x[0] for x in res]

# --- 3. QUẢN LÝ LỊCH SỬ LÃI LỖ ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center; color: #d9534f;'>🎯 v5.2 ULTRA-4 FINAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: bold;'>Chiến thuật: 4 Số Rời (40k) - Hồi Phục Vốn</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 SOI CẦU ĐA TẦNG", "📈 BÁO CÁO LÃI LỖ", "📜 HƯỚNG DẪN"])

with tab1:
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.markdown("### 📥 Nhập Kết Quả (OCR)")
        raw_input = st.text_area("Dán chuỗi số kỳ vừa mở (Ví dụ: 2 5 4 7 5):", height=200)
        
        if st.button("🚀 KÍCH HOẠT QUÉT"):
            res = analyze_engine(raw_input)
            if res:
                st.session_state.current_4 = res
                st.success("Đã tìm ra nhịp cầu tốt nhất!")
            else:
                st.error("Dữ liệu quá ngắn. Hãy dán đủ 5 số mở thưởng.")

    with col_r:
        if 'current_4' in st.session_state:
            s = st.session_state.current_4
            st.markdown(f"""
                <div class="solo-card">
                    <div class="card-label" style="font-weight:bold; color:#666;">DÀN 4 SỐ RỜI (40K)</div>
                    <div class="solo-nums">{s[0]} . {s[1]} . {s[2]} . {s[3]}</div>
                    <div class="target-text">🎯 Mục tiêu: Trúng 1 nháy lãi 58k</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 📝 Xác nhận kỳ này:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ TRÚNG (WIN)"):
                    st.session_state.history.append({"Số": s, "KQ": "WIN", "Tiền": "+58k"})
                    st.balloons()
            with c2:
                if st.button("❌ TRƯỢT (LOSS)"):
                    st.session_state.history.append({"Số": s, "KQ": "LOSS", "Tiền": "-40k"})

with tab2:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        
        # Tính toán tài chính
        wins = len([x for x in st.session_state.history if x['KQ'] == 'WIN'])
        losses = len([x for x in st.session_state.history if x['KQ'] == 'LOSS'])
        total_profit = (wins * 58) - (losses * 40)
        
        st.sidebar.metric("TỔNG LỢI NHUẬN", f"{total_profit}k", delta=f"{total_profit}k")
    else:
        st.info("Chưa có báo cáo kỳ nào.")

with tab3:
    st.markdown("""
    ### 🛡️ Cách Chơi 2 Số 5 Tinh (4 Số Rời):
    * **Bước 1:** Lấy 4 số rời từ Tool (VD: 0, 1, 2, 4).
    * **Bước 2:** Ghép cặp (0,1) và (2,4). Đặt mỗi cặp 20k -> Tổng 40k.
    * **Bước 3:** Chỉ cần 1 cặp xuất hiện cả 2 số trong kết quả là có lãi 58k.
    
    **⚠️ Lưu ý bảo vệ vốn:**
    - Trượt 2 kỳ liên tiếp: Dừng chơi 15 phút.
    - Thắng đủ chỉ tiêu: Rút tiền ngay, không tham.
    """)
