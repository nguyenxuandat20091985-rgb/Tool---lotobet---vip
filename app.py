import streamlit as st
import re
from collections import Counter
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN & STYLE ---
st.set_page_config(page_title="v5.2 ULTRA-4 FINAL", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; font-weight: bold; color: #495057; }
    .stTabs [aria-selected="true"] { color: #d9534f !important; border-bottom-color: #d9534f !important; }
    
    /* Khung kết quả chính */
    .result-card {
        background: #ffffff; padding: 30px; border-radius: 20px;
        border-left: 10px solid #d9534f; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center; margin: 10px 0;
    }
    .num-highlight { color: #d9534f; font-size: 80px; font-weight: 900; letter-spacing: 15px; }
    .report-win { color: #28a745; font-weight: bold; font-size: 20px; }
    .report-loss { color: #dc3545; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. HỆ THỐNG THUẬT TOÁN 6 TẦNG ---
def core_engine_v5(data):
    # Lọc dãy số
    raw_nums = [int(x) for x in re.findall(r'\d', data)]
    if len(raw_nums) < 5: return None
    
    last_kỳ = raw_nums[-5:] # 5 số của kỳ gần nhất
    freq = Counter(raw_nums[-30:]) # Tần suất 30 số gần đây
    shadow_map = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
    
    scored = {i: 0 for i in range(10)}
    for i in range(10):
        # Tầng 1: Bệt nhịp (Số vừa ra kỳ trước)
        if i in last_kỳ: scored[i] += 35
        # Tầng 2: Bóng đối xứng
        if any(i == shadow_map.get(x) for x in last_kỳ): scored[i] += 30
        # Tầng 3: Tần suất dày (Hot numbers)
        if freq[i] >= 3: scored[i] += 20
        # Tầng 4: Sát kép (Tiến lùi)
        if any(i == (x+1)%10 or i == (x-1)%10 for x in last_kỳ): scored[i] += 15
        # Tầng 5: Loại trừ số Gan (Số không ra trong 15 số gần nhất)
        if i not in raw_nums[-15:]: scored[i] -= 25
        # Tầng 6: Fibonacci nhịp nghỉ
        if i not in last_kỳ and i in raw_nums[-10:-5]: scored[i] += 10

    # Lấy 4 số rời rạc có điểm cao nhất
    top_4 = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:4]
    return [x[0] for x in top_4]

# --- 3. QUẢN LÝ TRẠNG THÁI ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 4. GIAO DIỆN CÁC TAB ---
st.title("🎯 v5.2 ULTRA-4: CHIẾN THUẬT HỒI PHỤC")
st.write("---")

tab_soi, tab_report, tab_settings = st.tabs(["🔍 SOI CẦU ĐA TẦNG", "📊 BÁO CÁO KẾT QUẢ", "⚙️ HƯỚNG DẪN"])

with tab_soi:
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.markdown("### 📥 Nhập kết quả")
        input_data = st.text_area("Dán chuỗi số kỳ gần nhất (S-Pen/OCR):", height=150)
        
        if st.button("🚀 KÍCH HOẠT HỆ THỐNG"):
            res = core_engine_v5(input_data)
            if res:
                st.session_state.current_res = res
                st.success("Đã tính toán xong nhịp cầu!")
            else:
                st.error("Dữ liệu không hợp lệ hoặc quá ngắn.")

    with col_out:
        if 'current_res' in st.session_state:
            nums = st.session_state.current_res
            st.markdown(f"""
                <div class="result-card">
                    <p style="color:#666; font-weight:bold;">4 SỐ RỜI (VỐN 40K)</p>
                    <div class="num-highlight">{".".join(map(str, nums))}</div>
                    <p style="margin-top:10px; color:#d9534f;">🎯 Mục tiêu: Trúng 1 nháy lãi 58k</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Form báo cáo nhanh
            st.write("---")
            st.markdown("##### 📝 Xác nhận kết quả kỳ này:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ TRÚNG (WIN)"):
                    st.session_state.history.append({"Số": nums, "KQ": "WIN", "Tiền": "+58k"})
                    st.toast("Chúc mừng! Đã lưu kết quả.")
            with c2:
                if st.button("❌ TRƯỢT (LOSS)"):
                    st.session_state.history.append({"Số": nums, "KQ": "LOSS", "Tiền": "-40k"})
                    st.toast("Không sao, giữ bình tĩnh chờ nhịp sau.")

with tab_report:
    st.markdown("### 📈 Nhật ký chiến đấu")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
        
        # Tính toán tổng kết
        wins = len([x for x in st.session_state.history if x['KQ'] == 'WIN'])
        losses = len([x for x in st.session_state.history if x['KQ'] == 'LOSS'])
        st.sidebar.markdown(f"### 📊 Tổng kết:")
        st.sidebar.success(f"Thắng: {wins}")
        st.sidebar.error(f"Thua: {losses}")
    else:
        st.info("Chưa có dữ liệu báo cáo.")

with tab_settings:
    st.markdown("""
    ### 🛡️ Nguyên tắc vàng bản v5.2:
    1. **Vốn:** Luôn đi đều tay 40k (10k mỗi số). Tuyệt đối không gấp thếp khi đang thua.
    2. **Dữ liệu:** Dán kết quả của ít nhất 3 kỳ gần nhất để thuật toán bắt nhịp bệt chính xác.
    3. **Dừng chơi:** - Nghỉ 10 phút nếu thắng liên tiếp 3 kỳ.
        - Dừng ngay nếu thua 2 kỳ liên tiếp (Cầu đang loạn).
    4. **Chính xác:** Ưu tiên dán số qua OCR để tránh nhập sai số làm hỏng thuật toán.
    """)
