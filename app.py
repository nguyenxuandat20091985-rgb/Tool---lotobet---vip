import streamlit as st
import re
from collections import Counter
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP ---
st.set_page_config(page_title="v5.2 ULTRA-4 FINAL", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; font-weight: bold; }
    /* Khung hiển thị 4 số rời */
    .solo-container {
        background: white; padding: 25px; border-radius: 20px;
        border-top: 10px solid #d9534f; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center; margin-bottom: 20px;
    }
    .solo-numbers { color: #d9534f; font-size: 70px; font-weight: 900; letter-spacing: 10px; }
    .card-label { color: #555; font-size: 20px; font-weight: bold; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THUẬT TOÁN PHÂN TÍCH 6 TẦNG ---
def engine_v5_2(data):
    # Trích xuất tất cả các con số đơn lẻ từ chuỗi dữ liệu dán vào
    nums = [int(x) for x in re.findall(r'\d', data)]
    if len(nums) < 10: return None
    
    last_results = nums[-10:] # Lấy kết quả 2 kỳ gần nhất (mỗi kỳ 5 số)
    freq = Counter(nums[-50:]) # Tần suất trong 50 số gần nhất
    shadow = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
    
    scored = {i: 0 for i in range(10)}
    for i in range(10):
        # T1: Nhịp bệt (Số vừa xuất hiện)
        if i in last_results[-5:]: scored[i] += 40
        # T2: Bóng đối xứng ngũ hành
        if any(i == shadow.get(x) for x in last_results[-5:]): scored[i] += 35
        # T3: Tần suất dày (Hot)
        if freq[i] >= 4: scored[i] += 25
        # T4: Cầu tiến lùi (Sát kép)
        if any(i == (x+1)%10 or i == (x-1)%10 for x in last_results[-5:]): scored[i] += 20
        # T5: Loại trừ số Gan (Lâu không ra)
        if i not in nums[-20:]: scored[i] -= 30
        # T6: Nhịp nghỉ Fibonacci
        if i not in last_results[-5:] and i in last_results[-10:-5]: scored[i] += 15

    # Lấy 4 số điểm cao nhất để tạo cặp
    final_4 = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:4]
    return [x[0] for x in final_4]

# --- 3. QUẢN LÝ LỊCH SỬ & TÀI CHÍNH ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- 4. GIAO DIỆN ĐIỀU KHIỂN ---
st.title("🛡️ v5.2 ULTRA-4: PHÂN TÍCH KUBET")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 SOI CẦU 2 SỐ 5 TINH", "📊 BÁO CÁO LÃI LỖ", "📜 HƯỚNG DẪN CHIẾN THUẬT"])

with tab1:
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.markdown("### 📥 Nhập Dữ Liệu")
        raw_text = st.text_area("Dán kết quả các kỳ gần nhất (OCR):", height=200, placeholder="Ví dụ: 2 5 4 7 5 ...")
        
        if st.button("🔥 PHÂN TÍCH ĐA TẦNG"):
            res = engine_v5_2(raw_text)
            if res:
                st.session_state.current_4 = res
                st.success("✅ Đã tìm ra 4 số tiềm năng nhất!")
            else:
                st.error("❌ Dữ liệu quá ngắn hoặc không đúng định dạng.")

    with col_r:
        if 'current_4' in st.session_state:
            s = st.session_state.current_4
            st.markdown(f"""
                <div class="solo-container">
                    <div class="card-label">Dàn 4 Số Rời (Vốn 40k)</div>
                    <div class="solo-numbers">{s[0]} . {s[1]} . {s[2]} . {s[3]}</div>
                    <p style="color: #666;">Ghép cặp: ({s[0]},{s[1]}) - ({s[2]},{s[3]})</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 📝 Xác nhận kết quả kỳ vừa đánh:")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ THẮNG (WIN)"):
                    st.session_state.logs.append({"Dàn": s, "KQ": "WIN", "Lợi nhuận": "+58,000"})
                    st.balloons()
            with c2:
                if st.button("❌ THUA (LOSS)"):
                    st.session_state.logs.append({"Dàn": s, "KQ": "LOSS", "Lợi nhuận": "-40,000"})

with tab2:
    st.markdown("### 📈 Nhật Ký Chiến Đấu")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df, use_container_width=True)
        
        # Thống kê tổng quát
        total_win = len([x for x in st.session_state.logs if x['KQ'] == 'WIN'])
        total_loss = len([x for x in st.session_state.logs if x['KQ'] == 'LOSS'])
        net_profit = (total_win * 58) - (total_loss * 40)
        
        st.metric("TỔNG LỢI NHUẬN", f"{net_profit},000 VNĐ", delta=f"{net_profit}k")
    else:
        st.info("Chưa có lịch sử cược.")

with tab3:
    st.info("Chế độ: 2 số 5 tinh - Không cố định")
    st.markdown("""
    **1. Cách đặt cược:**
    * Chọn 4 số từ Tool (Ví dụ: 0, 1, 2, 4).
    * Ghép thành 2 cặp: (0, 1) và (2, 4).
    * Mỗi cặp vào 20k -> Tổng 40k.
    
    **2. Điều kiện thắng:**
    * Chỉ cần 1 cặp xuất hiện đủ cả 2 số trong 5 hàng mở thưởng là trúng.
    * Tỉ lệ thưởng cao giúp bạn lãi ngay 58k chỉ với 1 nháy thắng.
    
    **3. Kỷ luật bảo toàn vốn:**
    * Thua 2 kỳ liên tiếp: **Dừng chơi 15 phút**.
    * Thắng đủ chỉ tiêu: **Rút lãi ngay**.
    """)
