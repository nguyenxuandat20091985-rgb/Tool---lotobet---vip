import streamlit as st
import re
from collections import Counter
import numpy as np

# --- 1. GIAO DIỆN SIÊU CẤP - HIỂN THỊ RÕ NÉT TRÊN MOBILE ---
st.set_page_config(page_title="OMEGA QUANTUM v5.5", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTextArea textarea { background-color: #080808; color: #00FF00; border: 2px solid #00FF00; font-size: 22px !important; font-weight: bold; }
    
    /* Xiên 4 Hoàng Gia */
    .mega-box {
        background: linear-gradient(145deg, #000 0%, #111 100%);
        padding: 40px; border-radius: 30px; border: 5px solid #FFD700;
        text-align: center; margin-bottom: 25px; box-shadow: 0 0 50px rgba(255,215,0,0.4);
    }
    .mega-num { color: #FFD700; font-size: 85px; font-weight: 900; letter-spacing: 15px; text-shadow: 0 0 30px #FFD700; line-height: 1; }
    
    /* Xiên 3 Hiện đại */
    .x3-grid { display: flex; gap: 15px; margin-bottom: 20px; }
    .x3-item {
        flex: 1; background: #ffffff; padding: 25px; border-radius: 20px;
        text-align: center; border: 4px solid #ff0000;
    }
    .x3-val { color: #ff0000 !important; font-size: 45px !important; font-weight: 900 !important; letter-spacing: 3px; }
    .x3-label { color: #000; font-weight: bold; font-size: 16px; text-transform: uppercase; }

    /* Nhật ký Trúng/Trượt */
    .win-log { background: rgba(0,255,0,0.15); border-left: 12px solid #00ff00; padding: 15px; margin-bottom: 10px; color: #00ff00; font-weight: bold; border-radius: 10px; font-size: 18px; }
    .loss-log { background: rgba(255,0,0,0.1); border-left: 12px solid #ff0000; padding: 15px; margin-bottom: 10px; color: #ff0000; font-weight: bold; border-radius: 10px; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

if 'omega_log' not in st.session_state: st.session_state.omega_log = []
if 'omega_pending' not in st.session_state: st.session_state.omega_pending = None

# --- 2. THUẬT TOÁN 10 TẦNG + LOGIC ĐỐI CHỈNH ---
def quantum_engine(data):
    # Trích xuất số từ sảnh
    raw_nums = re.findall(r'\d', data)
    nums = [int(n) for n in raw_nums]
    if len(nums) < 30: return None, len(nums)

    # TỰ ĐỘNG CHECK KẾT QUẢ KỲ TRƯỚC
    if st.session_state.omega_pending:
        last_5 = nums[-5:]
        sets = st.session_state.omega_pending
        x4_match = sum(1 for x in sets['x4'] if x in last_5)
        x3_match = any(sum(1 for x in s if x in last_5) >= 3 for s in [sets['x3_1'], sets['x3_2'], sets['x3_3']])
        
        res_view = "".join(map(str, last_5))
        if x4_match == 4:
            st.session_state.omega_log.insert(0, ("win-log", f"🏆 SIÊU PHẨM XIÊN 4! Giải mở: {res_view}"))
        elif x3_match:
            st.session_state.omega_log.insert(0, ("win-log", f"✅ ĂN XIÊN 3! Giải mở: {res_view}"))
        else:
            st.session_state.omega_log.insert(0, ("loss-log", f"❌ TRƯỢT (Trúng {x4_match}/4 số). Giải mở: {res_view}"))
        st.session_state.omega_pending = None

    # TÍNH TOÁN 10 TẦNG PHÂN TÍCH
    scores = np.zeros(10)
    freq_50 = Counter(nums[-50:])
    last_val = nums[-1]
    
    for n in range(10):
        # 1. Tầng Gap (Nhịp hồi kỹ thuật)
        gap = 0
        for v in reversed(nums[:-1]):
            if v == n: break
            gap += 1
        if 4 <= gap <= 9: scores[n] += 35
        
        # 2. Tầng Bóng (Shadow Logic)
        if n == {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}.get(last_val): scores[n] += 20
        
        # 3. Tầng Tần suất (Poisson Distribution)
        if 3 <= freq_50[n] <= 6: scores[n] += 25
        
        # 4. Tầng Tổng chạm (Sum Modulo)
        if n == (sum(nums[-5:]) % 10): scores[n] += 15
        
        # 5. Tầng Fibonacci Nhịp
        if gap in [3, 5, 8, 13]: scores[n] += 10
        
        # 6. Tầng Đối xứng (Mirror)
        if n == (10 - last_val) % 10: scores[n] += 5

        # 7. Tầng Repeat (Bệt)
        if n in nums[-3:]: scores[n] += 12

        # 8. Tầng Cầu Tiến/Lùi
        if n == (last_val + 1) % 10 or n == (last_val - 1) % 10: scores[n] += 8

        # --- BỘ LỌC CHỐNG THUA (CRITICAL FILTERS) ---
        if gap > 14: scores[n] -= 50 # T9: Chặn số Gan cực mạnh
        if freq_50[n] > 9: scores[n] -= 30 # T10: Chặn số đã nổ quá nhiều (Sắp đổi nhịp)

    # Sắp xếp và chọn bộ số
    top_indices = np.argsort(scores)[::-1]
    return top_indices, nums

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🌑 OMEGA QUANTUM v5.5")
st.subheader("Hệ thống Phân tích Xiên bao 5 Giải")

input_area = st.text_area("DÁN DỮ LIỆU CẦU MỚI NHẤT:", height=100)

col_cmd1, col_cmd2 = st.columns(2)
with col_cmd1:
    if st.button("🔥 KÍCH HOẠT OMEGA"):
        top_n, info = quantum_engine(input_area)
        if top_n is not None:
            st.session_state.omega_pending = {
                'x4': [top_n[0], top_n[1], top_n[2], top_n[3]],
                'x3_1': [top_n[0], top_n[1], top_n[2]],
                'x3_2': [top_n[0], top_n[1], top_n[3]],
                'x3_3': [top_n[0], top_n[2], top_n[4]],
                'top': top_n
            }
        else: st.error(f"Dữ liệu yếu! Cần 30 số (Hiện có {info})")
with col_cmd2:
    if st.button("♻️ RESET"):
        st.session_state.omega_log = []
        st.session_state.omega_pending = None
        st.rerun()

# --- 4. HIỂN THỊ KẾT QUẢ XIÊN ---
if st.session_state.omega_pending:
    p = st.session_state.omega_pending
    
    st.markdown(f"""
        <div class="mega-box">
            <div style="color: #FFD700; font-size: 20px; letter-spacing: 5px; margin-bottom:15px;">💎 BỘ TỨ XIÊN 4 (XÁC SUẤT CAO NHẤT)</div>
            <div class="mega-num">{"".join(map(str, p['x4']))}</div>
            <div style="margin-top:20px; color:#00FF00; font-weight:bold; font-size:20px;">HỘI TỤ QUANTUM: TỐI ƯU</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 XIÊN 3 CHIẾN THUẬT (TRẮNG - ĐỎ)")
    c1, c2, c3 = st.columns(3)
    for i, (col, key) in enumerate(zip([c1, c2, c3], ['x3_1', 'x3_2', 'x3_3'])):
        with col:
            st.markdown(f"""
                <div class="x3-item">
                    <div class="x3-label">MẪU {i+1}</div>
                    <div class="x3-val">{"".join(map(str, p[key]))}</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📋 BẢNG THEO DÕI KẾT QUẢ")
for css_class, text in st.session_state.omega_log[:15]:
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)
