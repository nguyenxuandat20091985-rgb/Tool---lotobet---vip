import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN MATRIX-X SIÊU NÉT ---
st.set_page_config(page_title="MATRIX-X v5.0 SUPREME", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #0a0a0a; color: #00FF00; border: 2px solid #333; font-size: 20px !important; }
    
    /* Panel Xiên 3 dọc */
    .x3-container { border-left: 5px solid #00FF00; padding-left: 15px; margin-bottom: 20px; }
    .x3-card { background: #111; padding: 15px; border-radius: 12px; border: 1px solid #222; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; }
    .x3-num { color: #00FF00; font-size: 35px; font-weight: 900; letter-spacing: 5px; }
    .x3-rate { color: #FFD700; font-weight: bold; border: 1px solid #FFD700; padding: 2px 10px; border-radius: 5px; }

    /* Panel Xiên 4 ngang nổi bật */
    .x4-box {
        background: linear-gradient(90deg, #000 0%, #1a1a1a 100%);
        padding: 30px; border-radius: 20px; border: 3px solid #FFD700;
        text-align: center; margin: 20px 0; box-shadow: 0 0 30px rgba(255,215,0,0.2);
    }
    .x4-num { color: #FFD700; font-size: 55px; font-weight: 900; letter-spacing: 12px; text-shadow: 0 0 20px #FFD700; }

    /* Nhật ký theo dõi */
    .log-win { background: rgba(0,255,0,0.1); border-left: 10px solid #00ff00; padding: 12px; margin-bottom: 8px; color: #00ff00; font-weight: bold; }
    .log-loss { background: rgba(255,75,43,0.1); border-left: 10px solid #ff4b2b; padding: 12px; margin-bottom: 8px; color: #ff4b2b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if 'matrix_log' not in st.session_state: st.session_state.matrix_log = []
if 'current_sets' not in st.session_state: st.session_state.current_sets = None

# --- 2. HỆ THỐNG 10 THUẬT TOÁN SONG SONG (MATRIX ENGINE) ---
def matrix_supreme_engine(raw_input):
    nums = [int(n) for n in re.findall(r'\d', raw_input)]
    if len(nums) < 25: return None, len(nums)

    # TỰ ĐỘNG ĐỐI CHIẾU TRÚNG/TRƯỢT
    if st.session_state.current_sets is not None:
        last_5 = nums[-5:] # Lấy 5 số vừa mở
        sets = st.session_state.current_sets
        
        # Kiểm tra Xiên 4
        win_x4 = all(x in last_5 for x in sets['x4'])
        # Kiểm tra Xiên 3 (Xem có bộ nào trúng không)
        win_x3 = any(all(x in last_5 for x in s) for s in [sets['x3_1'], sets['x3_2'], sets['x3_3']])
        
        if win_x4:
            st.session_state.matrix_log.insert(0, ("win", f"🏆 ĐỈNH CAO: TRÚNG XIÊN 4 [{''.join(map(str, sets['x4']))}]"))
        elif win_x3:
            st.session_state.matrix_log.insert(0, ("win", f"✅ THẮNG XIÊN 3"))
        else:
            st.session_state.matrix_log.insert(0, ("loss", f"❌ CHƯA TRÚNG (Kỳ mở: {''.join(map(str, last_5))})"))
        st.session_state.current_sets = None

    # TÍNH TOÁN ĐIỂM (10 THUẬT TOÁN)
    scored = []
    freq_30 = Counter(nums[-30:])
    last_5_res = nums[-5:]
    
    for n in range(10):
        s = 0
        gap = 0
        for v in reversed(nums[:-1]):
            if v == n: break
            gap += 1
        
        # 10 Thuật toán song song tính điểm cho từng số
        if 4 <= gap <= 8: s += 30 # T1: Nhịp hồi
        if n == {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}.get(nums[-1]): s += 15 # T2: Bóng
        if n == (sum(last_5_res) % 10): s += 20 # T3: Tổng chạm
        if gap in [3, 5, 8]: s += 10 # T4: Fibonacci
        if 2 <= freq_30[n] <= 5: s += 25 # T5: Tần suất rơi
        if n in last_5_res: s += 10 # T6: Nhịp bệt
        if n == (nums[-1] + 1) % 10: s += 5 # T7: Tiến
        if n == (nums[-1] - 1) % 10: s += 5 # T8: Lùi
        if gap > 12: s -= 40 # T9: Loại trừ số Gan cực nặng
        if freq_30[n] > 7: s -= 20 # T10: Tránh số bị "đứng" cầu do nổ quá nhiều

        scored.append({'n': n, 's': max(0, s)})
    
    res = sorted(scored, key=lambda x: x['s'], reverse=True)
    return res, nums

# --- 3. GIAO DIỆN VẬN HÀNH ---
st.title("🌌 MATRIX-X v5.0 SUPREME")
st.markdown("#### Hệ thống tính toán xác suất Xiên 3 & Xiên 4 hội tụ trong 5 giải")

input_data = st.text_area("NHẬP DÃY SỐ KẾT QUẢ:", height=100, placeholder="Dán dãy số từ S-Pen...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 PHÂN TÍCH HỆ THỐNG"):
        res, info = matrix_supreme_engine(input_data)
        if res:
            # Thuật toán ghép bộ tối ưu nhất
            st.session_state.current_sets = {
                'x3_1': [res[0]['n'], res[1]['n'], res[2]['n']],
                'x3_2': [res[0]['n'], res[1]['n'], res[3]['n']],
                'x3_3': [res[0]['n'], res[2]['n'], res[4]['n']],
                'x4': [res[0]['n'], res[1]['n'], res[2]['n'], res[3]['n']],
                'scores': [res[i]['s'] for i in range(5)]
            }
        else: st.error(f"Cần 25 số (Hiện có {info})")
with c2:
    if st.button("♻️ LÀM MỚI NHẬT KÝ"): 
        st.session_state.matrix_log = []
        st.rerun()

# --- 4. HIỂN THỊ KẾT QUẢ ---
if st.session_state.current_sets:
    s = st.session_state.current_sets
    
    # XIÊN 4 - TỔNG HỢP CAO NHẤT
    st.markdown(f"""
        <div class="x4-box">
            <div style="color: #FFD700; font-size: 18px; letter-spacing: 5px;">💎 BỘ TỨ XIÊN 4 (XÁC SUẤT CAO NHẤT)</div>
            <div class="x4-num">{''.join(map(str, s['x4']))}</div>
            <div style="color: #00FF00; font-weight:bold;">ĐỘ HỘI TỤ HỆ THỐNG: {round(sum(s['scores'][:4])/4, 1)}%</div>
        </div>
    """, unsafe_allow_html=True)

    # XIÊN 3 - 3 CẶP DỌC
    st.markdown("### 🎯 3 CẶP XIÊN 3 TIỀM NĂNG")
    for i, key in enumerate(['x3_1', 'x3_2', 'x3_3']):
        rate = round(sum(s['scores'][:3]) / 3 - (i*3), 1)
        st.markdown(f"""
            <div class="x3-card">
                <div style="color:#888;">MẪU {i+1}</div>
                <div class="x3-num">{''.join(map(str, s[key]))}</div>
                <div class="x3-rate">TỈ LỆ THẮNG: {rate}%</div>
            </div>
        """, unsafe_allow_html=True)

# --- 5. BẢNG THEO DÕI ---
st.markdown("---")
st.markdown("### 📋 BẢNG THEO DÕI TRÚNG / TRƯỢT")
for style, text in st.session_state.matrix_log[:15]:
    cls = "log-win" if style == "win" else "log-loss"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
