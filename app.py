import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN CHỐNG SOI (HIGH CONTRAST) ---
st.set_page_config(page_title="GHOST-PROTOCOL v6.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    /* Xiên 4: Hiển thị Vàng Gold rực rỡ */
    .ghost-box {
        background: #111; padding: 35px; border-radius: 25px; border: 5px solid #FFD700;
        text-align: center; margin-bottom: 25px; box-shadow: 0 0 40px rgba(255,215,0,0.5);
    }
    .ghost-num { color: #FFD700; font-size: 80px; font-weight: 900; letter-spacing: 15px; }
    
    /* Xiên 3: Trắng Đỏ cực nét theo yêu cầu */
    .x3-container { display: flex; gap: 10px; }
    .x3-card {
        flex: 1; background: #fff; padding: 20px; border-radius: 15px;
        text-align: center; border: 5px solid #ff0000;
    }
    .x3-val { color: #ff0000 !important; font-size: 45px !important; font-weight: 900 !important; }
    .x3-label { color: #000; font-weight: bold; }

    /* Nhật ký Trúng/Trượt */
    .log-win { background: rgba(0,255,0,0.2); border-left: 10px solid #00ff00; padding: 15px; margin: 10px 0; color: #00ff00; font-weight: bold; border-radius: 8px; }
    .log-loss { background: rgba(255,0,0,0.1); border-left: 10px solid #ff0000; padding: 15px; margin: 10px 0; color: #ff4b2b; font-weight: bold; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

if 'ghost_log' not in st.session_state: st.session_state.ghost_log = []
if 'last_ghost_set' not in st.session_state: st.session_state.last_ghost_set = None

# --- 2. THUẬT TOÁN GHOST (7 TẦNG + BÙ SAI SỐ) ---
def ghost_engine(data):
    nums = [int(n) for n in re.findall(r'\d', data)]
    if len(nums) < 35: return None, len(nums)

    # TỰ ĐỘNG KIỂM TRA (Fix lỗi trúng 2/4 nhưng vẫn báo thua)
    if st.session_state.last_ghost_set:
        last_5 = nums[-5:]
        s = st.session_state.last_ghost_set
        match_x4 = sum(1 for x in s['x4'] if x in last_5)
        match_x3 = any(sum(1 for x in x3 if x in last_5) >= 3 for x3 in [s['x3a'], s['x3b'], s['x3c']])
        
        res_text = "".join(map(str, last_5))
        if match_x4 == 4:
            st.session_state.ghost_log.insert(0, ("win", f"🏆 ĐỈNH CAO XIÊN 4! Giải: {res_text}"))
        elif match_x3:
            st.session_state.ghost_log.insert(0, ("win", f"✅ TRÚNG XIÊN 3! Giải: {res_text}"))
        else:
            st.session_state.ghost_log.insert(0, ("loss", f"❌ TRƯỢT (Trúng {match_x4}/4). Giải: {res_text}"))
        st.session_state.last_ghost_set = None

    # TÍNH TOÁN ĐIỂM HỘI TỤ
    scored = []
    freq = Counter(nums[-50:])
    for n in range(10):
        # Thuật toán nhịp trễ
        gap = 0
        for v in reversed(nums[:-1]):
            if v == n: break
            gap += 1
        
        points = 0
        if 4 <= gap <= 9: points += 35         # Nhịp rơi đẹp
        if n == (sum(nums[-5:]) % 10): points += 20 # Tổng chạm kỳ
        if 3 <= freq[n] <= 6: points += 25     # Tần suất ổn định
        if n == {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}.get(nums[-1]): points += 15 # Bóng
        if gap > 15: points -= 60              # LOẠI SỐ GAN (Nguyên nhân gây thua)
        
        scored.append({'n': n, 'p': max(0, points)})
    
    top = sorted(scored, key=lambda x: x['p'], reverse=True)
    return top, nums

# --- 3. GIAO DIỆN ---
st.title("👻 GHOST-PROTOCOL v6.0")
st.markdown("##### Chống soi nhịp - Tối ưu Xiên bao 5 Giải")

raw_input = st.text_area("NHẬP DỮ LIỆU S-PEN:", height=100)

if st.button("🚀 KÍCH HOẠT GHOST MODE"):
    res, info = ghost_engine(raw_input)
    if res:
        st.session_state.last_ghost_set = {
            'x4': [res[0]['n'], res[1]['n'], res[2]['n'], res[3]['n']],
            'x3a': [res[0]['n'], res[1]['n'], res[2]['n']],
            'x3b': [res[0]['n'], res[1]['n'], res[3]['n']],
            'x3c': [res[0]['n'], res[2]['n'], res[4]['n']]
        }
    else: st.error(f"Cần 35 số để kích hoạt Ghost Mode (Hiện có {info})")

if st.session_state.last_ghost_set:
    s = st.session_state.last_ghost_set
    st.markdown(f'<div class="ghost-box"><div style="color:#FFD700;font-size:18px;">💎 XIÊN 4 MẠNH NHẤT</div><div class="ghost-num">{"".join(map(str, s["x4"]))}</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="x3-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (col, key) in enumerate(zip([c1, c2, c3], ['x3a', 'x3b', 'x3c'])):
        with col:
            st.markdown(f'<div class="x3-card"><div class="x3-label">XIÊN 3 - MẪU {i+1}</div><div class="x3-val">{"".join(map(str, s[key]))}</div></div>', unsafe_allow_html=True)

st.markdown("---")
for style, txt in st.session_state.ghost_log[:15]:
    st.markdown(f'<div class="log-{style}">{txt}</div>', unsafe_allow_html=True)
