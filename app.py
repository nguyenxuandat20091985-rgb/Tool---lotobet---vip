import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN FIX CỨNG LỖI HIỂN THỊ ---
st.set_page_config(page_title="AI MATRIX v4.6 FIX", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTextArea textarea { background-color: #050505; color: #00FF00; border: 1px solid #333; }
    
    /* Box Bạch Thủ */
    .bt-box {
        background: #0a0a0a; padding: 20px; border-radius: 15px; 
        border: 2px solid #00FF00; text-align: center; margin-bottom: 15px;
    }
    .bt-val { font-size: 70px; color: #00FF00; font-weight: bold; }
    
    /* BOX XIÊN - ĐÃ FIX MÀU CHỮ TRẮNG/VÀNG CHO RÕ */
    .xien-container { display: flex; gap: 10px; margin-bottom: 20px; }
    .xien-box {
        flex: 1; background: #151515; padding: 15px; border-radius: 12px;
        border: 1px solid #444; text-align: center;
    }
    .label-xien { color: #FFD700; font-size: 14px; font-weight: bold; margin-bottom: 5px; }
    .val-xien { color: #FFFFFF !important; font-size: 26px !important; font-weight: bold !important; opacity: 1 !important; }

    .status-win { color: #00ff00; font-weight: bold; background: rgba(0,255,0,0.1); padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .status-loss { color: #ff4b2b; font-weight: bold; background: rgba(255,75,43,0.1); padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'saved_res' not in st.session_state: st.session_state.saved_res = None

# --- 2. THUẬT TOÁN MATRIX 6 LUỒNG ---
def analyze_matrix_v2(raw):
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # Check thắng thua tự động
    if st.session_state.last_pred is not None:
        if st.session_state.last_pred in nums[-5:]:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 15: return None, nums

    counts = Counter(nums)
    last_val = nums[-1]
    last_5 = nums[-5:]
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i

    scored = []
    for n in range(10):
        s = 0
        gap = (len(nums) - 1) - last_pos[n]
        # Tổng hợp 6 thuật toán
        if 4 <= gap <= 9: s += 35 # Nhịp hồi
        if n == (sum(last_5) % 10): s += 20 # Tổng chạm
        if n == {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}.get(last_val): s += 15 # Bóng
        s += (counts[n] / len(nums)) * 40 # Tần suất
        if n in last_5: s += 10 # Bệt
        if gap > 15: s -= 40 # Gan
        scored.append({'n': n, 's': round(max(0, s), 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. HIỂN THỊ ---
st.title("⚡ MATRIX v4.6 PRO")
input_text = st.text_area("Dán cầu S-Pen:", height=80, label_visibility="collapsed")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🚀 PHÂN TÍCH MATRIX"):
        res, clean_nums = analyze_matrix_v2(input_text)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.saved_res = {'res': res, 'nums': clean_nums}
        else: st.error("Cần thêm dữ liệu!")
with col_btn2:
    if st.button("🔄 LÀM MỚI"): st.session_state.clear(); st.rerun()

if st.session_state.saved_res:
    r = st.session_state.saved_res['res']
    
    # BẠCH THỦ
    st.markdown(f"""
        <div class="bt-box">
            <div style="color:#888; font-size:14px;">BẠCH THỦ MATRIX</div>
            <div class="bt-val">{r[0]['n']}</div>
            <div style="color:#ff4b2b; font-weight:bold;">ĐIỂM TIN CẬY: {r[0]['s']}/100</div>
        </div>
    """, unsafe_allow_html=True)
    
    # XIÊN 2, 3 - FIX CỨNG HIỂN THỊ
    st.markdown(f"""
        <div class="xien-container">
            <div class="xien-box">
                <div class="label-xien">✨ XIÊN 2</div>
                <div class="val-xien">{r[0]['n']} - {r[1]['n']}</div>
            </div>
            <div class="xien-box">
                <div class="label-xien">🏆 XIÊN 3</div>
                <div class="val-xien">{r[0]['n']}-{r[1]['n']}-{r[2]['n']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
c_w, c_l, c_r = st.columns(3)
with c_w:
    if st.button("✅ WIN"): st.session_state.log.insert(0, "✅ Thắng (Tay)")
with c_l:
    if st.button("❌ LOSS"): st.session_state.log.insert(0, "❌ Thua (Tay)")
with c_r:
    if st.button("🗑️ CLEAR"): st.session_state.log = []; st.rerun()

for item in st.session_state.log[:10]:
    cls = "status-win" if "✅" in item else "status-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)
