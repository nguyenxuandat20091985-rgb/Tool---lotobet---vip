import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN TỐI ƯU (HIỆN THỊ RÕ RÀNG XIÊN 2, 3) ---
st.set_page_config(page_title="AI SUPREME v4.6 REBORN", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #0a0a0a; color: #00FF00; border: 1px solid #444; font-size: 16px !important; }
    
    /* Khung Bạch Thủ */
    .bt-box {
        background: linear-gradient(90deg, #111, #222);
        padding: 15px; border-radius: 12px; border: 2px solid #00FF00;
        text-align: center; margin-bottom: 10px;
    }
    .bt-val { font-size: 50px; color: #00FF00; font-weight: bold; line-height: 1; }
    
    /* Khung Xiên 2, 3 - ĐẢM BẢO HIỂN THỊ */
    .xien-container {
        display: flex; gap: 10px; margin-bottom: 15px;
    }
    .xien-box {
        flex: 1; background: #1a1a1a; padding: 12px; border-radius: 10px;
        border: 1px solid #444; text-align: center;
    }
    .xien-label { font-size: 12px; color: #888; margin-bottom: 5px; }
    .xien-val { font-size: 22px; color: #fff; font-weight: bold; }

    .status-win { color: #00ff00; font-weight: bold; border-left: 4px solid #00ff00; padding-left: 10px; margin-bottom: 5px; background: rgba(0,255,0,0.1); }
    .status-loss { color: #ff4b2b; font-weight: bold; border-left: 4px solid #ff4b2b; padding-left: 10px; margin-bottom: 5px; background: rgba(255,75,43,0.1); }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo Session
if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'saved_res' not in st.session_state: st.session_state.saved_res = None

BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}

# --- 2. THUẬT TOÁN MẠNH NHẤT (6 LỚP + NHẬN DIỆN TREND) ---
def analyze_v46_reborn(raw):
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # TỰ ĐỘNG CHECK THẮNG/THUA (Quét 5 số mới nhất)
    if st.session_state.last_pred is not None:
        if st.session_state.last_pred in nums[-5:]:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 10: return None, nums

    counts = Counter(nums)
    last_val = nums[-1]
    last_sum = sum(nums[-5:]) % 10
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i
    
    scored = []
    total = len(nums)
    for n in range(10):
        gap = (total - 1) - last_pos[n]
        s = (counts[n] * 0.6)
        if 4 <= gap <= 8: s += 35 # Nhịp hồi vàng
        if n == BONG.get(last_val): s += 15 # Bóng
        if n == last_sum: s += 10 # Tổng chạm
        if gap > 12: s -= 20 # Số gan
        scored.append({'n': n, 's': round(max(0, s), 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. BỐ TRÍ 3 TẦNG (FIXED XIÊN 2, 3) ---

# TẦNG 1: NHẬP LIỆU
st.title("🚀 AI REBORN v4.6")
input_text = st.text_area("Dán cầu S-Pen vào đây:", height=80, label_visibility="collapsed")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 PHÂN TÍCH MỚI"):
        res, clean_nums = analyze_v46_reborn(input_text)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.saved_res = {'res': res, 'nums': clean_nums}
        else: st.error("Dữ liệu không đủ!")
with c2:
    if st.button("🗑️ RESET"): st.session_state.clear(); st.rerun()

# TẦNG 2: KẾT QUẢ (LUÔN HIỆN XIÊN)
if st.session_state.saved_res:
    r = st.session_state.saved_res['res']
    n = st.session_state.saved_res['nums']
    
    # Hiện Bạch Thủ
    st.markdown(f"""
        <div class="bt-box">
            <div style="color:#888; font-size:14px;">BẠCH THỦ KỲ TỚI</div>
            <div class="bt-val">{r[0]['n']}</div>
            <div style="color:#ff4b2b; font-weight:bold; margin-top:5px;">ĐIỂM NỔ: {r[0]['s']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Hiện Xiên 2, 3 (Thiết kế mới, cực kỳ rõ ràng)
    st.markdown(f"""
        <div class="xien-container">
            <div class="xien-box">
                <div class="xien-label">✨ XIÊN 2</div>
                <div class="xien-val">{r[0]['n']} - {r[1]['n']}</div>
            </div>
            <div class="xien-box">
                <div class="xien-label">🏆 XIÊN 3</div>
                <div class="xien-val">{r[0]['n']}-{r[1]['n']}-{r[2]['n']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"💡 Soi cầu: Bóng: {BONG.get(n[-1])} | Tổng Chạm: {sum(n[-5:])%10}")

# TẦNG 3: THỐNG KÊ
st.markdown("---")
cw, cl, cr = st.columns(3)
with cw:
    if st.button("✅ WIN"): st.session_state.log.insert(0, "✅ Thắng (Tay)")
with cl:
    if st.button("❌ LOSS"): st.session_state.log.insert(0, "❌ Thua (Tay)")
with cr:
    if st.button("🗑️ CLEAR"): st.session_state.log = []; st.rerun()

for item in st.session_state.log[:12]:
    cls = "status-win" if "✅" in item else "status-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 CẦU LOẠN! ĐÃ THUA 3 TRẬN - DỪNG CHƠI!")
