import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN (FIXED) ---
st.set_page_config(page_title="AI SUPREME v4.6 FIXED", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #111; color: #00FF00; border: 1px solid #333; height: 100px !important; }
    .predict-bar {
        background: linear-gradient(90deg, #111, #222);
        padding: 15px; border-radius: 10px; border: 1px solid #444;
        display: flex; justify-content: space-between; align-items: center; margin: 10px 0;
    }
    .bt-num { font-size: 40px; color: #00FF00; font-weight: bold; }
    .log-win { color: #00ff00; font-size: 13px; margin-bottom: 3px; border-left: 3px solid #00ff00; padding-left: 10px; }
    .log-loss { color: #ff4b2b; font-size: 13px; margin-bottom: 3px; border-left: 3px solid #ff4b2b; padding-left: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo Session State bền vững
if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'current_view' not in st.session_state: st.session_state.current_view = None

BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}

# --- 2. HỆ THỐNG XỬ LÝ (FIXED LOGIC) ---
def analyze_v46_fixed(raw):
    # Khử nhiễu cực mạnh cho S-Pen
    clean_text = re.sub(r'\d{6,}', ' ', raw) # Loại mã kỳ
    nums = [int(n) for n in re.findall(r'\d', clean_text)]
    
    if not nums: return None, None

    # TỰ ĐỘNG CHECK KẾT QUẢ (Fix lỗi mất dấu dự đoán)
    if st.session_state.last_pred is not None:
        latest = nums[-1]
        if latest == st.session_state.last_pred:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - WIN")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - LOSS")
        st.session_state.last_pred = None

    if len(nums) < 10: return None, nums

    # Thuật toán tổng hợp 6 lớp
    counts = Counter(nums)
    last_val = nums[-1]
    last_sum = sum(nums[-5:]) % 10
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i
    
    scored = []
    for n in range(10):
        gap = (len(nums) - 1) - last_pos[n]
        s = (counts[n] * 0.6)
        if 4 <= gap <= 8: s += 30 # Nhịp hồi vàng
        if n == BONG.get(last_val): s += 15 # Cầu bóng
        if n == last_sum: s += 10 # Cầu tổng
        if gap > 15: s -= 20 # Số gan
        scored.append({'n': n, 's': round(max(0, s), 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. GIAO DIỆN 3 TẦNG ---
st.title("🛡️ AI SUPREME v4.6 FIXED")

# TẦNG 1: NHẬP LIỆU
raw_input = st.text_area("Dán cầu mới từ sảnh A:", placeholder="Dùng S-Pen quét vùng số đỏ...")
if st.button("🚀 PHÂN TÍCH HỆ THỐNG"):
    res, clean_nums = analyze_v46_fixed(raw_input)
    if res:
        st.session_state.last_pred = res[0]['n']
        st.session_state.current_view = {'res': res, 'nums': clean_nums}
    else:
        st.error("Cần tối thiểu 10 số kết quả!")

# TẦNG 2: KẾT QUẢ
if st.session_state.current_view:
    v = st.session_state.current_view
    top = v['res'][:3]
    st.markdown(f"""
        <div class="predict-bar">
            <div><span style="color:#888">BẠCH THỦ:</span> <span class="bt-num">{top[0]['n']}</span></div>
            <div style="text-align:right">
                <span style="color:#ff4b2b; font-weight:bold">ĐIỂM NỔ: {top[0]['s']}</span><br>
                <span style="color:#888; font-size:12px">An toàn: {top[0]['s']}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"💡 Soi cầu: Bóng kỳ trước: {BONG.get(v['nums'][-1])} | Tổng chạm: {sum(v['nums'][-5:])%10}")
    
    c_x2, c_x3 = st.columns(2)
    c_x2.info(f"✨ Xiên 2: {top[0]['n']}-{top[1]['n']}")
    c_x3.success(f"🏆 Xiên 3: {top[0]['n']}-{top[1]['n']}-{top[2]['n']}")

# TẦNG 3: THỐNG KÊ
st.markdown("---")
cw, cl, cr = st.columns(3)
if cw.button("✅ THẮNG"):
    if st.session_state.last_pred: st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - WIN")
if cl.button("❌ THUA"):
    if st.session_state.last_pred: st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - LOSS")
if cr.button("🗑️ RESET"):
    st.session_state.log = []
    st.session_state.last_pred = None
    st.rerun()

for item in st.session_state.log[:10]:
    style = "log-win" if "✅" in item else "log-loss"
    st.markdown(f'<div class="{style}">{item}</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 THUA 3 TRẬN LIÊN TIẾP - NÊN DỪNG LẠI!")
