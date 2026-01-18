import streamlit as st
import re
from collections import Counter

# --- GIAO DIỆN HIỂN THỊ SIÊU TƯƠNG PHẢN (CHỐNG MỜ) ---
st.set_page_config(page_title="X-MATRIX v4.9", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #0a0a0a; color: #00FF00; border: 2px solid #222; font-size: 20px !important; }
    
    .main-card {
        background: #000; padding: 30px; border-radius: 25px; border: 4px solid #00FF00;
        text-align: center; margin-bottom: 20px; box-shadow: 0 0 50px rgba(0,255,0,0.2);
    }
    .val-large { font-size: 110px; color: #00FF00; font-weight: bold; line-height: 1; text-shadow: 0 0 30px #00FF00; }
    
    .xien-grid { display: flex; gap: 15px; margin-top: 20px; }
    .xien-item {
        flex: 1; background: #111; padding: 25px; border-radius: 15px;
        border: 2px solid #444; text-align: center;
    }
    .xien-val { color: #FFFFFF !important; font-size: 40px !important; font-weight: 900 !important; }

    .win-log { color: #00ff00; font-weight: bold; background: rgba(0,255,0,0.1); padding: 10px; border-radius: 5px; border-left: 10px solid #00ff00; margin-bottom: 5px; }
    .loss-log { color: #ff4b2b; font-weight: bold; background: rgba(255,75,43,0.1); padding: 10px; border-radius: 5px; border-left: 10px solid #ff4b2b; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo bộ nhớ nhịp cầu
if 'history' not in st.session_state: st.session_state.history = []
if 'last_p' not in st.session_state: st.session_state.last_p = None

def engine_x49(raw):
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if len(nums) < 25: return None, nums

    # TỰ ĐỘNG KIỂM TRA (Quét sâu 5 số)
    if st.session_state.last_p is not None:
        if st.session_state.last_p in nums[-5:]:
            st.session_state.log.insert(0, f"✅ THẮNG: {st.session_state.last_p}")
        else:
            st.session_state.log.insert(0, f"❌ THUA: {st.session_state.last_p}")
        st.session_state.last_p = None

    # HỆ THỐNG 6 THUẬT TOÁN NEURAL BIAS
    scored = []
    last_val = nums[-1]
    last_5 = nums[-5:]
    counts = Counter(nums[-30:]) # Chỉ quét 30 số gần nhất để nhạy bén

    for n in range(10):
        s = 0
        # T.Toán 1: Nhịp Rơi Tự Do (Ưu tiên nhịp 4, 6, 8)
        gap = 0
        for i, v in enumerate(reversed(nums[:-1])):
            if v == n: break
            gap += 1
        
        if gap in [4, 6, 8]: s += 30
        # T.Toán 2: Đối xứng Quantum (Loại trừ bóng chết)
        if n == {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}.get(last_val): s += 15
        # T.Toán 3: Tổng cân bằng sảnh A
        if n == (sum(last_5) % 10): s += 20
        # T.Toán 4: Thuật toán Bệt (Nếu đang bệt thì đánh tiếp)
        if n == last_val and nums[-1] == nums[-2]: s += 25
        # T.Toán 5: Tần suất an toàn (Né số nổ > 6 lần/30 kỳ)
        if 1 <= counts[n] <= 4: s += 10
        # T.Toán 6: Điểm rơi lùi 2 kỳ
        if n == (nums[-2] + 1) % 10: s += 5

        # BỘ LỌC TỬ THẦN (Né 100% số Gan cực dài)
        if gap > 12: s = 0 
        
        scored.append({'n': n, 's': s})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ X-MATRIX v4.9 NEURAL")
data_in = st.text_area("DÁN DỮ LIỆU S-PEN:", height=100)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🚀 PHÂN TÍCH X-MATRIX"):
        res, clean_nums = engine_x49(data_in)
        if res:
            st.session_state.last_p = res[0]['n']
            st.session_state.saved = res
        else: st.error("Cần tối thiểu 25 số!")
with col_b:
    if st.button("🔄 LÀM MỚI"): st.session_state.clear(); st.rerun()

if 'saved' in st.session_state:
    r = st.session_state.saved
    st.markdown(f"""
        <div class="main-card">
            <div style="color:#888; font-size:18px;">BẠCH THỦ CHỐT KỲ</div>
            <div class="val-large">{r[0]['n']}</div>
            <div style="color:{'#00FF00' if r[0]['s'] >= 65 else '#FF4B2B'}; font-size:24px; font-weight:bold;">
                ĐIỂM TIN CẬY: {r[0]['s']}/100
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="xien-grid">
            <div class="xien-item"><div style="color:#FFD700;">XIÊN 2</div><div class="xien-val">{r[0]['n']}-{r[1]['n']}</div></div>
            <div class="xien-item"><div style="color:#FFD700;">XIÊN 3</div><div class="xien-val">{r[0]['n']}-{r[1]['n']}-{r[2]['n']}</div></div>
        </div>
    """, unsafe_allow_html=True)

    if r[0]['s'] < 65:
        st.warning("⚠️ Cảnh báo: Nhịp cầu đang nhiễu mạnh. Chỉ nên đánh nhẹ!")
