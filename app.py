import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN PLATINUM (SIÊU NÉT) ---
st.set_page_config(page_title="AI v4.5 PLATINUM", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #0a0a0a; color: #00FF00; border: 2px solid #444; font-size: 18px !important; }
    
    .card-bt {
        background: linear-gradient(180deg, #111 0%, #000 100%);
        padding: 35px; border-radius: 25px; border: 3px solid #00FF00;
        text-align: center; margin-bottom: 20px; box-shadow: 0 0 40px rgba(0,255,0,0.2);
    }
    .val-bt { font-size: 130px; color: #00FF00; font-weight: 900; line-height: 1; text-shadow: 0 0 45px #00FF00; }
    
    .xien-item {
        background: #fff; padding: 20px; border-radius: 15px;
        text-align: center; border: 3px solid #FFD700; width: 100%;
    }
    .xien-label { color: #000; font-size: 14px; font-weight: bold; text-transform: uppercase; }
    .xien-val { color: #ff0000 !important; font-size: 40px !important; font-weight: 900 !important; }

    .log-win { color: #00ff00; font-weight: bold; background: rgba(0,255,0,0.1); padding: 12px; border-radius: 8px; margin-bottom: 5px; border-left: 8px solid #00ff00; }
    .log-loss { color: #ff4b2b; font-weight: bold; background: rgba(255,75,43,0.1); padding: 12px; border-radius: 8px; margin-bottom: 5px; border-left: 8px solid #ff4b2b; }
    </style>
    """, unsafe_allow_html=True)

if 'log_v45' not in st.session_state: st.session_state.log_v45 = []
if 'last_pred_v45' not in st.session_state: st.session_state.last_pred_v45 = None

# --- 2. HỆ THỐNG 7 TẦNG THUẬT TOÁN ĐỘC LẬP ---
def analyze_7_layers(raw):
    nums = [int(n) for n in re.findall(r'\d', raw)]
    if len(nums) < 25: return None, len(nums)

    # Check kết quả tự động
    if st.session_state.last_pred_v45 is not None:
        if st.session_state.last_pred_v45 in nums[-5:]:
            st.session_state.log_v45.insert(0, ("win", f"✅ KỲ VỪA RỒI: TRÚNG {st.session_state.last_pred_v45}"))
        else:
            st.session_state.log_v45.insert(0, ("loss", f"❌ KỲ VỪA RỒI: TRƯỢT {st.session_state.last_pred_v45}"))
        st.session_state.last_pred_v45 = None

    scores = []
    total = len(nums)
    last_val = nums[-1]
    last_5 = nums[-5:]
    counts = Counter(nums[-40:])

    for n in range(10):
        s = 0
        gap = 0
        for i, v in enumerate(reversed(nums[:-1])):
            if v == n: break
            gap += 1

        # TẦNG 1: Nhịp hồi kỹ thuật (Gap 4-8)
        if 4 <= gap <= 8: s += 25
        # TẦNG 2: Bóng số sảnh A (0-5, 1-6, 2-7, 3-8, 4-9)
        if n == {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}.get(last_val): s += 15
        # TẦNG 3: Tổng chạm kỳ (Sum modulo 10)
        if n == (sum(last_5) % 10): s += 15
        # TẦNG 4: Đối xứng Fibonacci (Nhịp 3, 5, 8)
        if gap in [3, 5, 8]: s += 10
        # TẦNG 5: Tần suất an toàn (Số nổ 2-4 lần trong 40 kỳ)
        if 2 <= counts[n] <= 4: s += 15
        # TẦNG 6: Cầu bệt linh hoạt (Repeat detection)
        if n == last_val and nums[-1] == nums[-2]: s += 10
        # TẦNG 7: Cầu đảo lùi 2 bước
        if n == (nums[-2] + 1) % 10: s += 10

        # BỘ LỌC CHỐNG GAN: Loại bỏ tuyệt đối số > 13 kỳ chưa về
        if gap > 13: s = 0 
        
        scores.append({'n': n, 's': round(s, 1)})
    
    return sorted(scores, key=lambda x: x['s'], reverse=True), nums

# --- 3. GIAO DIỆN VẬN HÀNH ---
st.title("🛡️ AI v4.5 PLATINUM")
input_data = st.text_area("DÁN DỮ LIỆU CẦU (S-PEN):", height=100)

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 PHÂN TÍCH 7 TẦNG"):
        res, info = analyze_7_layers(input_data)
        if res:
            st.session_state.last_pred_v45 = res[0]['n']
            st.session_state.res_v45 = res
        else:
            st.error(f"Cần 25 số (Hiện có {info})")
with c2:
    if st.button("🗑️ RESET"):
        st.session_state.log_v45 = []
        st.session_state.last_pred_v45 = None
        st.rerun()

if 'res_v45' in st.session_state:
    r = st.session_state.res_v45
    
    st.markdown(f"""
        <div class="card-bt">
            <div style="color:#888; font-size:16px; letter-spacing:3px;">BẠCH THỦ 7 TẦNG</div>
            <div class="val-bt">{r[0]['n']}</div>
            <div style="color:#00FF00; font-size:20px; font-weight:bold; margin-top:10px;">ĐIỂM HỘI TỤ: {r[0]['s']}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_x2, col_x3 = st.columns(2)
    with col_x2:
        st.markdown(f'<div class="xien-item"><div class="xien-label">XIÊN 2</div><div class="xien-val">{r[0]["n"]}-{r[1]["n"]}</div></div>', unsafe_allow_html=True)
    with col_x3:
        st.markdown(f'<div class="xien-item"><div class="xien-label">XIÊN 3</div><div class="xien-val">{r[0]["n"]}-{r[1]["n"]}-{r[2]["n"]}</div></div>', unsafe_allow_html=True)

st.markdown("---")
for style, txt in st.session_state.log_v45[:10]:
    st.markdown(f'<div class="log-{style}">{txt}</div>', unsafe_allow_html=True)
