import streamlit as st
import re
from collections import Counter

# --- CẤU HÌNH GIAO DIỆN DARK MATRIX ---
st.set_page_config(page_title="AI MATRIX v4.6", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #050505; color: #00FF00; border: 1px solid #1f1f1f; font-size: 16px !important; }
    .bt-box {
        background: linear-gradient(180deg, #000 0%, #0a0a0a 100%);
        padding: 25px; border-radius: 20px; border: 2px solid #00FF00;
        text-align: center; margin-bottom: 15px; box-shadow: 0 0 30px rgba(0,255,0,0.15);
    }
    .bt-val { font-size: 70px; color: #00FF00; font-weight: bold; line-height: 1; text-shadow: 0 0 20px #00FF00; }
    .xien-box {
        background: #111; padding: 15px; border-radius: 12px;
        border: 1px solid #333; text-align: center; width: 100%;
    }
    .status-win { color: #00ff00; font-weight: bold; border-left: 5px solid #00ff00; padding: 10px; margin-bottom: 5px; background: rgba(0,255,0,0.05); }
    .status-loss { color: #ff4b2b; font-weight: bold; border-left: 5px solid #ff4b2b; padding: 10px; margin-bottom: 5px; background: rgba(255,75,43,0.05); }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'saved_res' not in st.session_state: st.session_state.saved_res = None

# --- HỆ THỐNG MATRIX: 6 THUẬT TOÁN HOẠT ĐỘNG LIÊN TỤC ---
def analyze_matrix(raw):
    # 1. Thuật toán KHỬ NHIỄU (De-noise) - Hoạt động ngay khi dán số
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # TỰ ĐỘNG ĐỐI CHIẾU THẮNG/THUA (Quét toàn bộ 5 số giải thưởng)
    if st.session_state.last_pred is not None:
        if st.session_state.last_pred in nums[-5:]:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - WIN")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - LOSS")
        st.session_state.last_pred = None

    if len(nums) < 15: return None, nums

    # Khai báo dữ liệu nền cho Matrix
    counts = Counter(nums)
    last_val = nums[-1]
    last_5 = nums[-5:]
    total = len(nums)
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i

    scored = []
    for n in range(10):
        # MATRIX SCORE (Tối đa 100 điểm cho mỗi con số)
        m_score = 0
        gap = (total - 1) - last_pos[n]

        # T.Toán 1: NHỊP HỒI (Gap 4-9) - Tỉ lệ nổ cao nhất sảnh A
        if 4 <= gap <= 9: m_score += 35
        
        # T.Toán 2: TỔNG CHẠM (Sum Logic) - Tính từ 5 số vừa về
        if n == (sum(last_5) % 10): m_score += 20
        
        # T.Toán 3: ĐỐI XỨNG (Mirror) - Bắt cầu lộn 1-6, 2-7...
        mirror = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        if n == mirror.get(last_val): m_score += 15
        
        # T.Toán 4: TẦN SUẤT (Frequency) - Ưu tiên số nổ đều, né số gan
        if counts[n] > 0: m_score += (counts[n] / total) * 50
        if gap > 15: m_score -= 40 # Trừ điểm cực nặng cho số gan
        
        # T.Toán 5: CẦU BỆT (Repeat Logic) - Kiểm tra nhịp rơi lại
        if n in last_5: m_score += 10
        
        # T.Toán 6: CÂN BẰNG (Normalization) - Giới hạn biên độ
        final_s = round(max(0, min(100, m_score)), 1)
        scored.append({'n': n, 's': final_s})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- GIAO DIỆN 3 TẦNG ---

# TẦNG 1: NHẬP LIỆU
st.title("⚡ AI MATRIX v4.6")
input_text = st.text_area("NHẬP KẾT QUẢ SẢNH A:", height=70, label_visibility="collapsed", placeholder="Dán chuỗi số từ S-Pen...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 KÍCH HOẠT MATRIX"):
        res, clean_nums = analyze_matrix(input_text)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.saved_res = {'res': res, 'nums': clean_nums}
        else: st.error("Cần tối thiểu 15 số!")
with c2:
    if st.button("🗑️ RESET"): st.session_state.clear(); st.rerun()

# TẦNG 2: KẾT QUẢ HỘI TỤ
if st.session_state.saved_res:
    r = st.session_state.saved_res['res']
    st.markdown(f"""
        <div class="bt-box">
            <div style="color:#888; font-size:14px; letter-spacing:3px;">MATRIX BẠCH THỦ</div>
            <div class="bt-val">{r[0]['n']}</div>
            <div style="color:#ff4b2b; font-weight:bold; margin-top:10px;">ĐIỂM HỘI TỤ: {r[0]['s']}/100</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_x2, col_x3 = st.columns(2)
    with col_x2:
        st.markdown(f'<div class="xien-box"><div style="color:#888;font-size:12px;">XIÊN 2</div><div style="font-size:22px;font-weight:bold;">{r[0]["n"]} - {r[1]["n"]}</div></div>', unsafe_allow_html=True)
    with col_x3:
        st.markdown(f'<div class="xien-box"><div style="color:#888;font-size:12px;">XIÊN 3</div><div style="font-size:22px;font-weight:bold;">{r[0]["n"]}-{r[1]["n"]}-{r[2]["n"]}</div></div>', unsafe_allow_html=True)

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
    st.error("🚨 MATRIX CẢNH BÁO: CẦU ĐANG GÃY - DỪNG CHƠI!")
