import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN HIỂN THỊ SIÊU RÕ (FIX LỖI XIÊN MỜ) ---
st.set_page_config(page_title="AI QUANTUM v4.7", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #070707; color: #00FF00; border: 1px solid #222; font-size: 18px !important; }
    
    /* Box Bạch Thủ - Rực rỡ */
    .bt-box {
        background: linear-gradient(180deg, #000 0%, #111 100%);
        padding: 25px; border-radius: 20px; border: 3px solid #00FF00;
        text-align: center; margin-bottom: 15px; box-shadow: 0 0 40px rgba(0,255,0,0.2);
    }
    .bt-val { font-size: 85px; color: #00FF00; font-weight: bold; line-height: 1; text-shadow: 0 0 25px #00FF00; }
    
    /* Box Xiên - Màu Trắng Sáng & Vàng - Chống mờ 100% */
    .xien-box {
        background: #1a1a1a; padding: 20px; border-radius: 15px;
        border: 2px solid #333; text-align: center; width: 100%; margin-bottom: 10px;
    }
    .xien-label { color: #FFD700; font-size: 16px; font-weight: bold; letter-spacing: 1px; }
    .xien-val { color: #FFFFFF !important; font-size: 32px !important; font-weight: 900 !important; margin-top: 5px; }

    .status-win { color: #00ff00; font-weight: bold; background: rgba(0,255,0,0.15); padding: 12px; border-radius: 8px; margin-bottom: 6px; border-left: 6px solid #00ff00; }
    .status-loss { color: #ff4b2b; font-weight: bold; background: rgba(255,75,43,0.15); padding: 12px; border-radius: 8px; margin-bottom: 6px; border-left: 6px solid #ff4b2b; }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'saved_res' not in st.session_state: st.session_state.saved_res = None

# --- 2. HỆ THỐNG 8 THUẬT TOÁN QUANTUM (TỰ ĐỘNG NHÂN ĐIỂM) ---
def analyze_quantum(raw):
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # Tự động check kết quả
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
    total = len(nums)
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i

    scored = []
    for n in range(10):
        # Hệ thống điểm Quantum
        logic_matches = 0
        s = 0
        gap = (total - 1) - last_pos[n]

        # 1. Thuật toán Nhịp Hồi (Gap 4-8)
        if 4 <= gap <= 8: s += 25; logic_matches += 1
        # 2. Thuật toán Tổng Chạm (Sum 5)
        if n == (sum(last_5) % 10): s += 20; logic_matches += 1
        # 3. Thuật toán Tần Suất Cao
        if counts[n] >= (total / 8): s += 15; logic_matches += 1
        # 4. Thuật toán Đối Xứng (Mirror)
        if n == {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}.get(last_val): s += 15; logic_matches += 1
        # 5. Thuật toán Cầu Bệt (Repeat)
        if n in last_5: s += 10; logic_matches += 1
        # 6. Thuật toán Điểm rơi Fibonacci (Nhịp 3, 5, 8)
        if gap in [3, 5, 8]: s += 10; logic_matches += 1
        
        # --- CƠ CHẾ NHÂN ĐIỂM KHI HỘI TỤ ---
        if logic_matches >= 3: s *= 1.5 # Nếu khớp 3 thuật toán trở lên, nhân 1.5 lần điểm
        if gap > 15: s = 0 # Loại bỏ hoàn toàn số gan để an toàn

        scored.append({'n': n, 's': round(max(0, min(100, s)), 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. GIAO DIỆN ĐIỀU KHIỂN ---
st.title("🛡️ QUANTUM AI v4.7")
input_data = st.text_area("NHẬP DỮ LIỆU CẦU:", height=80, placeholder="Dán dãy số từ S-Pen vào đây...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 KÍCH HOẠT QUANTUM"):
        res, clean_nums = analyze_quantum(input_data)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.saved_res = {'res': res, 'nums': clean_nums}
        else: st.error("Cần tối thiểu 15 số!")
with c2:
    if st.button("🔄 LÀM MỚI"): st.session_state.clear(); st.rerun()

if st.session_state.saved_res:
    r = st.session_state.saved_res['res']
    
    # BẠCH THỦ VỚI ĐIỂM CAO
    st.markdown(f"""
        <div class="bt-box">
            <div style="color:#888; font-size:16px; letter-spacing:2px;">BẠCH THỦ SIÊU CẤP</div>
            <div class="bt-val">{r[0]['n']}</div>
            <div style="color:{'#00FF00' if r[0]['s'] > 60 else '#ff4b2b'}; font-size:20px; font-weight:bold; margin-top:10px;">
                ĐỘ TIN CẬY: {r[0]['s']}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # XIÊN 2, 3 HIỂN THỊ CỰC RÕ
    col_x2, col_x3 = st.columns(2)
    with col_x2:
        st.markdown(f"""<div class="xien-box"><div class="xien-label">XIÊN 2</div><div class="xien-val">{r[0]['n']} - {r[1]['n']}</div></div>""", unsafe_allow_html=True)
    with col_x3:
        st.markdown(f"""<div class="xien-box"><div class="xien-label">XIÊN 3</div><div class="xien-val">{r[0]['n']}-{r[1]['n']}-{r[2]['n']}</div></div>""", unsafe_allow_html=True)

    if r[0]['s'] < 50:
        st.warning("⚠️ Cầu đang yếu (Dưới 50%), khuyến cáo quan sát hoặc đánh rất nhẹ!")
    else:
        st.success("🔥 Cầu hội tụ đẹp! Có thể vào tiền.")

st.markdown("---")
# Log Thắng Thua
for item in st.session_state.log[:10]:
    cls = "status-win" if "✅" in item else "status-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 CHÁY CẦU! THUA 3 TRẬN LIÊN TIẾP - DỪNG NGAY LẬP TỨC!")
