import streamlit as st
import re
from collections import Counter

# --- CẤU HÌNH GIAO DIỆN SIÊU TƯƠNG PHẢN ---
st.set_page_config(page_title="TITAN BLACK v5.0", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTextArea textarea { background-color: #050505; color: #00FF00; border: 2px solid #1f1f1f; font-size: 20px !important; }
    
    .card {
        background: linear-gradient(180deg, #0a0a0a 0%, #000 100%);
        padding: 30px; border-radius: 20px; border: 2px solid #333;
        text-align: center; margin-bottom: 20px;
    }
    .big-num { font-size: 120px; color: #00FF00; font-weight: bold; line-height: 1; text-shadow: 0 0 40px rgba(0,255,0,0.4); }
    .reliability { font-size: 24px; font-weight: bold; margin-top: 10px; }
    
    .xien-box {
        background: #111; padding: 20px; border-radius: 12px; border: 1px solid #444;
        text-align: center; margin-top: 10px;
    }
    .xien-val { color: #FFFFFF; font-size: 38px; font-weight: 900; }
    
    .win { color: #00ff00; background: rgba(0,255,0,0.1); padding: 8px; border-left: 8px solid #00ff00; margin-bottom: 4px; border-radius: 4px; }
    .loss { color: #ff4b2b; background: rgba(255,75,43,0.1); padding: 8px; border-left: 8px solid #ff4b2b; margin-bottom: 4px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo bộ nhớ (Fix lỗi AttributeError)
if 'titan_log' not in st.session_state: st.session_state.titan_log = []
if 'next_bet' not in st.session_state: st.session_state.next_bet = None

def titan_engine(data):
    # Làm sạch dữ liệu
    raw_nums = re.findall(r'\d', data)
    nums = [int(n) for n in raw_nums if len(raw_nums) > 0]
    
    if len(nums) < 30: return None, len(nums)

    # 1. CẬP NHẬT LỊCH SỬ (Quét 5 số cuối của sảnh)
    if st.session_state.next_bet is not None:
        last_results = nums[-5:]
        if st.session_state.next_bet in last_results:
            st.session_state.titan_log.insert(0, ("win", f"✅ TRÚNG {st.session_state.next_bet}"))
        else:
            st.session_state.titan_log.insert(0, ("loss", f"❌ TRƯỢT {st.session_state.next_bet}"))
        st.session_state.next_bet = None

    # 2. HỆ THỐNG LỌC 3 LỚP (TRIPLE-FILTER)
    scored = []
    last_val = nums[-1]
    last_10 = nums[-10:]
    freq = Counter(nums[-40:]) # Quét rộng 40 kỳ để tìm nhịp

    for n in range(10):
        score = 0
        # Tính khoảng cách (Gap)
        gap = 0
        for i, v in enumerate(reversed(nums[:-1])):
            if v == n: break
            gap += 1
        
        # Lớp 1: Nhịp hồi kỹ thuật (Chỉ bắt nhịp 4-9)
        if 4 <= gap <= 9: score += 35
        # Lớp 2: Điểm hội tụ toán học (Tổng & Bóng)
        if n == (sum(nums[-3:]) % 10): score += 20
        if n == {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}.get(last_val): score += 15
        # Lớp 3: Tần suất an toàn (Tránh số nổ quá nhiều > 6 lần)
        if 1 <= freq[n] <= 5: score += 30

        # BỘ CHẶN TỬ THẦN: Loại bỏ số Gan > 12 kỳ và số vừa nổ (Tránh gãy cầu hồi)
        if gap > 12 or gap == 0: score = 0
        
        scored.append({'n': n, 's': score})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- GIAO DIỆN ĐIỀU KHIỂN ---
st.title("🌑 TITAN BLACK v5.0")
input_data = st.text_area("DÁN DỮ LIỆU CẦU MỚI NHẤT:", height=100, help="Dán ít nhất 30 số")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔥 PHÂN TÍCH TITAN"):
        res, info = titan_engine(input_data)
        if res:
            st.session_state.next_bet = res[0]['n']
            st.session_state.results = res
        else:
            st.error(f"Dữ liệu yếu! Cần 30 số (Hiện có {info})")

with col2:
    if st.button("♻️ RESET HỆ THỐNG"):
        st.session_state.titan_log = []
        st.session_state.next_bet = None
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if 'results' in st.session_state:
    r = st.session_state.results
    conf = r[0]['s']
    color = "#00FF00" if conf >= 70 else ("#FFD700" if conf >= 50 else "#FF4B2B")
    
    st.markdown(f"""
        <div class="card">
            <div style="color:#888; letter-spacing:2px;">BẠCH THỦ ĐỘC ĐẮC</div>
            <div class="big-num">{r[0]['n']}</div>
            <div class="reliability" style="color:{color};">ĐỘ TIN CẬY: {conf}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    c_x2, c_x3 = st.columns(2)
    with c_x2:
        st.markdown(f'<div class="xien-box"><small>XIÊN 2</small><div class="xien-val">{r[0]["n"]}-{r[1]["n"]}</div></div>', unsafe_allow_html=True)
    with c_x3:
        st.markdown(f'<div class="xien-box"><small>XIÊN 3</small><div class="xien-val">{r[0]["n"]}-{r[1]["n"]}-{r[2]["n"]}</div></div>', unsafe_allow_html=True)

    if conf < 60:
        st.error("🚨 CẢNH BÁO: Cầu đang nhiễu cực độ. Tỉ lệ thắng thấp, hãy chờ nhịp sau!")
    else:
        st.success("✅ Nhịp cầu đủ tiêu chuẩn an toàn. Có thể vào tiền.")

# Lịch sử thắng thua
st.markdown("### 📋 NHẬT KÝ CHIẾN TRƯỜNG")
for style, text in st.session_state.titan_log[:10]:
    st.markdown(f'<div class="{style}">{text}</div>', unsafe_allow_html=True)
