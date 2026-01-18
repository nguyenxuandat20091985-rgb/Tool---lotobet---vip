import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN NOTE 10+ (OPTIMIZED) ---
st.set_page_config(page_title="AI SUPREME v4.6 FULL", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #111; color: #00FF00; border: 1px solid #333; font-size: 14px !important; height: 80px !important; }
    
    /* Khu vực Bạch Thủ Bar */
    .predict-bar {
        background: linear-gradient(90deg, #111, #222);
        padding: 10px; border-radius: 8px; border: 1px solid #444;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;
    }
    .bt-num { font-size: 30px; color: #00FF00; font-weight: bold; }
    .bt-score { font-size: 14px; color: #ff4b2b; font-weight: bold; }

    /* Lịch sử Log */
    .log-win { color: #00ff00; font-size: 12px; border-left: 3px solid #00ff00; padding-left: 10px; margin-bottom: 2px; }
    .log-loss { color: #ff4b2b; font-size: 12px; border-left: 3px solid #ff4b2b; padding-left: 10px; margin-bottom: 2px; }
    
    .stButton>button { height: 40px; border-radius: 8px; font-size: 14px !important; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo bộ nhớ dữ liệu
if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None

# Dữ liệu Bóng số & Cặp đối
BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
CAP_DOI = {14:41, 41:14, 23:32, 32:23, 12:21, 21:12, 56:65, 65:56, 78:87, 87:78, 09:90, 90:09}

# --- 2. HỆ THỐNG 6 THUẬT TOÁN & 4 CÁCH SOI CẦU ---
def supreme_analytics_v46(raw):
    # [Thuật toán 5]: Khử nhiễu De-noise (Lọc S-Pen & Mã kỳ)
    nums = [int(n) for n in re.findall(r'\d', re.sub(r'\d{6,}', ' ', raw))]
    if not nums: return None, None

    # TỰ ĐỘNG KIỂM TRA THẮNG THUA (Cầu mới so với dự đoán cũ)
    if st.session_state.last_pred is not None:
        if nums[-1] == st.session_state.last_pred:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 10: return None, nums

    # [Thuật toán 4]: Tần suất (Frequency)
    counts = Counter(nums)
    last_5 = nums[-5:]
    last_val = nums[-1]
    
    # [Soi cầu Tổng]: Tính tổng chạm
    sum_touch = sum(last_5) % 10
    
    # Vị trí cuối cùng của các số
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i
    
    scored = []
    total = len(nums)
    
    for n in range(10):
        # [Thuật toán 1]: Nhịp hồi (Gap 4-8)
        gap = (total - 1) - last_pos[n]
        s = 0
        if 4 <= gap <= 8: s += 25
        
        # [Thuật toán 2 & Soi cầu Bóng]: Bóng số
        if n == BONG.get(last_val): s += 12
        
        # [Soi cầu Tổng]: Cầu tổng
        if n == sum_touch: s += 10
        
        # [Soi cầu Bệt/Nhảy]: Nếu n là số vừa ra (Bệt)
        if n == last_val: s += 5 
        
        # [Thuật toán 4 tiếp tục]: Trừ điểm nếu số Gan (vắng > 12 kỳ)
        if gap > 12: s -= 20
        
        # Điểm tần suất nền
        s += (counts[n] * 0.5)
        
        # [Thuật toán 6]: Cân bằng (Normalization 0-50)
        final_s = round(max(0, min(50, s)), 1)
        scored.append({'n': n, 's': final_s})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. BỐ TRÍ GIAO DIỆN 3 TẦNG (TAB-LIKE LAYOUT) ---

# TẦNG 1: TRUNG TÂM NHẬP LIỆU
st.title("🤖 AI SUPREME v4.6 FULL")
raw_input = st.text_area("NẠP DỮ LIỆU (QUÉT S-PEN):", label_visibility="collapsed")
c1, c2 = st.columns(2)
with c1:
    btn_run = st.button("🚀 PHÂN TÍCH")
with c2:
    if st.button("🔄 RESET"): st.rerun()

# TẦNG 2: BẢNG KẾT QUẢ AI
if btn_run and raw_input:
    res, clean_nums = supreme_analytics_v46(raw_input)
    if res:
        st.session_state.last_pred = res[0]['n']
        top = res[:3]
        
        # [Tab 2]: Ô Bạch thủ Bar & Thông báo Bóng/Tổng
        st.markdown(f"""
            <div class="predict-bar">
                <div><span class="bt-label">BẠCH THỦ:</span> <span class="bt-num">{top[0]['n']}</span></div>
                <div class="bt-score">ĐIỂM NỔ: {top[0]['s']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"📢 Bóng kỳ trước: {BONG.get(clean_nums[-1])} | Tổng chạm: {sum(clean_nums[-5:])%10}")
        
        # 2 ô Xiên hiển thị song song
        col_x2, col_x3 = st.columns(2)
        col_x2.info(f"✨ Xiên 2: {top[0]['n']}-{top[1]['n']}")
        col_x3.success(f"🏆 Xiên 3: {top[0]['n']}-{top[1]['n']}-{top[2]['n']}")
        st.session_state.has_result = True

# TẦNG 3: THỐNG KÊ & QUẢN LÝ VỐN
st.markdown("---")
# Hàng nút bấm: Thắng - Thua - Xóa
tw, tl, tc = st.columns(3)
if tw.button("✅ THẮNG"): 
    if st.session_state.last_pred: st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - WIN")
if tl.button("❌ THUA"): 
    if st.session_state.last_pred: st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - LOSS")
if tc.button("🗑️ XÓA"):
    st.session_state.log = []
    st.rerun()

# Bảng Log 10-15 trận & Cảnh báo đỏ
st.markdown('<div style="background:#0a0a0a; padding:10px; border-radius:5px;">', unsafe_allow_html=True)
for item in st.session_state.log[:12]:
    cls = "log-win" if "✅" in item else "log-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3:
    if all("❌" in x for x in st.session_state.log[:3]):
        st.error("🚨 CẢNH BÁO: THUA 3 TRẬN LIÊN TIẾP - NÊN DỪNG LẠI!")
