import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN (3 KHU VỰC CHÍNH) ---
st.set_page_config(page_title="AI SUPREME v4.6 ULTIMATE", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #111; color: #00FF00; border: 1px solid #333; height: 70px !important; }
    
    /* Tab 2: Bảng kết quả Bar */
    .predict-bar {
        background: linear-gradient(90deg, #1a1a1a, #000);
        padding: 10px 15px; border-radius: 10px; border: 1px solid #444;
        display: flex; justify-content: space-between; align-items: center; margin: 10px 0;
    }
    .bt-num { font-size: 38px; color: #00FF00; font-weight: bold; }
    .score-val { color: #ff4b2b; font-weight: bold; font-size: 14px; }
    
    /* Tab 3: Thống kê & Log */
    .log-win { color: #00ff00; font-size: 13px; border-left: 3px solid #00ff00; padding-left: 10px; margin-bottom: 2px; background: rgba(0,255,0,0.05);}
    .log-loss { color: #ff4b2b; font-size: 13px; border-left: 3px solid #ff4b2b; padding-left: 10px; margin-bottom: 2px; background: rgba(255,75,43,0.05);}
    
    .stButton>button { height: 40px; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo bộ nhớ hệ thống
if 'log' not in st.session_state: st.session_state.log = []
if 'last_prediction' not in st.session_state: st.session_state.last_prediction = None
if 'current_display' not in st.session_state: st.session_state.current_display = None

BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
CAP_DOI = {0:9, 9:0, 1:8, 8:1, 2:7, 7:2, 3:6, 6:3, 4:5, 5:4}

# --- 2. HỆ THỐNG 6 THUẬT TOÁN CỐT LÕI ---
def analyze_v46_ultimate(raw_input):
    # [T.Toán 5]: Khử nhiễu De-noise (Lọc S-Pen & Mã kỳ)
    clean_text = re.sub(r'\d{6,}', ' ', raw_input)
    all_nums = [int(n) for n in re.findall(r'\d', clean_text)]
    
    if not all_nums: return None, None

    # --- TỰ ĐỘNG ĐỐI CHIẾU THẮNG/THUA (TAB 3) ---
    # Lưu ý: Bạch thủ nổ ở bất kỳ vị trí nào trong 5 số của giải mới nhất
    if st.session_state.last_prediction is not None:
        new_result_set = all_nums[-5:] # Lấy 5 số vừa về
        if st.session_state.last_prediction in new_result_set:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_prediction} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_prediction} - THUA")
        st.session_state.last_prediction = None

    if len(all_nums) < 10: return None, all_nums

    # Chuẩn bị dữ liệu cho 6 lớp lọc
    counts = Counter(all_nums)
    last_5 = all_nums[-5:]
    last_val = all_nums[-1]
    sum_val = sum(last_5) % 10 # Cầu Tổng
    
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(all_nums): last_pos[v] = i
    
    scored = []
    total = len(all_nums)
    
    for n in range(10):
        s = 0
        # 1. Nhịp Hồi (Gap 4-8)
        gap = (total - 1) - last_pos[n]
        if 4 <= gap <= 8: s += 25
        
        # 2. Bóng Số (Shadow) & Cầu Bóng truyền thống
        if n == BONG.get(last_val): s += 12
        
        # 3. Tổng Chạm (Sum) & Cầu Tổng truyền thống
        if n == sum_val: s += 10
        
        # 4. Tần suất (Frequency) & Tránh số Gan
        s += (counts[n] * 0.5)
        if gap > 12: s -= 20 # Số quá gan
        
        # 5. Cầu Bệt/Nhảy & Cầu Đối
        if n == last_val: s += 5 # Bệt
        if n == CAP_DOI.get(last_val): s += 8 # Cầu đối
        
        # 6. Cân bằng (Normalization 0-50)
        final_score = round(max(0, min(50, s)), 1)
        scored.append({'n': n, 's': final_score})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), all_nums

# --- 3. BỐ TRÍ GIAO DIỆN 3 TẦNG (NOTE 10+) ---

# TẦNG 1: TRUNG TÂM NHẬP LIỆU
st.title("🤖 AI SUPREME v4.6 ULTIMATE")
input_data = st.text_area("NHẬP GIẢI THƯỞNG (S-PEN):", label_visibility="collapsed", placeholder="Dán kết quả sảnh A...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 PHÂN TÍCH"):
        res, nums = analyze_v46_ultimate(input_data)
        if res:
            st.session_state.last_prediction = res[0]['n']
            st.session_state.current_display = {'res': res, 'nums': nums}
        else: st.warning("Cần thêm dữ liệu!")
with c2:
    if st.button("🔄 LÀM MỚI"): 
        st.session_state.last_prediction = None
        st.session_state.current_display = None
        st.rerun()

# TẦNG 2: BẢNG KẾT QUẢ AI (THU GỌN)
if st.session_state.current_display:
    d = st.session_state.current_display
    top = d['res'][:3]
    
    st.markdown(f"""
        <div class="predict-bar">
            <div><span style="color:#888; font-size:12px;">BẠCH THỦ:</span> <span class="bt-num">{top[0]['n']}</span></div>
            <div class="score-val">ĐIỂM NỔ: {top[0]['s']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Dòng thông báo Bóng & Tổng đối chiếu nhanh
    st.caption(f"🔍 Soi nhanh: Bóng: {BONG.get(d['nums'][-1])} | Tổng Chạm: {sum(d['nums'][-5:])%10}")
    
    col_x2, col_x3 = st.columns(2)
    col_x2.info(f"✨ Xiên 2: {top[0]['n']}-{top[1]['n']}")
    col_x3.success(f"🏆 Xiên 3: {top[0]['n']}-{top[1]['n']}-{top[2]['n']}")

# TẦNG 3: THỐNG KÊ & QUẢN LÝ VỐN
st.markdown("---")
tw, tl, tc = st.columns(3)
with tw:
    if st.button("✅ THẮNG"): st.session_state.log.insert(0, "✅ Thắng (Thủ công)")
with tl:
    if st.button("❌ THUA"): st.session_state.log.insert(0, "❌ Thua (Thủ công)")
with tc:
    if st.button("🗑️ XÓA LOG"): 
        st.session_state.log = []
        st.rerun()

# Bảng Log & Cảnh báo
log_container = st.container()
with log_container:
    for item in st.session_state.log[:12]:
        cls = "log-win" if "✅" in item else "log-loss"
        st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 CẢNH BÁO: THUA 3 TRẬN LIÊN TIẾP - DỪNG LẠI!")
