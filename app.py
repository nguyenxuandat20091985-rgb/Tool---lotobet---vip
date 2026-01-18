import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN TỐI ƯU DARK MODE ---
st.set_page_config(page_title="AI SUPREME v4.6 PRO MAX", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #0a0a0a; color: #00FF00; border: 1px solid #444; font-size: 16px !important; }
    .predict-bar {
        background: linear-gradient(90deg, #111, #222);
        padding: 15px; border-radius: 12px; border: 2px solid #333;
        display: flex; justify-content: space-between; align-items: center; margin: 10px 0;
    }
    .bt-num { font-size: 45px; color: #00FF00; font-weight: bold; text-shadow: 0 0 10px #00FF00; }
    .status-win { color: #00ff00; font-weight: bold; border-left: 4px solid #00ff00; padding-left: 10px; margin-bottom: 5px; background: rgba(0,255,0,0.1); }
    .status-loss { color: #ff4b2b; font-weight: bold; border-left: 4px solid #ff4b2b; padding-left: 10px; margin-bottom: 5px; background: rgba(255,75,43,0.1); }
    .stButton>button { border-radius: 10px; font-weight: bold; height: 50px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'display_data' not in st.session_state: st.session_state.display_data = None

BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}
CAP_DOI = {0:9, 9:0, 1:8, 8:1, 2:7, 7:2, 3:6, 6:3, 4:5, 5:4}

# --- 2. THUẬT TOÁN PRO MAX (DYNAMIC WEIGHTING) ---
def analyze_v46_promax(raw_input):
    # Lọc khử nhiễu mã kỳ và S-Pen
    clean = re.sub(r'\d{6,}', ' ', raw_input)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # TỰ ĐỘNG CHECK THẮNG/THUA (Quét 5 số giải mới nhất)
    if st.session_state.last_pred is not None:
        new_set = nums[-5:]
        if st.session_state.last_pred in new_set:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 15: return None, nums

    # PHÂN TÍCH XU HƯỚNG CẦU (Bệt hay Nhảy)
    last_10 = nums[-10:]
    is_bet_trend = len(set(last_10)) < 7 # Nếu 10 kỳ mà chỉ loanh quanh vài số -> Cầu đang bệt
    
    counts = Counter(nums)
    last_val = nums[-1]
    last_sum = sum(nums[-5:]) % 10
    
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i
    
    scored = []
    total = len(nums)
    
    for n in range(10):
        s = 0
        gap = (total - 1) - last_pos[n]
        
        # 1. Thuật toán Nhịp Hồi Vàng (Ưu tiên gap 3-7 kỳ)
        if 3 <= gap <= 7: s += 30
        
        # 2. Thuật toán Bóng & Đối
        if n == BONG.get(last_val): s += 15
        if n == CAP_DOI.get(last_val): s += 10
        
        # 3. Thuật toán Tổng Chạm
        if n == last_sum: s += 12
        
        # 4. ĐIỀU CHỈNH THEO XU HƯỚNG (Mạnh hơn v4.5)
        if is_bet_trend and n in last_10: s += 15 # Ưu tiên số vừa ra nếu đang bệt
        if not is_bet_trend and gap == 0: s -= 10 # Trừ điểm bệt nếu đang cầu nhảy
        
        # 5. Khử số Gan (vắng > 12 kỳ)
        if gap > 12: s -= 25
        
        # 6. Cân bằng Điểm Nổ (0-50)
        final_s = round(max(0, min(50, s + (counts[n]*0.4))), 1)
        scored.append({'n': n, 's': final_s})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. GIAO DIỆN 3 TẦNG CHUYÊN BIỆT ---

# TẦNG 1: NHẬP LIỆU
st.title("⚡ AI PRO MAX v4.6")
input_text = st.text_area("Dán cầu mới (S-Pen):", label_visibility="collapsed", placeholder="Quét vùng 5 số đỏ...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 KÍCH HOẠT SOI CẦU"):
        res, clean_nums = analyze_v46_promax(input_text)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.display_data = {'res': res, 'nums': clean_nums}
        else: st.warning("Cần thêm dữ liệu!")
with c2:
    if st.button("🗑️ LÀM MỚI"): 
        st.session_state.clear(); st.rerun()

# TẦNG 2: KẾT QUẢ (Bạch thủ nổ bất kỳ vị trí nào)
if st.session_state.display_data:
    d = st.session_state.display_data
    top = d['res'][:3]
    st.markdown(f"""
        <div class="predict-bar">
            <div><span style="color:#888; font-size:12px;">BẠCH THỦ KỲ TỚI:</span> <br><span class="bt-num">{top[0]['n']}</span></div>
            <div style="text-align:right">
                <span style="color:#ff4b2b; font-size:18px; font-weight:bold;">ĐIỂM NỔ: {top[0]['s']}</span><br>
                <span style="color:#00ff00; font-size:12px;">Xác suất: {int(top[0]['s']*2)}%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info(f"💡 Soi nhanh: Bóng: {BONG.get(d['nums'][-1])} | Tổng Chạm: {sum(d['nums'][-5:])%10}")
    
    cx2, cx3 = st.columns(2)
    cx2.markdown(f"<div style='background:#111;padding:10px;border:1px solid #444;text-align:center;'>✨ Xiên 2: <b>{top[0]['n']}-{top[1]['n']}</b></div>", unsafe_allow_html=True)
    cx3.markdown(f"<div style='background:#111;padding:10px;border:1px solid #444;text-align:center;'>🏆 Xiên 3: <b>{top[0]['n']}-{top[1]['n']}-{top[2]['n']}</b></div>", unsafe_allow_html=True)

# TẦNG 3: THỐNG KÊ & CẢNH BÁO
st.markdown("---")
col_w, col_l, col_r = st.columns(3)
with col_w:
    if st.button("✅ WIN"): st.session_state.log.insert(0, "✅ Thắng (Thủ công)")
with col_l:
    if st.button("❌ LOSS"): st.session_state.log.insert(0, "❌ Thua (Thủ công)")
with col_r:
    if st.button("🗑️ CLEAR"): st.session_state.log = []; st.rerun()

for item in st.session_state.log[:12]:
    cls = "status-win" if "✅" in item else "status-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)

if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 CẦU LOẠN! ĐÃ THUA 3 TRẬN - DỪNG CHƠI NGAY!")
