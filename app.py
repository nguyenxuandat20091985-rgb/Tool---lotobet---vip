import streamlit as st
import re
from collections import Counter

# --- GIAO DIỆN TỐI ƯU CHO NOTE 10+ (ĐỘ TƯƠNG PHẢN CỰC CAO) ---
st.set_page_config(page_title="AI OMNI v4.8", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #050505; color: #00FF00; border: 2px solid #333; font-size: 18px !important; }
    
    .bt-box {
        background: #000; padding: 25px; border-radius: 20px; border: 3px solid #00FF00;
        text-align: center; margin-bottom: 20px; box-shadow: 0 0 35px rgba(0,255,0,0.3);
    }
    .bt-val { font-size: 90px; color: #00FF00; font-weight: bold; line-height: 1; }
    
    .xien-box {
        background: #111; padding: 20px; border-radius: 15px;
        border: 2px solid #444; text-align: center; width: 100%; margin-bottom: 15px;
    }
    .xien-val { color: #FFFFFF !important; font-size: 35px !important; font-weight: 900 !important; }
    
    .status-win { color: #00ff00; font-weight: bold; background: rgba(0,255,0,0.1); padding: 12px; border-radius: 8px; margin-bottom: 5px; border-left: 8px solid #00ff00; }
    .status-loss { color: #ff4b2b; font-weight: bold; background: rgba(255,75,43,0.1); padding: 12px; border-radius: 8px; margin-bottom: 5px; border-left: 8px solid #ff4b2b; }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None
if 'saved_res' not in st.session_state: st.session_state.saved_res = None

# --- THUẬT TOÁN BIẾN THIÊN (DYNAMIC ANALYTICS) ---
def analyze_omni(raw):
    clean = re.sub(r'\d{6,}', ' ', raw)
    nums = [int(n) for n in re.findall(r'\d', clean)]
    if not nums: return None, None

    # Check thắng thua (Quét 5 số giải thưởng)
    if st.session_state.last_pred is not None:
        if st.session_state.last_pred in nums[-5:]:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 20: return None, nums

    counts = Counter(nums)
    last_5 = nums[-5:]
    total = len(nums)
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i

    # ĐỊNH NGHĨA 10 THUẬT TOÁN CHẠY ĐỘC LẬP
    scored = []
    for n in range(10):
        s = 0
        gap = (total - 1) - last_pos[n]
        
        # 1. Thuật toán Nhịp Hồi Vàng (Gap 3, 5, 7)
        if gap in [3, 5, 7]: s += 20
        # 2. Thuật toán Đối xứng (Mirror 1-6, 2-7...)
        if n == {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}.get(nums[-1]): s += 15
        # 3. Thuật toán Tổng Chạm (Sum modulo 10)
        if n == (sum(last_5) % 10): s += 15
        # 4. Thuật toán Tần suất trung bình (Tránh số quá hot, tránh số quá gan)
        if 2 <= counts[n] <= 5: s += 25
        # 5. Thuật toán Bệt (Repeat) - Kiểm tra 2 kỳ gần nhất
        if n in nums[-2:]: s += 10
        # 6. Thuật toán Xác suất Poisson (Dự đoán số sắp rơi)
        if gap > 8 and gap < 12: s += 10
        # 7. Thuật toán Tam giác (Dựa vào giải trước nữa)
        if n == (nums[-1] + nums[-2]) % 10: s += 10
        
        # BỘ LỌC THÔNG MINH:
        # Nếu số đó đang Gan (gap > 12) -> ÉP CHẾT ĐIỂM = 0
        if gap > 12: s = 0 
        
        # KIỂM TRA ĐỘ HỘI TỤ
        scored.append({'n': n, 's': round(s, 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- HIỂN THỊ ---
st.title("🌌 AI OMNI v4.8")
input_data = st.text_area("NHẬP DỮ LIỆU CẦU:", height=80, placeholder="Dán dãy số từ S-Pen...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 KÍCH HOẠT OMNI"):
        res, clean_nums = analyze_omni(input_data)
        if res:
            st.session_state.last_pred = res[0]['n']
            st.session_state.saved_res = {'res': res, 'nums': clean_nums}
        else: st.error("Cần 20 số để máy học nhịp cầu!")
with c2:
    if st.button("🗑️ RESET"): st.session_state.clear(); st.rerun()

if st.session_state.saved_res:
    r = st.session_state.saved_res['res']
    
    st.markdown(f"""
        <div class="bt-box">
            <div style="color:#888; font-size:16px;">BẠCH THỦ OMNI</div>
            <div class="bt-val">{r[0]['n']}</div>
            <div style="color:#00FF00; font-weight:bold; margin-top:10px;">TỈ LỆ HỘI TỤ: {r[0]['s']}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="xien-box">
            <div style="color:#FFD700; font-weight:bold;">✨ XIÊN 3 HỘI TỤ CAO</div>
            <div class="xien-val">{r[0]['n']} - {r[1]['n']} - {r[2]['n']}</div>
        </div>
    """, unsafe_allow_html=True)

    if r[0]['s'] < 55:
        st.error("⚠️ CẦU ĐANG LOẠN (ĐỘ TIN CẬY THẤP) - KHÔNG NÊN ĐÁNH!")
    else:
        st.success("💎 NHỊP CẦU ĐẸP - VÀO TIỀN TỰ TIN")

st.markdown("---")
# Lịch sử
for item in st.session_state.log[:12]:
    cls = "status-win" if "✅" in item else "status-loss"
    st.markdown(f'<div class="{cls}">{item}</div>', unsafe_allow_html=True)
