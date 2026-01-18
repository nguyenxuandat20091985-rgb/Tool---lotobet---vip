import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP ---
st.set_page_config(page_title="AI SUPREME v4.6 SAFE MODE", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #ffffff; }
    .predict-container {
        background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
        padding: 15px; border-radius: 15px; border: 1px solid #222;
        text-align: center; margin-bottom: 10px; box-shadow: 0px 4px 15px rgba(0,255,0,0.05);
    }
    .bt-number { font-size: 50px; color: #00FF00; font-weight: bold; line-height: 1; }
    .safe-badge { background: #004d00; color: #00ff00; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    .log-win { color: #00ff00; font-size: 13px; border-left: 3px solid #00ff00; padding-left: 10px; margin-bottom: 4px; }
    .log-loss { color: #ff4b2b; font-size: 13px; border-left: 3px solid #ff4b2b; padding-left: 10px; margin-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

if 'log' not in st.session_state: st.session_state.log = []
if 'last_pred' not in st.session_state: st.session_state.last_pred = None

BONG = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}

# --- 2. THUẬT TOÁN ĐỘ NHẠY AN TOÀN (SAFE LOGIC) ---
def analyze_safe(raw_input):
    # Lọc số chuẩn
    nums = [int(n) for n in re.findall(r'\d', re.sub(r'\d{6,}', ' ', raw_input))]
    if not nums: return None, None

    # TỰ ĐỘNG CHECK KẾT QUẢ
    if st.session_state.last_pred is not None:
        if nums[-1] == st.session_state.last_pred:
            st.session_state.log.insert(0, f"✅ Số {st.session_state.last_pred} - THẮNG")
        else:
            st.session_state.log.insert(0, f"❌ Số {st.session_state.last_pred} - THUA")
        st.session_state.last_pred = None

    if len(nums) < 15: return None, nums # Cần ít nhất 15 số để soi cầu an toàn

    counts = Counter(nums)
    last_val = nums[-1]
    last_sum = sum(nums[-5:]) % 10
    last_pos = {i: -1 for i in range(10)}
    for i, v in enumerate(nums): last_pos[v] = i
    
    scored = []
    total = len(nums)
    for n in range(10):
        gap = (total - 1) - last_pos[n]
        # HỆ SỐ AN TOÀN CAO:
        s = (counts[n] * 0.5) # Giảm trọng số tần suất đơn thuần
        
        # Chỉ cộng điểm mạnh nếu rơi vào nhịp hồi vàng 4-9
        if 4 <= gap <= 9: s += 30 
        
        # Phải trùng bóng hoặc trùng tổng mới được cộng thêm điểm lớn
        if n == BONG.get(last_val): s += 10
        if n == last_sum: s += 10
        
        # Trừ điểm nặng nếu số quá khan (gan) trên 15 kỳ
        if gap > 15: s -= 25
        
        scored.append({'n': n, 's': round(s, 1)})
    
    return sorted(scored, key=lambda x: x['s'], reverse=True), nums

# --- 3. GIAO DIỆN ---
st.title("🛡️ AI SUPREME v4.6 SAFE")
st.markdown("<span class='safe-badge'>CHẾ ĐỘ AN TOÀN ĐANG BẬT</span>", unsafe_allow_html=True)

input_data = st.text_area("Dán kết quả mới nhất (S-Pen):", height=80, label_visibility="collapsed")

if st.button("🚀 PHÂN TÍCH AN TOÀN"):
    results, clean_nums = analyze_safe(input_data)
    if results:
        st.session_state.last_pred = results[0]['n']
        st.session_state.current_res = results
    else:
        st.error("Dữ liệu quá mỏng! Hãy dán ít nhất 15-20 số gần nhất.")

if 'current_res' in st.session_state:
    res = st.session_state.current_res
    score = res[0]['s']
    
    st.markdown(f"""
        <div class="predict-container">
            <div style="color:#888; font-size:12px;">BẠCH THỦ TIỀM NĂNG</div>
            <div class="bt-number">{res[0]['n']}</div>
            <div style="color:{'#00FF00' if score > 35 else '#FFBB00'}; font-weight:bold;">
                MỨC ĐỘ TIN CẬY: {score}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Lệnh thực chiến an toàn
    if score > 40:
        st.success("🔥 CẦU RẤT ĐẸP: VÀO TIỀN ĐƯỢC")
    elif score > 25:
        st.warning("⚠️ CẦU TRUNG BÌNH: ĐÁNH NHẸ TAY")
    else:
        st.info("⏳ CẦU XẤU: NÊN ĐỨNG NGOÀI QUAN SÁT")

# --- 4. THỐNG KÊ WIN/LOSS ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("✅ THẮNG"): st.session_state.log.insert(0, "✅ Thắng (Thủ công)"); st.rerun()
with col2:
    if st.button("❌ THUA"): st.session_state.log.insert(0, "❌ Thua (Thủ công)"); st.rerun()
with col3:
    if st.button("🗑️ RESET"): st.session_state.log = []; st.rerun()

# Hiển thị LOG
for item in st.session_state.log[:10]:
    style = "log-win" if "✅" in item else "log-loss"
    st.markdown(f'<div class="{style}">{item}</div>', unsafe_allow_html=True)

# Cảnh báo gãy cầu
if len(st.session_state.log) >= 3 and all("❌" in x for x in st.session_state.log[:3]):
    st.error("🚨 CẢNH BÁO: THUA 3 VÁN. DỪNG CHƠI NGAY HÔM NAY!")
