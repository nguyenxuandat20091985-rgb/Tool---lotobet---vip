import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN HIỆN ĐẠI & TƯƠNG PHẢN CAO ---
st.set_page_config(page_title="TITAN-MATRIX v5.2", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #050505; color: #00FF00; border: 2px solid #00FF00; font-size: 22px !important; }
    
    /* Thiết kế Tab Xiên 3 dọc */
    .column-x3 { border-right: 1px solid #333; padding: 15px; }
    .card-x3 {
        background: #111; padding: 20px; border-radius: 15px; border-left: 10px solid #00FF00;
        margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;
    }
    .num-x3 { color: #00FF00; font-size: 38px; font-weight: 900; letter-spacing: 4px; }
    
    /* Thiết kế Xiên 4 ngang hiện đại */
    .box-x4 {
        background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
        padding: 40px; border-radius: 25px; border: 4px solid #FFD700;
        text-align: center; margin-bottom: 25px; box-shadow: 0 0 40px rgba(255,215,0,0.3);
    }
    .num-x4 { color: #FFD700; font-size: 70px; font-weight: 900; letter-spacing: 15px; text-shadow: 0 0 20px #FFD700; }
    
    /* Bảng theo dõi trúng trượt */
    .log-win { color: #00ff00; background: rgba(0,255,0,0.1); padding: 15px; border-radius: 10px; border-left: 10px solid #00ff00; margin-bottom: 10px; font-weight: bold; font-size: 18px; }
    .log-loss { color: #ff4b2b; background: rgba(255,75,43,0.1); padding: 15px; border-radius: 10px; border-left: 10px solid #ff4b2b; margin-bottom: 10px; font-weight: bold; font-size: 18px; }
    .rate-tag { background: #FFD700; color: #000; padding: 3px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

if 'history_matrix' not in st.session_state: st.session_state.history_matrix = []
if 'last_sets' not in st.session_state: st.session_state.last_sets = None

# --- 2. HỆ THỐNG 10 THUẬT TOÁN SONG SONG ---
def analyze_titan(raw):
    # Lấy 1 số cuối của các giải thưởng (đúng luật lotobet)
    nums = [int(n) for n in re.findall(r'\d', raw)]
    if len(nums) < 30: return None, len(nums)

    # TỰ ĐỘNG CHECK KẾT QUẢ KHI CÓ DỮ LIỆU MỚI
    if st.session_state.last_sets:
        new_result = nums[-5:] # 5 số vừa mở của sảnh
        sets = st.session_state.last_sets
        
        # Check Xiên 4
        x4_match = sum(1 for x in sets['x4'] if x in new_result)
        # Check Xiên 3
        x3_match = any(sum(1 for x in s if x in new_result) >= 3 for s in [sets['x3a'], sets['x3b'], sets['x3c']])
        
        res_str = "".join(map(str, new_result))
        if x4_match == 4:
            st.session_state.history_log.insert(0, ("win", f"🏆 RỰC RỠ XIÊN 4! Giải: {res_str}"))
        elif x3_match:
            st.session_state.history_log.insert(0, ("win", f"✅ TRÚNG XIÊN 3! Giải: {res_str}"))
        else:
            st.session_state.history_log.insert(0, ("loss", f"❌ TRƯỢT (Trúng {x4_match}/4 số). Giải: {res_str}"))
        st.session_state.last_sets = None

    # TÍNH TOÁN ĐIỂM 10 LỚP
    scored = []
    freq_40 = Counter(nums[-40:])
    last_5 = nums[-5:]
    
    for n in range(10):
        s = 0
        gap = 0
        for v in reversed(nums[:-1]):
            if v == n: break
            gap += 1
            
        # 10 THUẬT TOÁN SONG SONG
        if 4 <= gap <= 8: s += 30                # 1. Nhịp hồi (Gap chuẩn)
        if n == {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}.get(nums[-1]): s += 20 # 2. Bóng âm dương
        if n == (sum(last_5) % 10): s += 15     # 3. Tổng chạm
        if gap in [3, 5, 8]: s += 10            # 4. Nhịp Fibonacci
        if 2 <= freq_40[n] <= 5: s += 25        # 5. Tần suất rơi ổn định
        if n in last_5: s += 10                 # 6. Nhịp bệt (Repeat)
        if n == (nums[-1] + 1) % 10: s += 5     # 7. Nhịp tiến
        if n == (nums[-1] - 1) % 10: s += 5     # 8. Nhịp lùi
        if gap > 13: s -= 45                    # 9. BỘ LỌC SỐ GAN (Cực quan trọng)
        if freq_40[n] > 8: s -= 20              # 10. Né số "đứng" cầu

        scored.append({'n': n, 's': max(0, s)})
    
    sorted_res = sorted(scored, key=lambda x: x['s'], reverse=True)
    return sorted_res, nums

# --- 3. GIAO DIỆN ĐIỀU KHIỂN ---
st.title("🛡️ TITAN-MATRIX v5.2")
st.markdown("##### Chuyên gia dự đoán Xiên 3 & Xiên 4 (Bao 5 Giải LotoBet)")

data_input = st.text_area("DÁN DỮ LIỆU S-PEN:", height=100)

col_f1, col_f2 = st.columns(2)
with col_f1:
    if st.button("🚀 PHÂN TÍCH MATRIX"):
        res, info = analyze_titan(data_input)
        if res:
            st.session_state.last_sets = {
                'x4': [res[0]['n'], res[1]['n'], res[2]['n'], res[3]['n']],
                'x3a': [res[0]['n'], res[1]['n'], res[2]['n']],
                'x3b': [res[0]['n'], res[1]['n'], res[3]['n']],
                'x3c': [res[0]['n'], res[2]['n'], res[4]['n']],
                'scores': [res[i]['s'] for i in range(5)]
            }
        else: st.error(f"Cần tối thiểu 30 số (Hiện có {info})")
with col_f2:
    if st.button("♻️ LÀM MỚI"):
        st.session_state.history_log = []
        st.session_state.last_sets = None
        st.rerun()

# --- 4. HIỂN THỊ KẾT QUẢ XIÊN 3 & 4 ---
if st.session_state.last_sets:
    ls = st.session_state.last_sets
    
    # XIÊN 4 (Hàng ngang - Trung tâm)
    st.markdown(f"""
        <div class="box-x4">
            <div style="color: #FFD700; font-size: 18px; letter-spacing: 5px; margin-bottom:10px;">💎 TỔNG HỢP XIÊN 4 MẠNH NHẤT</div>
            <div class="num-x4">{"".join(map(str, ls['x4']))}</div>
            <div style="margin-top:15px;"><span class="rate-tag">TỈ LỆ HỘI TỤ: {round(sum(ls['scores'][:4])/4, 1)}%</span></div>
        </div>
    """, unsafe_allow_html=True)

    # XIÊN 3 (3 Mẫu - Tab Dọc)
    st.markdown("### 🎯 DANH SÁCH XIÊN 3 TIỀM NĂNG")
    col_x3a, col_x3b, col_x3c = st.columns(3)
    for i, (col, key) in enumerate(zip([col_x3a, col_x3b, col_x3c], ['x3a', 'x3b', 'x3c'])):
        with col:
            rate_x3 = round(sum(ls['scores'][:3])/3 - (i*2), 1)
            st.markdown(f"""
                <div class="card-x3">
                    <div>
                        <small style="color:#888;">MẪU {i+1}</small><br>
                        <span class="num-x3">{"".join(map(str, ls[key]))}</span>
                    </div>
                    <div style="color:#FFD700; font-weight:bold;">{rate_x3}%</div>
                </div>
            """, unsafe_allow_html=True)

# --- 5. BẢNG THEO DÕI DỰ ĐOÁN ---
st.markdown("---")
st.markdown("### 📋 NHẬT KÝ KIỂM CHỨNG (TRÚNG / TRƯỢT)")
if 'history_log' not in st.session_state: st.session_state.history_log = []
for type_log, text in st.session_state.history_log[:15]:
    st.markdown(f'<div class="log-{type_log}">{text}</div>', unsafe_allow_html=True)
