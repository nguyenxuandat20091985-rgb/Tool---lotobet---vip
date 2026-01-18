import streamlit as st
import re
from collections import Counter

# --- 1. GIAO DIỆN HIỆN ĐẠI (TỐI ƯU TAB DỌC/NGANG) ---
st.set_page_config(page_title="MATRIX-X v5.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stTextArea textarea { background-color: #080808; color: #00FF00; border: 2px solid #333; font-size: 18px !important; }
    
    /* Thiết kế thẻ Xiên */
    .xien-card {
        background: #111; padding: 20px; border-radius: 15px;
        border: 2px solid #444; text-align: center; margin-bottom: 15px;
        transition: 0.3s;
    }
    .xien-card:hover { border-color: #00FF00; box-shadow: 0 0 20px rgba(0,255,0,0.2); }
    .label-xien { color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; }
    .val-xien { color: #00FF00; font-size: 32px; font-weight: 900; margin: 10px 0; }
    .rate-xien { font-size: 18px; font-weight: bold; padding: 5px 15px; border-radius: 20px; display: inline-block; }
    
    /* Màu sắc tỉ lệ thắng */
    .rate-high { background: rgba(0,255,0,0.2); color: #00FF00; border: 1px solid #00FF00; }
    .rate-mid { background: rgba(255,215,0,0.2); color: #FFD700; border: 1px solid #FFD700; }
    .rate-low { background: rgba(255,75,43,0.2); color: #FF4B2B; border: 1px solid #FF4B2B; }

    /* Nhật ký dự đoán */
    .log-entry { padding: 10px; border-radius: 8px; margin-bottom: 8px; border-left: 10px solid; font-weight: bold; }
    .log-win { background: rgba(0,255,0,0.1); border-color: #00FF00; color: #00FF00; }
    .log-loss { background: rgba(255,75,43,0.1); border-color: #FF4B2B; color: #FF4B2B; }
    </style>
    """, unsafe_allow_html=True)

if 'log_matrix' not in st.session_state: st.session_state.log_matrix = []
if 'last_sets' not in st.session_state: st.session_state.last_sets = None

# --- 2. THUẬT TOÁN ĐA LUỒNG (10 THUẬT TOÁN SONG SONG) ---
def matrix_engine(raw):
    nums = [int(n) for n in re.findall(r'\d', raw)]
    if len(nums) < 30: return None, len(nums)

    # TỰ ĐỘNG CHECK TRÚNG/TRƯỢT (Kiểm tra xem bộ số cũ có xuất hiện trong 5 số mới nhất)
    if st.session_state.last_sets is not None:
        new_res = nums[-5:]
        all_predicted = st.session_state.last_sets['x3_1'] + st.session_state.last_sets['x3_2'] + st.session_state.last_sets['x3_3'] + st.session_state.last_sets['x4']
        matches = [str(n) for n in set(all_predicted) if n in new_res]
        
        if matches:
            st.session_state.log_matrix.insert(0, ("win", f"🔥 TRÚNG: {', '.join(matches)} (Dựa trên {len(nums)} số)"))
        else:
            st.session_state.log_matrix.insert(0, ("loss", f"❄️ TRƯỢT (Dựa trên {len(nums)} số)"))
        st.session_state.last_sets = None

    # Tính toán điểm 10 tầng
    scores = []
    total = len(nums)
    last_5 = nums[-5:]
    counts = Counter(nums[-50:])
    
    for n in range(10):
        s = 0
        gap = 0
        for v in reversed(nums[:-1]):
            if v == n: break
            gap += 1
        
        # 10 LUỒNG DỰ TOÁN
        if 4 <= gap <= 8: s += 25             # 1. Nhịp hồi chuẩn
        if n == {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}.get(nums[-1]): s += 15 # 2. Bóng âm dương
        if n == (sum(last_5) % 10): s += 15  # 3. Tổng chạm
        if gap in [3, 5, 8, 13]: s += 10      # 4. Fibonacci Gap
        if 2 <= counts[n] <= 5: s += 20      # 5. Tần suất rơi ổn định
        if n == nums[-1] == nums[-2]: s += 10 # 6. Cầu bệt
        if n == (nums[-1] + 1) % 10: s += 5  # 7. Tiến nhịp
        if n == (nums[-1] - 1) % 10: s += 5  # 8. Lùi nhịp
        if n in nums[-10:]: s += 10          # 9. Nhịp bám đuôi
        if gap > 15: s -= 40                 # 10. Loại trừ số Gan (Bộ lọc âm)

        scores.append({'n': n, 's': max(0, s)})
    
    sorted_s = sorted(scores, key=lambda x: x['s'], reverse=True)
    return sorted_s, nums

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🌌 MATRIX-X v5.0")
st.caption("Hệ thống đa thuật toán: Tập trung Xiên 3 & Xiên 4")

input_data = st.text_area("DÁN DỮ LIỆU S-PEN TẠI ĐÂY:", height=100)

col_run, col_reset = st.columns([1, 1])
with col_run:
    if st.button("🚀 PHÂN TÍCH MATRIX"):
        res, info = matrix_engine(input_data)
        if res:
            # Tạo bộ Xiên
            st.session_state.last_sets = {
                'x3_1': [res[0]['n'], res[1]['n'], res[2]['n']],
                'x3_2': [res[0]['n'], res[1]['n'], res[3]['n']],
                'x3_3': [res[0]['n'], res[2]['n'], res[4]['n']],
                'x4': [res[0]['n'], res[1]['n'], res[2]['n'], res[3]['n']],
                'rates': [res[i]['s'] for i in range(5)]
            }
        else: st.error(f"Cần 30 số (Hiện có {info})")
with col_reset:
    if st.button("🗑️ XÓA NHẬT KÝ"): 
        st.session_state.log_matrix = []
        st.rerun()

# --- 4. HIỂN THỊ KẾT QUẢ XIÊN ---
if 'last_sets' in st.session_state and st.session_state.last_sets:
    sets = st.session_state.last_sets
    
    st.markdown("### 🎯 DANH SÁCH XIÊN TIỀM NĂNG")
    
    # XIÊN 3 (Hàng Ngang)
    t1, t2, t3 = st.columns(3)
    for i, col in enumerate([t1, t2, t3]):
        key = f'x3_{i+1}'
        avg_rate = sum(sets['rates'][:3]) / 3 + (i * -2) # Giảm nhẹ tỉ lệ theo độ ưu tiên
        cls = "rate-high" if avg_rate > 60 else "rate-mid"
        
        with col:
            st.markdown(f"""
                <div class="xien-card">
                    <div class="label-xien">XIÊN 3 - MẪU {i+1}</div>
                    <div class="val-xien">{''.join(map(str, sets[key]))}</div>
                    <div class="rate-xien {cls}">Tỉ lệ: {round(avg_rate, 1)}%</div>
                </div>
            """, unsafe_allow_html=True)
            
    # XIÊN 4 (Hàng Ngang - Trung tâm)
    st.markdown("---")
    avg_rate_x4 = sum(sets['rates'][:4]) / 4
    cls_x4 = "rate-high" if avg_rate_x4 > 65 else "rate-mid"
    
    st.markdown(f"""
        <div class="xien-card" style="border-color: #FFD700; background: linear-gradient(180deg, #111 0%, #000 100%);">
            <div class="label-xien" style="color: #FFD700;">💎 TỔNG HỢP XIÊN 4 (MẠNH NHẤT)</div>
            <div class="val-xien" style="font-size: 55px; letter-spacing: 10px; color: #FFD700;">{''.join(map(str, sets['x4']))}</div>
            <div class="rate-xien {cls_x4}">Xác suất hội tụ: {round(avg_rate_x4, 1)}%</div>
        </div>
    """, unsafe_allow_html=True)

# --- 5. BẢNG THEO DÕI DỰ ĐOÁN ---
st.markdown("### 📋 BẢNG THEO DÕI TRÚNG / TRƯỢT")
if st.session_state.log_matrix:
    for style, text in st.session_state.log_matrix[:15]:
        cls = "log-win" if style == "win" else "log-loss"
        st.markdown(f'<div class="log-entry {cls}">{text}</div>', unsafe_allow_html=True)
else:
    st.caption("Chưa có dữ liệu dự đoán. Hãy dán số và bấm Phân tích.")
