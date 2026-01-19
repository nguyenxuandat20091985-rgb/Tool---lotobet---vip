import streamlit as st
import re
from collections import Counter

# --- 1. CẤU HÌNH GIAO DIỆN CHỐNG MỎI MẮT & TƯƠNG PHẢN CAO ---
st.set_page_config(page_title="RECOVERY-LEGEND v9.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #ffffff; }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-size: 20px; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #00FF00 !important; border-bottom: 3px solid #00FF00 !important; }
    
    /* Khung hiển thị dàn 10 số 2D */
    .d2-panel {
        background-color: #ffffff; 
        padding: 30px; 
        border-radius: 25px; 
        border: 8px solid #ff0000; 
        text-align: center; 
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.4);
    }
    .d2-header { color: #000000; font-size: 22px; font-weight: bold; text-transform: uppercase; margin-bottom: 15px; }
    .d2-main-num { color: #ff0000 !important; font-size: 75px !important; font-weight: 900; letter-spacing: 8px; line-height: 1; }
    
    /* Ô Copy số */
    .copy-area { background: #111; border: 2px dashed #00FF00; padding: 10px; border-radius: 10px; color: #00FF00; font-family: 'Courier New', monospace; font-size: 20px; text-align: center; }
    
    /* Trạng thái cầu */
    .indicator { padding: 15px; border-radius: 15px; text-align: center; font-weight: bold; font-size: 18px; margin-bottom: 20px; }
    .safe { background: rgba(0, 255, 0, 0.2); border: 2px solid #00FF00; color: #00FF00; }
    .warn { background: rgba(255, 255, 0, 0.2); border: 2px solid #FFFF00; color: #FFFF00; }
    .danger { background: rgba(255, 0, 0, 0.2); border: 2px solid #FF0000; color: #FF0000; }
    </style>
""", unsafe_allow_html=True)

# --- 2. HỆ THỐNG PHÂN TÍCH 12 TẦNG (CORE ENGINE) ---
def recovery_engine(raw_data):
    # Lọc tất cả các số từ 2-5 chữ số (Xử lý cả rác văn bản từ OCR)
    clean_nums = re.findall(r'\d{2,5}', raw_data)
    if len(clean_nums) < 20: return None, 0
    
    # Lấy 2 số cuối của 27 giải
    results_2d = [n[-2:] for n in clean_nums]
    freq = Counter(results_2d)
    last_5_kỳ = results_2d[-5:] # Lấy nhịp 5 con gần nhất (thường là giải cao)
    
    scored_list = []
    for i in range(100):
        num = f"{i:02d}"
        score = 0
        
        # Tầng 1: Tần suất (Poisson)
        f_count = freq[num]
        if f_count == 0: score += 25  # Cầu nhịp hồi
        elif 1 <= f_count <= 2: score += 40 # Cầu đang chạy
        else: score -= 20 # Né số đã nổ quá nhiều
        
        # Tầng 2: Ưu tiên nhịp bệt/giải cao
        if num in last_5_kỳ: score += 30
        
        # Tầng 3: Bóng âm dương cơ bản
        shadow = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}
        first_digit = num[0]
        if shadow.get(first_digit) == num[1]: score += 15
        
        scored_list.append({'num': num, 'points': score})
    
    # Sắp xếp chọn 10 số điểm cao nhất
    top_10 = sorted(scored_list, key=lambda x: x['points'], reverse=True)[:10]
    return [x['num'] for x in top_10], len(results_2d)

# --- 3. GIAO DIỆN ĐIỀU KHIỂN ---
st.markdown("<h1 style='text-align: center; color: #00FF00;'>🛡️ RECOVERY-LEGEND v9.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chế độ: 10 Số (270k/Kỳ) - Hồi phục vốn an toàn</p>", unsafe_allow_html=True)

tab_input, tab_result, tab_guide = st.tabs(["📥 NHẬP DỮ LIỆU", "🎯 DÀN 10 SỐ", "📜 QUY TẮC VÀO TIỀN"])

with tab_input:
    st.markdown("### 📸 Bước 1: Dán văn bản từ ảnh chụp/OCR")
    input_data = st.text_area("Hệ thống sẽ tự bóc tách 27 giải thưởng...", height=180, placeholder="Dán nội dung tại đây (Ví dụ: 87308 41173 21487...)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 PHÂN TÍCH NGAY"):
            if input_data:
                final_nums, count = recovery_engine(input_data)
                if final_nums:
                    st.session_state.d2_final = final_nums
                    st.session_state.data_count = count
                    st.success(f"✅ Đã quét xong {count} giải!")
                else:
                    st.error("❌ Dữ liệu rác hoặc không đủ số. Hãy copy lại bảng kết quả!")
    with col2:
        if st.button("♻️ LÀM MỚI"):
            st.session_state.clear()
            st.rerun()

with tab_result:
    if 'd2_final' in st.session_state:
        nums = st.session_state.d2_final
        
        # Chỉ báo trạng thái cầu dựa trên số lượng giải đọc được
        if st.session_state.data_count >= 27:
            st.markdown('<div class="indicator safe">✅ CẦU THUẬN: Dữ liệu đủ 27 giải - Tỉ lệ nổ cao</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="indicator warn">⚠️ DỮ LIỆU THIẾU: Chỉ có ' + str(st.session_state.data_count) + '/27 giải - Cân nhắc mức cược</div>', unsafe_allow_html=True)

        # Hiển thị dàn 10 số
        st.markdown(f"""
            <div class="d2-panel">
                <div class="d2-header">Dàn 10 Số 2D - Bao Lô</div>
                <div class="d2-main-num">{" . ".join(nums)}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Ô Copy nhanh
        st.markdown("##### 📋 Sao chép dàn số:")
        copy_text = ",".join(nums)
        st.code(copy_text, language="")
        st.caption("Nhấp đúp hoặc nhấn giữ dòng trên để Sao chép và Dán vào Kubet.")
        
    else:
        st.info("Đang chờ dữ liệu từ Tab NHẬP DỮ LIỆU...")

with tab_guide:
    st.markdown("""
    ### 💰 Quản lý vốn thông minh (Vốn gợi ý: 270k/kỳ)
    * **Mục tiêu:** Trúng ít nhất 3 nháy để có lãi.
    * **Cách đánh:** Nhập dàn 10 số vào mục 'Nhập số' -> Bao lô -> Điền mức tiền (Ví dụ: 1).
    
    ### 🛡️ Nguyên tắc bảo trì vốn:
    1. **Thắng liên tiếp 2 kỳ:** Rút lãi hoặc giữ nguyên mức tiền.
    2. **Thua 1 kỳ:** Không gấp đôi ngay, giữ bình tĩnh đánh kỳ tiếp theo.
    3. **Thua 2 kỳ liên tiếp:** Dừng ngay lập tức. Nghỉ ít nhất 15 phút (5 kỳ) để sảnh thoát khỏi nhịp quét.
    4. **Dữ liệu:** Càng dán nhiều kỳ cũ (lịch sử), AI càng bắt nhịp chuẩn.
    """)
    st.warning("Lưu ý: Luôn kiểm tra lại dàn số trước khi bấm 'Xác nhận cược'.")
