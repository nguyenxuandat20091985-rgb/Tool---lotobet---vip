import streamlit as st
import re
import pandas as pd
from collections import Counter

# Tối ưu giao diện điện thoại
st.set_page_config(page_title="AI LOTOBET V5 - BỆT DETECTOR", layout="wide")

# Khởi tạo bộ nhớ dài hạn cho AI học tập
if 'long_term_memory' not in st.session_state:
    st.session_state.long_term_memory = []

st.title("🚀 SIÊU AI V5 - HỆ THỐNG NHẬN DIỆN CẦU BỆT")
st.sidebar.header("🤖 TRUNG TÂM ĐIỀU KHIỂN AI")

# Tính năng xóa bộ nhớ để AI học lại từ đầu nếu muốn
if st.sidebar.button("🗑️ Reset AI & Xóa dữ liệu cũ"):
    st.session_state.long_term_memory = []
    st.rerun()

st.markdown("---")

# --- KHU VỰC NHẬP DỮ LIỆU ---
st.subheader("📊 Nhập kết quả đa nguồn")
input_data = st.text_area("Dán dữ liệu thô (Copy từ nhà cái, ảnh quét Lens...):", height=150)

if st.button("🔥 PHÂN TÍCH CHUYÊN SÂU & SOI CẦU BỆT"):
    if input_data:
        # Lọc số thông minh
        digits = "".join(re.findall(r'\d', input_data))
        new_kỳs = [digits[i:i+5] for i in range(0, len(digits)-4, 5)]
        
        # AI học tập: Cộng dồn vào bộ nhớ
        st.session_state.long_term_memory.extend(new_kỳs)
        st.session_state.long_term_memory = st.session_state.long_term_memory[-300:] # Nhớ 300 kỳ gần nhất

        if len(st.session_state.long_term_memory) > 5:
            all_str = "".join(st.session_state.long_term_memory)
            counts = Counter(all_str)
            
            # --- THUẬT TOÁN NHẬN DIỆN CẦU BỆT ---
            st.success(f"✅ AI đã nạp {len(st.session_state.long_term_memory)} kỳ vào bộ nhớ học tập.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("📉 BẢNG TẦN SUẤT CHI TIẾT")
                df = pd.DataFrame(counts.items(), columns=['Số', 'Lần']).sort_values(by='Lần', ascending=False)
                st.table(df)

            with col2:
                st.warning("🔮 DỰ ĐOÁN TỪ HỆ THỐNG AI")
                top_3 = counts.most_common(3)
                s1, s2, s3 = top_3[0][0], top_3[1][0], top_3[2][0]
                
                st.subheader("⭐ TAM THỦ LÔ (Tỉ lệ thắng cao)")
                st.code(f"{s1} - {s2} - {s3}", language="text")
                
                # Logic soi cầu bệt
                st.subheader("🚨 CẢNH BÁO CẦU BỆT")
                recent_data = "".join(st.session_state.long_term_memory[-10:])
                bet_found = False
                for num in "0123456789":
                    if recent_data.count(num) >= 5: # Nếu 1 số xuất hiện > 5 lần trong 10 kỳ
                        st.error(f"Phát hiện cầu BỆT số: {num} (Rất mạnh!)")
                        bet_found = True
                if not bet_found:
                    st.write("Hiện chưa có cầu bệt rõ ràng.")

            st.markdown("---")
            st.subheader("📈 XU HƯỚNG DÒNG SỐ")
            st.line_chart(df.set_index('Số'))
        else:
            st.error("Dữ liệu quá ít. Hãy nạp thêm kỳ quay để AI học hỏi!")
