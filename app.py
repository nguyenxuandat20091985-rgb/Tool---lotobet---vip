import streamlit as st
import re
import itertools
import pandas as pd
from collections import Counter

st.set_page_config(page_title="AI LOTOBET V8 - PRO TRADER", layout="wide")

if 'memory' not in st.session_state:
    st.session_state.memory = []

st.title("🚀 SIÊU AI V8 - CHIẾN THUẬT VÀO TIỀN & SOI CẦU")
st.markdown("---")

# --- KHU VỰC QUẢN LÝ VỐN ---
st.sidebar.header("💰 CÀI ĐẶT VỐN")
von = st.sidebar.number_input("Nhập tổng vốn của bạn:", min_value=0, value=1000)
muc_cuoc = st.sidebar.selectbox("Chiến thuật vào tiền:", ["Đều tay (1-1-1)", "Gấp thếp nhẹ (1-2-4)", "Tiến cấp (1-3-8)"])

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset Dữ Liệu"):
    st.session_state.memory = []
    st.rerun()

# --- NHẬP DỮ LIỆU ---
input_data = st.text_area("Dán kết quả thô vào đây:", height=150)

if st.button("🔥 PHÂN TÍCH & CHỐT ĐƠN"):
    if input_data:
        digits = "".join(re.findall(r'\d', input_data))
        new_kỳs = [digits[i:i+5] for i in range(0, len(digits)-4, 5)]
        st.session_state.memory.extend(new_kỳs)
        st.session_state.memory = st.session_state.memory[-300:]

        if len(st.session_state.memory) >= 2:
            all_str = "".join(st.session_state.memory)
            counts = Counter(all_str)
            top_3 = counts.most_common(3)
            s1, s2, s3 = top_3[0][0], top_3[1][0], top_3[2][0]

            col1, col2 = st.columns(2)
            
            with col1:
                st.error("🎯 CHỐT CẶP 2 SỐ 5 TINH")
                st.subheader(f"Cặp 1: {s1} - {s2}")
                st.subheader(f"Cặp 2: {s1} - {s3}")
                st.subheader(f"Cặp 3: {s2} - {s3}")
                st.write("---")
                st.info("💡 Cách đánh: Cược cả 3 cặp để phủ xác suất cao nhất.")

            with col2:
                st.warning("💵 BẢNG VÀO TIỀN GỢI Ý")
                unit = von // 100 # Đơn vị cược cơ bản là 1% vốn
                if muc_cuoc == "Đều tay (1-1-1)":
                    st.write(f"Ván 1: Mỗi cặp {unit}đ")
                    st.write(f"Ván 2: Mỗi cặp {unit}đ")
                elif muc_cuoc == "Gấp thếp nhẹ (1-2-4)":
                    st.write(f"Ván 1: Mỗi cặp {unit}đ")
                    st.write(f"Ván 2: Mỗi cặp {unit*2}đ (Nếu ván 1 thua)")
                    st.write(f"Ván 3: Mỗi cặp {unit*4}đ (Nếu ván 2 thua)")
                
                st.success(f"Dự kiến lãi mỗi ván: ~{unit * 2}đ")

            # Nhận diện nhịp cầu bệt
            recent_data = "".join(st.session_state.memory[-5:])
            for num in s1+s2+s3:
                if recent_data.count(num) >= 4:
                    st.toast(f"Cảnh báo: Số {num} đang BỆT rất mạnh!", icon="🚨")
        else:
            st.error("Hãy nạp thêm dữ liệu để AI tính toán!")
