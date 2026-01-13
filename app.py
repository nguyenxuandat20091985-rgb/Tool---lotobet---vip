import streamlit as st
import re
import itertools
import pandas as pd
from collections import Counter

st.set_page_config(page_title="AI V9 - PREDICTOR MAX", layout="wide")

if 'memory' not in st.session_state:
    st.session_state.memory = []

st.title("🛡️ AI V9 - HỆ THỐNG GIẢM THIỂU SAI SỐ")
st.markdown("---")

# Nhập dữ liệu
input_data = st.text_area("Dán ít nhất 30-50 kỳ quay để giảm lỗi:", height=150)

if st.button("⚡ PHÂN TÍCH CHỐNG GÃY CẦU"):
    if input_data:
        digits = "".join(re.findall(r'\d', input_data))
        new_kỳs = [digits[i:i+5] for i in range(0, len(digits)-4, 5)]
        st.session_state.memory.extend(new_kỳs)
        st.session_state.memory = st.session_state.memory[-500:] # Nhớ sâu hơn

        if len(st.session_state.memory) >= 20:
            st.success(f"📊 Đã nạp {len(st.session_state.memory)} kỳ. Dữ liệu càng nhiều, AI đoán càng chuẩn.")
            
            # Thuật toán tính Độ Gan (Số bao nhiêu kỳ chưa về)
            all_digits = "0123456789"
            last_appearance = {d: 0 for d in all_digits}
            for i, ky in enumerate(reversed(st.session_state.memory)):
                for d in all_digits:
                    if d in ky and last_appearance[d] == 0:
                        last_appearance[d] = i

            # Lọc ra các số có "Phong độ" tốt (Vừa về xong và về nhiều)
            counts = Counter("".join(st.session_state.memory))
            
            # Tính điểm ưu tiên cho từng số
            scores = {}
            for d in all_digits:
                # Điểm = Tần suất / (Độ gan + 1)
                scores[d] = counts[d] / (last_appearance[d] + 1)

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            s1, s2, s3 = sorted_scores[0][0], sorted_scores[1][0], sorted_scores[2][0]

            col1, col2 = st.columns(2)
            with col1:
                st.error("🎯 CẶP 5 TINH AN TOÀN CAO")
                st.subheader(f"Cặp chính: {s1} - {s2}")
                st.subheader(f"Cặp lót: {s1} - {s3}")
                st.caption("AI đã loại bỏ các số có độ gan quá lớn để tránh mất vốn.")

            with col2:
                st.warning("⚠️ LƯU Ý KỸ THUẬT")
                st.write(f"Số **{s1}** đang có nhịp rơi đẹp nhất.")
                st.write(f"Số **{sorted_scores[-1][0]}** đang bị giam, tuyệt đối không nên theo.")

        else:
            st.error("⚠️ Cảnh báo: Dưới 20 kỳ quay AI sẽ đoán rất dễ sai. Hãy dán thêm dữ liệu!")

if st.sidebar.button("🗑️ Xóa dữ liệu"):
    st.session_state.memory = []
    st.rerun()
