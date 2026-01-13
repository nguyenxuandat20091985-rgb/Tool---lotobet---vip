import streamlit as st
import re
import itertools
import pandas as pd
from collections import Counter

st.set_page_config(page_title="AI LOTOBET V7 - 5 TINH", layout="wide")

if 'memory' not in st.session_state:
    st.session_state.memory = []

st.title("🚀 SIÊU AI V7 - CHUYÊN GIA SOI CẶP 2 SỐ 5 TINH")
st.markdown("---")

input_data = st.text_area("Dán dữ liệu kết quả tại đây:", height=150)

if st.button("🔥 PHÂN TÍCH CẶP 5 TINH"):
    if input_data:
        digits = "".join(re.findall(r'\d', input_data))
        new_kỳs = [digits[i:i+5] for i in range(0, len(digits)-4, 5)]
        st.session_state.memory.extend(new_kỳs)
        st.session_state.memory = st.session_state.memory[-300:]

        if len(st.session_state.memory) >= 2:
            st.success(f"📊 Đã nạp {len(st.session_state.memory)} kỳ quay vào bộ nhớ AI.")
            
            # --- THUẬT TOÁN SOI CẶP 5 TINH ---
            all_pairs = []
            for ky in st.session_state.memory:
                # Lấy các số độc nhất trong 1 kỳ (vì quy tắc 2 số 5 tinh chỉ cần xuất hiện)
                unique_nums = sorted(list(set(ky)))
                # Tạo các cặp kết hợp (ví dụ kỳ 12121 -> có số 1 và 2 -> cặp 1-2)
                pairs = list(itertools.combinations(unique_nums, 2))
                all_pairs.extend(pairs)
            
            pair_counts = Counter(all_pairs).most_common(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.error("💎 TOP 5 CẶP 2 SỐ 5 TINH (Hay về cùng nhau nhất)")
                for pair, count in pair_counts:
                    st.subheader(f"Cặp: {pair[0]} - {pair[1]}")
                    st.write(f"Đã xuất hiện cùng nhau {count} lần")

            with col2:
                st.warning("🔮 DỰ ĐOÁN KỲ TIẾP THEO")
                # Lấy 3 số đơn lẻ về nhiều nhất để gợi ý cặp xoay vòng
                all_digits = "".join(st.session_state.memory)
                top_3_single = Counter(all_digits).most_common(3)
                s1, s2, s3 = top_3_single[0][0], top_3_single[1][0], top_3_single[2][0]
                
                st.info("Gợi ý tổ hợp 2 số 5 tinh:")
                st.code(f"Cặp chính: {s1}, {s2}", language="text")
                st.code(f"Cặp lót 1: {s1}, {s3}", language="text")
                st.code(f"Cặp lót 2: {s2}, {s3}", language="text")

            # --- NHẬN DIỆN CẦU BỆT 5 TINH ---
            st.markdown("---")
            recent_ky = st.session_state.memory[-1]
            st.write(f"Kỳ gần nhất: **{recent_ky}**")
            st.caption("AI khuyên: Nếu kỳ trước nổ bệt (ví dụ 12121), hãy ưu tiên đánh lại cặp của kỳ đó cho kỳ sau.")
        else:
            st.error("Cần thêm dữ liệu kỳ quay để phân tích cặp!")

if st.sidebar.button("🗑️ Xóa bộ nhớ AI"):
    st.session_state.memory = []
    st.rerun()
