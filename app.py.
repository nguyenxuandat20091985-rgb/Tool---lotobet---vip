import streamlit as st
import re
from collections import Counter
import random

# Tối ưu giao diện cho điện thoại
st.set_page_config(page_title="LOTOBET AI PRO", layout="centered")

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🎯 AI LOTOBET PRO")

# --- HỆ THỐNG LỌC DỮ LIỆU THÔNG MINH ---
st.subheader("📥 Nhập kết quả (Mọi định dạng)")
input_raw = st.text_area("Dán chuỗi số tại đây (Ví dụ: 91043 34193...):", height=150)

if st.button("🔥 XỬ LÝ & PHÂN TÍCH"):
    # Thuật toán: Chỉ nhặt ra các chữ số, bỏ qua mọi khoảng trắng hay ký tự lạ
    clean_digits = re.findall(r'\d', input_raw)
    # Tự động gom 5 số thành 1 kỳ quay
    new_rows = [clean_digits[i:i+5] for i in range(0, len(clean_digits), 5) if len(clean_digits[i:i+5]) == 5]
    
    if new_rows:
        st.session_state.history = [[int(n) for n in r] for r in new_rows]
        st.success(f"✅ Đã nạp thành công {len(new_rows)} kỳ quay!")
    else:
        st.error("❌ Không tìm thấy bộ 5 số hợp lệ. Hãy kiểm tra lại!")

# --- DỰ ĐOÁN 2 TINH (SONG THỦ) ---
if len(st.session_state.history) >= 5:
    st.markdown("---")
    if st.button("🔮 DỰ ĐOÁN 2 TINH KỲ TIẾP"):
        all_nums = [n for r in st.session_state.history for n in r]
        hot = [i[0] for i in Counter(all_nums).most_common(6)]
        last = st.session_state.history[-1]
        pool = list(set(hot + last))
        
        st.subheader("🎯 3 BỘ SỐ VÀNG (XÁC SUẤT CAO):")
        cols = st.columns(3)
        res = []
        while len(res) < 3:
            pair = "".join(map(str, sorted(random.sample(pool, 2))))
            if pair not in res:
                res.append(pair)
                with cols[len(res)-1]:
                    st.success(f"**{pair}**")
