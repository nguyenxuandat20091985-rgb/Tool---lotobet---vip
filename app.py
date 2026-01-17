import streamlit as st
import pandas as pd
import re
import datetime

st.set_page_config(page_title="LOTOBET AI v2.0", layout="wide")

# Giao diện dán trực tiếp, không dùng SideBar để tránh lỗi cảm ứng
st.title("🛡️ LOTOBET AI v2.0 - FIX CẢM ỨNG")

if 'raw_data' not in st.session_state: st.session_state.raw_data = []
if 'history' not in st.session_state: st.session_state.history = []

# PHẦN 1: NHẬP DỮ LIỆU NGAY TẠI ĐÂY
st.markdown("### 📥 BƯỚC 1: DÁN DỮ LIỆU VÀO ĐÂY")
input_data = st.text_area("Dán 20-50 kỳ (5 số mỗi dòng) từ Ku:", height=150)

if st.button("👉 BẮT ĐẦU PHÂN TÍCH"):
    clean = re.findall(r'\b\d{5}\b', input_data)
    if clean:
        st.session_state.raw_data = clean
    else:
        st.error("Chưa tìm thấy dữ liệu 5 chữ số. Hãy kiểm tra lại!")

# PHẦN 2: KẾT QUẢ PHÂN TÍCH
if st.session_state.raw_data:
    data_list = [[int(d) for d in list(s)] for s in st.session_state.raw_data]
    
    # Tính toán nhanh số mạnh nhất
    results = []
    for n in range(10):
        gap = 0
        for p in reversed(data_list):
            if n in p: break
            gap += 1
        recent = sum(1 for p in data_list[-10:] if n in p) / 10 * 100
        score = (recent * 0.6) + (min(gap * 5, 40))
        results.append({"SỐ": n, "XÁC SUẤT": round(score, 1), "ĐỘ TRỄ": gap})
    
    analysis = sorted(results, key=lambda x: x['XÁC SUẤT'], reverse=True)
    best = analysis[0]

    st.markdown("---")
    st.markdown(f"""
        <div style="background: #1e1e2f; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #ff4b4b;">
            <h2 style="color: white;">SỐ MAY MẮN TIẾP THEO</h2>
            <h1 style="color: #ff4b4b; font-size: 80px;">{best['SỐ']}</h1>
            <p style="color: white;">Tỷ lệ nổ: {best['XÁC SUẤT']}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Chi tiết 0-9")
    st.table(pd.DataFrame(analysis))
