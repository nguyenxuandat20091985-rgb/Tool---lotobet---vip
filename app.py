import streamlit as st
import pandas as pd
import re
import datetime

# --- THIẾT KẾ GIAO DIỆN CHUYÊN NGHIỆP (ULTIMATE UI) ---
st.set_page_config(page_title="LOTOBET ELITE v2.5", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'Roboto Mono', monospace; }
    
    /* Thẻ dự đoán chính phong cách Glassmorphism */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px);
        margin-bottom: 20px;
    }
    
    .main-number {
        font-size: 100px !important;
        font-weight: 700;
        color: #FF3131;
        text-shadow: 0 0 20px rgba(255, 49, 49, 0.5);
        margin: 0px !important;
    }

    /* Tối ưu hóa bảng - Chữ trắng tinh khôi trên nền đậm */
    .stTable { 
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    table { width: 100% !important; color: #FFFFFF !important; background-color: #111 !important; }
    thead tr th { background-color: #222 !important; color: #00FFC2 !important; font-size: 14px !important; }
    tbody tr td { border-bottom: 1px solid #222 !important; font-size: 16px !important; padding: 12px !important; text-align: center !important; }
    
    /* Nút bấm High-tech */
    .stButton>button {
        background: linear-gradient(45deg, #FF3131, #8B0000);
        color: white; border: none; border-radius: 10px;
        font-weight: bold; letter-spacing: 1px; height: 50px;
        transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

if 'raw_data' not in st.session_state: st.session_state.raw_data = []
if 'history' not in st.session_state: st.session_state.history = []

# --- HEADER CHUYÊN NGHIỆP ---
st.markdown("<h2 style='text-align: center; color: #00FFC2;'>💎 LOTOBET ELITE v2.5</h2>", unsafe_allow_html=True)

# --- NHẬP LIỆU GỌN GÀNG ---
with st.expander("🔌 HỆ THỐNG NẠP DỮ LIỆU", expanded=not st.session_state.raw_data):
    input_data = st.text_area("Dán chuỗi dữ liệu kết quả:", height=100, placeholder="Ví dụ: 12345 67890 11223...")
    if st.button("KÍCH HOẠT PHÂN TÍCH"):
        clean = re.findall(r'\b\d{5}\b', input_data)
        if clean:
            st.session_state.raw_data = clean
            st.rerun()

# --- HIỂN THỊ KẾT QUẢ ELITE ---
if st.session_state.raw_data:
    data_list = [[int(d) for d in list(s)] for s in st.session_state.raw_data]
    
    results = []
    for n in range(10):
        gap = 0
        for p in reversed(data_list):
            if n in p: break
            gap += 1
        recent = (sum(1 for p in data_list[-10:] if n in p) / 10) * 100
        total_freq = (sum(1 for p in data_list if n in p) / len(data_list)) * 100
        score = (recent * 0.5) + (total_freq * 0.2) + (min(gap * 8, 30))
        
        indicator = "🔴 MẠNH" if score > 75 else "🟡 KHÁ" if score > 60 else "⚪ CHỜ"
        results.append({"SỐ": n, "TỶ LỆ %": round(min(score, 98.9), 1), "TRỄ": gap, "TÍN HIỆU": indicator})
    
    sorted_res = sorted(results, key=lambda x: x['TỶ LỆ %'], reverse=True)
    best = sorted_res[0]

    # Dashboard dự đoán chính
    st.markdown(f"""
        <div class="premium-card">
            <div style="color: #00FFC2; font-size: 16px; font-weight: bold; letter-spacing: 2px;">DỰ ĐOÁN KỲ TIẾP THEO</div>
            <div class="main-number">{best['SỐ']}</div>
            <div style="color: #FFFFFF; font-size: 18px;">ĐỘ TIN CẬY: <span style="color: #FF3131; font-weight: bold;">{best['TỶ LỆ %']}%</span></div>
            <div style="margin-top: 10px; font-size: 14px; color: #aaa;">Tín hiệu: {best['TÍN HIỆU']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Bảng chi tiết sắc nét
    st.markdown("<p style='text-align: center; color: #00FFC2; margin-bottom: 5px;'>📊 MA TRẬN PHÂN TÍCH CHI TIẾT</p>", unsafe_allow_html=True)
    df_display = pd.DataFrame(sorted_res)
    st.table(df_display)

    # Hệ thống đối soát chuyên nghiệp
    with st.sidebar:
        st.markdown("<h3 style='color: #FF3131;'>🎯 ĐỐI SOÁT</h3>", unsafe_allow_html=True)
        actual = st.text_input("Kết quả vừa về:", key="actual_input")
        if st.button("GHI NHẬT KÝ"):
            if len(actual) == 5:
                win = str(best['SỐ']) in actual
                st.session_state.history.insert(0, {"Số": best['SỐ'], "Kết quả": actual, "KQ": "WIN ✅" if win else "LOSE ❌"})
                st.rerun()
        
        if st.session_state.history:
            st.markdown("---")
            st.markdown("<p style='color: #00FFC2;'>📜 LỊCH SỬ GẦN ĐÂY</p>", unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state.history).head(5))

# Footer
st.markdown("<p style='text-align: center; color: #333; font-size: 10px;'>Elite Algorithm v2.5 - Professional Grade</p>", unsafe_allow_html=True)
