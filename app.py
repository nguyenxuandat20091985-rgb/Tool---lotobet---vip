import streamlit as st
import pandas as pd
import re
import datetime

# --- CẤU HÌNH GIAO DIỆN SIÊU GỌN ---
st.set_page_config(page_title="LOTOBET v2.2", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    /* Khung dự đoán nhỏ gọn lại */
    .result-box { 
        background: #161B22; 
        padding: 15px; border-radius: 12px; 
        text-align: center; border: 1px solid #ff4b4b;
        margin-bottom: 10px;
    }
    h1 { font-size: 60px !important; margin: 0px !important; color: #ff4b4b; }
    h2 { font-size: 20px !important; margin: 0px !important; }
    h3 { font-size: 18px !important; margin: 0px !important; color: #00ff00; }
    /* Làm rõ bảng chi tiết */
    .stDataFrame, .stTable { 
        background-color: #1F2937 !important; 
        border-radius: 8px;
    }
    th { background-color: #374151 !important; color: white !important; }
    td { color: #FFFFFF !important; font-weight: 500 !important; border-bottom: 1px solid #374151 !important; }
    /* Nút bấm gọn */
    .stButton>button { width: 100%; height: 45px; border-radius: 8px; background: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if 'raw_data' not in st.session_state: st.session_state.raw_data = []
if 'history' not in st.session_state: st.session_state.history = []

# --- PHẦN NHẬP LIỆU (ẨN GỌN SAU KHI CÓ DỮ LIỆU) ---
with st.expander("📥 NHẬP DỮ LIỆU (Bấm để mở)", expanded=not st.session_state.raw_data):
    input_data = st.text_area("Dán kết quả Ku:", height=100)
    if st.button("🚀 PHÂN TÍCH"):
        clean = re.findall(r'\b\d{5}\b', input_data)
        if clean:
            st.session_state.raw_data = clean
            st.rerun()

# --- PHẦN HIỂN THỊ KẾT QUẢ ---
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
        results.append({"SỐ": n, "XÁC SUẤT": round(min(score, 98.9), 1), "TRỄ": gap, "LỆNH": "🔥" if score > 75 else "⚡" if score > 60 else "⏱️"})
    
    best = sorted(results, key=lambda x: x['XÁC SUẤT'], reverse=True)[0]

    # Khung dự đoán mini
    st.markdown(f"""
        <div class="result-box">
            <h2>SỐ TIỀM NĂNG</h2>
            <h1>{best['SỐ']}</h1>
            <h3>TỶ LỆ: {best['XÁC SUẤT']}%</h3>
        </div>
    """, unsafe_allow_html=True)

    # Bảng chi tiết (Dùng st.table để hiển thị rõ nét nhất trên mobile)
    st.markdown("### 📊 CHI TIẾT 0-9")
    df_display = pd.DataFrame(results).sort_values(by="XÁC SUẤT", ascending=False)
    st.table(df_display)

    # Đối soát gọn
    with st.expander("📝 ĐỐI SOÁT & NHẬT KÝ"):
        actual = st.text_input("Kết quả vừa về:", placeholder="12345")
        if st.button("LƯU KẾT QUẢ"):
            if len(actual) == 5:
                is_win = str(best['SỐ']) in actual
                st.session_state.history.insert(0, {"Số": best['SỐ'], "Về": actual, "KQ": "✅" if is_win else "❌"})
                st.rerun()
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history).head(5))
