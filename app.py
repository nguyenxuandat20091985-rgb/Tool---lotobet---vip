import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import Counter
import datetime

# --- CẤU HÌNH GIAO DIỆN ELITE v4.1 PRO ---
st.set_page_config(page_title="LOTOBET AI v4.1 PRO", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
    .card-elite {
        background: rgba(26, 26, 26, 0.98);
        border: 2px solid #FF3131; border-radius: 15px;
        padding: 20px; text-align: center; margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(255, 49, 49, 0.3);
    }
    .main-num { font-size: 70px; font-weight: bold; color: #FF3131; text-shadow: 0 0 15px #FF3131; line-height: 1.1; margin: 5px 0; }
    .progress-bg { background: #222; border-radius: 10px; height: 14px; width: 100%; margin: 15px 0; border: 1px solid #444; }
    .progress-fill { background: linear-gradient(90deg, #FF3131, #00FFC2); height: 12px; border-radius: 10px; transition: 1s ease-in-out; }
    table { width: 100% !important; background: #000 !important; color: white !important; border-collapse: collapse; }
    th { color: #00FFC2 !important; background: #1A1A1A !important; padding: 12px !important; border-bottom: 2px solid #333 !important; }
    td { padding: 15px !important; border-bottom: 1px solid #222 !important; text-align: center !important; font-weight: bold !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161B22; border-radius: 8px 8px 0 0; color: #999; }
    .stTabs [aria-selected="true"] { background-color: #FF3131 !important; color: white !important; }
    .stButton>button { width: 100%; border-radius: 10px; background: linear-gradient(90deg, #FF3131, #8B0000); color: white; font-weight: bold; height: 50px; border: none; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo Session State
if 'data_history' not in st.session_state: st.session_state.data_history = []
if 'log_entries' not in st.session_state: st.session_state.log_entries = []

# --- HEADER ---
st.markdown("<div style='display:flex; justify-content:space-between; font-size:12px; color:#666; padding:5px;'><span>📡 AI ENGINE v4.1 PRO</span><span>⚡ FILE SYSTEM READY</span></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#FF3131; margin-top:0;'>🛡️ LOTOBET AI v4.1 PRO</h1>", unsafe_allow_html=True)

# --- 3 TAB CHÍNH ---
tab_input, tab_predict, tab_detail = st.tabs(["📥 QUẢN LÝ DỮ LIỆU", "🎯 DỰ ĐOÁN", "📊 CHI TIẾT 0-9"])

# --- TAB 1: NẠP KỲ, IMPORT/EXPORT ---
with tab_input:
    st.markdown("### 📥 NẠP DỮ LIỆU MỚI")
    input_text = st.text_area("Dán kết quả tay (5 số mỗi dòng):", height=100)
    if st.button("🚀 KÍCH HOẠT PHÂN TÍCH"):
        clean_data = re.findall(r'\b\d{5}\b', input_text)
        if clean_data:
            st.session_state.data_history = clean_data
            st.toast("Đã nạp dữ liệu tay!", icon="✅")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📂 IMPORT/EXPORT FILE")
    uploaded_file = st.file_uploader("Nạp dữ liệu từ file .txt hoặc .csv", type=['txt', 'csv'])
    if uploaded_file:
        file_contents = uploaded_file.read().decode("utf-8")
        clean_from_file = re.findall(r'\b\d{5}\b', file_contents)
        if st.button("XÁC NHẬN NẠP TỪ FILE"):
            st.session_state.data_history = clean_from_file
            st.toast(f"Đã nạp {len(clean_from_file)} kỳ!", icon="📂")
            st.rerun()

    st.markdown("---")
    st.subheader("📜 NHẬT KÝ ĐỐI SOÁT")
    c1, c2 = st.columns([3, 1])
    with c1:
        last_res = st.text_input("KQ thực tế vừa về:", key="last_res_val")
    with c2:
        if st.button("LƯU"):
            if len(last_res) == 5:
                st.session_state.log_entries.insert(0, {"Giờ": datetime.datetime.now().strftime("%H:%M"), "Số về": last_res})
                st.rerun()

    if st.session_state.log_entries:
        st.table(pd.DataFrame(st.session_state.log_entries).head(5))
        df_export = pd.DataFrame(st.session_state.log_entries)
        csv = df_export.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="💾 TẢI NHẬT KÝ (.CSV)", data=csv, file_name='nhat_ky_lotobet.csv', mime='text/csv')

# --- TAB 2: DỰ ĐOÁN (3 TAB PHỤ) ---
with tab_predict:
    if not st.session_state.data_history:
        st.info("👈 Hãy nạp dữ liệu ở Tab 📥")
    else:
        data_sets = [set(map(int, list(s))) for s in st.session_state.data_history]
        sub1, sub2, sub3 = st.tabs(["🎯 1 SỐ", "⚔️ XIÊN 2", "🔥 XIÊN 3"])
        
        with sub1:
            all_digits = [n for s in data_sets for n in s]
            best_1 = Counter(all_digits).most_common(1)[0]
            conf = min((best_1[1] / len(data_sets)) * 260, 98.9)
            st.markdown(f"<div class='card-elite'><div style='color:#00FFC2;'>BẠCH THỦ</div><div class='main-num'>{best_1[0]}</div><div class='progress-bg'><div class='progress-fill' style='width:{conf}%;'></div></div><div>Tỷ lệ: {conf:.1f}%</div></div>", unsafe_allow_html=True)

        with sub2:
            c2_list = []
            for s in data_sets: c2_list.extend(combinations(sorted(s), 2))
            for combo, count in Counter(c2_list).most_common(3):
                st.markdown(f"<div class='card-elite'><div class='main-num' style='font-size:45px;'>{combo[0]} - {combo[1]}</div><div>Hội tụ: {count} kỳ</div></div>", unsafe_allow_html=True)

        with sub3:
            c3_list = []
            for s in data_sets: c3_list.extend(combinations(sorted(s), 3))
            for combo, count in Counter(c3_list).most_common(2):
                st.markdown(f"<div class='card-elite'><div class='main-num' style='font-size:35px;'>{combo[0]}-{combo[1]}-{combo[2]}</div><div>Hội tụ: {count} kỳ</div></div>", unsafe_allow_html=True)

# --- TAB 3: CHI TIẾT 0-9 ---
with tab_detail:
    if st.session_state.data_history:
        matrix = []
        for n in range(10):
            gap = 0
            for p in reversed(data_sets):
                if n in p: break
                gap += 1
            freq = sum(1 for s in st.session_state.data_history if str(n) in s)
            matrix.append({"SỐ": n, "TẦN SUẤT": freq, "ĐỘ TRỄ": gap, "TÍN HIỆU": "🔥 MẠNH" if gap > 7 else "⏱️ CHỜ"})
        st.table(pd.DataFrame(matrix).sort_values(by="TẦN SUẤT", ascending=False))

st.markdown("<p style='text-align:center; color:#444; font-size:10px;'>v4.1 PRO Edition</p>", unsafe_allow_html=True)
