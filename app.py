import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import Counter
import datetime

# --- CẤU HÌNH GIAO DIỆN ELITE v4.0 (TỐI ƯU MOBILE ANDROID) ---
st.set_page_config(page_title="LOTOBET ELITE v4.0", layout="wide")

st.markdown("""
    <style>
    /* Tổng thể nền đen và font chữ sáng nét */
    .stApp { background-color: #050505; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
    
    /* Thẻ Neon Glassmorphism cho con số dự đoán */
    .card-elite {
        background: rgba(26, 26, 26, 0.98);
        border: 2px solid #FF3131; border-radius: 15px;
        padding: 20px; text-align: center; margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(255, 49, 49, 0.3);
    }
    
    /* Con số Neon đỏ rực */
    .main-num { font-size: 70px; font-weight: bold; color: #FF3131; text-shadow: 0 0 15px #FF3131; line-height: 1.1; margin: 5px 0; }
    
    /* Thanh xác suất (Progress Bar) tùy chỉnh bằng CSS */
    .progress-bg { background: #222; border-radius: 10px; height: 14px; width: 100%; margin: 15px 0; border: 1px solid #444; }
    .progress-fill { background: linear-gradient(90deg, #FF3131, #00FFC2); height: 12px; border-radius: 10px; transition: 1s ease-in-out; }
    
    /* Bảng hiển thị sắc nét cho điện thoại */
    table { width: 100% !important; background: #000 !important; color: white !important; border-collapse: collapse; }
    th { color: #00FFC2 !important; background: #1A1A1A !important; padding: 12px !important; border-bottom: 2px solid #333 !important; font-size: 14px; }
    td { padding: 15px !important; border-bottom: 1px solid #222 !important; text-align: center !important; font-weight: bold !important; font-size: 16px; }
    
    /* Tối ưu hóa các Tab Menu */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161B22; border-radius: 8px 8px 0 0; color: #999; border: none; }
    .stTabs [aria-selected="true"] { background-color: #FF3131 !important; color: white !important; font-weight: bold; }
    
    /* Nút bấm cảm ứng High-tech */
    .stButton>button { width: 100%; border-radius: 10px; background: linear-gradient(90deg, #FF3131, #8B0000); color: white; font-weight: bold; border: none; height: 55px; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo Session State để lưu trữ dữ liệu
if 'data_history' not in st.session_state: st.session_state.data_history = []
if 'log_entries' not in st.session_state: st.session_state.log_entries = []

# --- HEADER STATUS ---
st.markdown("<div style='display:flex; justify-content:space-between; font-size:12px; color:#666; padding:5px;'><span>📡 AI ENGINE v4.0 ACTIVE</span><span>⚡ BATTERY SAFE</span><span>🛡️ SECURE</span></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#FF3131; margin-top:0;'>🛡️ LOTOBET ELITE v4.0</h1>", unsafe_allow_html=True)

# --- CẤU TRÚC 3 TAB CHÍNH ---
tab_input, tab_predict, tab_detail = st.tabs(["📥 NẠP KỲ & LƯU", "🎯 TRUNG TÂM DỰ ĐOÁN", "📊 CHI TIẾT 0-9"])

# --- TAB 1: NẠP DỮ LIỆU & NHẬT KÝ ---
with tab_input:
    input_text = st.text_area("Dán danh sách kết quả (5 số mỗi dòng):", height=150, placeholder="Ví dụ:\n12345\n67890\n88273")
    if st.button("🚀 KÍCH HOẠT PHÂN TÍCH"):
        # Lọc sạch dữ liệu, chỉ lấy các cụm 5 chữ số
        clean_data = re.findall(r'\b\d{5}\b', input_text)
        if clean_data:
            st.session_state.data_history = clean_data
            st.toast(f"Đã nạp {len(clean_data)} kỳ thành công!", icon="✅")
            st.rerun()
    
    st.markdown("---")
    st.subheader("📜 NHẬT KÝ ĐỐI SOÁT")
    c1, c2 = st.columns([3, 1])
    with c1:
        last_res = st.text_input("KQ vừa về:", key="last_res_val", placeholder="12345")
    with c2:
        if st.button("LƯU"):
            if len(last_res) == 5:
                st.session_state.log_entries.insert(0, {"Giờ": datetime.datetime.now().strftime("%H:%M"), "Số về": last_res})
                st.toast("Đã lưu lịch sử!", icon="📝")
                st.rerun()
    if st.session_state.log_entries:
        st.table(pd.DataFrame(st.session_state.log_entries).head(5))

# --- TAB 2: TRUNG TÂM DỰ ĐOÁN (3 TAB PHỤ) ---
with tab_predict:
    if not st.session_state.data_history:
        st.info("👈 Vui lòng nạp dữ liệu ở Tab 📥 để bắt đầu.")
    else:
        # Xử lý ma trận dữ liệu tập hợp (sets)
        data_sets = [set(map(int, list(s))) for s in st.session_state.data_history]
        
        # 3 Tab phụ theo yêu cầu
        sub1, sub2, sub3 = st.tabs(["🎯 1 SỐ", "⚔️ XIÊN 2", "🔥 XIÊN 3"])
        
        with sub1:
            all_digits = [n for s in data_sets for n in s]
            best_1 = Counter(all_digits).most_common(1)[0]
            # Thuật toán tính độ tin cậy dựa trên tần suất (max 98.9%)
            conf = min((best_1[1] / len(data_sets)) * 260, 98.9)
            st.markdown(f"""
                <div class='card-elite'>
                    <div style='color:#00FFC2; font-size:16px; letter-spacing:2px; font-weight:bold;'>BẠCH THỦ TIỀM NĂNG</div>
                    <div class='main-num'>{best_1[0]}</div>
                    <div class='progress-bg'><div class='progress-fill' style='width:{conf}%;'></div></div>
                    <div style='color:#DDD; font-size:14px;'>Độ tin cậy xác suất: <b>{conf:.1f}%</b></div>
                </div>
            """, unsafe_allow_html=True)

        with sub2:
            st.markdown("<p style='text-align:center; color:#00FFC2; font-weight:bold;'>TOP 3 CẶP XIÊN 2 HỘI TỤ</p>", unsafe_allow_html=True)
            c2_list = []
            for s in data_sets: c2_list.extend(combinations(sorted(s), 2))
            for combo, count in Counter(c2_list).most_common(3):
                st.markdown(f"""
                    <div class='card-elite'>
                        <div class='main-num' style='font-size:45px;'>{combo[0]} - {combo[1]}</div>
                        <div style='color:#00FFC2; font-size:14px;'>Điểm hội tụ: {count} kỳ đồng hành</div>
                    </div>
                """, unsafe_allow_html=True)

        with sub3:
            st.markdown("<p style='text-align:center; color:#FFFF00; font-weight:bold;'>TOP 2 BỘ XIÊN 3 HỘI TỤ MẠNH</p>", unsafe_allow_html=True)
            c3_list = []
            for s in data_sets: c3_list.extend(combinations(sorted(s), 3))
            for combo, count in Counter(c3_list).most_common(2):
                st.markdown(f"""
                    <div class='card-elite'>
                        <div class='main-num' style='font-size:38px;'>{combo[0]} - {combo[1]} - {combo[2]}</div>
                        <div style='color:#FFFF00; font-size:14px;'>Điểm hội tụ: {count} kỳ đồng hành</div>
                    </div>
                """, unsafe_allow_html=True)

# --- TAB 3: CHI TIẾT 0-9 ---
with tab_detail:
    if st.session_state.data_history:
        matrix_data = []
        for n in range(10):
            gap = 0
            # Tìm độ trễ (Gap)
            for p in reversed(data_sets):
                if n in p: break
                gap += 1
            # Tần suất xuất hiện
            freq = sum(1 for s in data_sets if n in s)
            # Tín hiệu AI cảnh báo
            signal = "🔥 MẠNH" if gap > 7 else "⚡ KHÁ" if freq > 15 else "⏱️ CHỜ"
            matrix_data.append({"SỐ": n, "TẦN SUẤT": freq, "ĐỘ TRỄ": gap, "TÍN HIỆU": signal})
        
        st.table(pd.DataFrame(matrix_data).sort_values(by="TẦN SUẤT", ascending=False))
    else:
        st.info("Dữ liệu phân tích chi tiết sẽ hiển thị tại đây.")

# --- FOOTER ---
st.markdown("<p style='text-align:center; color:#444; font-size:10px; margin-top:30px;'>ELITE HYBRID v4.0 | No-Graphics Optimized for Android</p>", unsafe_allow_html=True)
