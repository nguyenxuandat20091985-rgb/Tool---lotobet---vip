import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PIL import Image

# Thử import Plotly, nếu lỗi sẽ báo rõ cho người dùng
try:
    import plotly.express as px
except ImportError:
    st.error("Thiếu thư viện 'plotly'. Vui lòng thêm vào requirements.txt hoặc chạy 'pip install plotly'")

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="LOTOBET V3 PRO", layout="wide")
DATA_FILE = "loto_database.csv"

# Khởi tạo file lưu trữ nếu chưa có
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=['Thời gian', 'Kết quả']).to_csv(DATA_FILE, index=False)

# --- THUẬT TOÁN PHÂN TÍCH NHỊP (GAP ANALYSIS) ---
def analyze_trends(df):
    """Phân tích nhịp rơi từ dữ liệu thực tế"""
    if df.empty: return pd.DataFrame()
    
    results = []
    total_records = len(df)
    
    for n in range(10):
        digit = str(n)
        # Tìm các kỳ mà số này xuất hiện trong dãy 5 tinh
        appearances = df.index[df['Kết quả'].astype(str).str.contains(digit)].tolist()
        
        if not appearances:
            gap = total_records
            score = 0
        else:
            gap = (total_records - 1) - appearances[-1]
            # Tính khoảng cách giữa các lần xuất hiện (giống đường nối màu xanh trong ảnh)
            intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
            avg_interval = sum(intervals) / len(intervals) if intervals else 5
            
            # Tính điểm tin cậy: Ưu tiên số đang đến nhịp rơi trung bình
            score = max(0, 100 - abs(gap - avg_interval) * 15)
            
        results.append({
            "Số": n,
            "Nhịp hiện tại (Gap)": gap,
            "Điểm tin cậy": round(score, 2),
            "Trạng thái": "🔥 Chờ nổ" if gap >= 3 else "Đang chạy"
        })
    
    return pd.DataFrame(results).sort_values("Điểm tin cậy", ascending=False)

# --- GIAO DIỆN NGƯỜI DÙNG ---
st.title("📊 LOTOBET V3 - TRỢ LÝ SOI CẦU 2 SỐ 5 TINH")
st.info("Dựa trên dữ liệu thực tế từ bảng kết quả và biểu đồ nhịp rơi.")

col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("📥 Nhập dữ liệu kỳ mới")
    # Phương pháp nhập tay an toàn nhất khi OCR gặp lỗi thư viện
    raw_input = st.text_input("Nhập dãy 5 số (VD: 57221)", placeholder="Ví dụ: 01234")
    
    if st.button("Lưu kết quả"):
        if len(raw_input) == 5 and raw_input.isdigit():
            new_data = pd.DataFrame({'Thời gian': [datetime.now().strftime("%H:%M:%S")], 'Kết quả': [raw_input]})
            new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
            st.success(f"Đã lưu kỳ mới: {raw_input}")
            st.rerun()
        else:
            st.error("Vui lòng nhập đúng 5 chữ số!")

    st.divider()
    st.write("📖 **Quy tắc 2 số 5 tinh:** Chọn 2 số, chỉ cần xuất hiện trong 5 vị trí là thắng. Tỷ lệ ăn 6.61.")

with col_out:
    df_history = pd.read_csv(DATA_FILE)
    
    if not df_history.empty:
        analysis_data = analyze_trends(df_history)
        
        # Biểu đồ trực quan
        st.subheader("📈 Biểu đồ độ nóng & Nhịp rơi")
        fig = px.bar(analysis_res := analysis_data, x='Số', y='Điểm tin cậy', 
                     color='Điểm tin cậy', color_continuous_scale='Turbo',
                     labels={'Điểm tin cậy': 'Mức độ tiềm năng'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Gợi ý dàn số
        top_numbers = analysis_data.head(4)['Số'].tolist()
        st.success(f"🎯 **Gợi ý dàn 4 số tiềm năng:** {', '.join(map(str, top_numbers))}")
        
        with st.expander("Xem bảng chi tiết thông số"):
            st.table(analysis_data)
    else:
        st.warning("Chưa có dữ liệu để phân tích. Hãy nhập kỳ đầu tiên ở bên trái.")

st.subheader("🕒 Lịch sử 10 kỳ gần nhất")
if not df_history.empty:
    st.dataframe(df_history.tail(10), use_container_width=True)
