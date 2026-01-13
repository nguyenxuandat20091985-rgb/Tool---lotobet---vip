import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from datetime import datetime

# --- KHỞI TẠO CẤU HÌNH ---
st.set_page_config(page_title="LOTOBET V3 PRO", layout="wide")

# Kiểm tra thư viện tesseract (cần thiết cho OCR)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

DATA_FILE = "loto_data.csv"
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=['time', 'numbers']).to_csv(DATA_FILE, index=False)

# --- HÀM XỬ LÝ NHẬN DIỆN ẢNH (OCR) ---
def scan_results_from_image(image):
    try:
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Xử lý ảnh để làm nổi bật số mở thưởng
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        text = pytesseract.image_to_string(thresh, config='--psm 6 digits')
        # Lọc các dãy 5 số từ kết quả mở thưởng
        found = [n for n in text.split() if len(n) == 5]
        return found
    except Exception as e:
        st.error(f"Lỗi quét ảnh: {e}")
        return []

# --- THUẬT TOÁN SOI CẦU NHỊP RƠI ---
def calculate_trends(df):
    if df.empty: return pd.DataFrame()
    results = []
    # Phân tích từ số 0 đến 9
    for n in range(10):
        target = str(n)
        # Kiểm tra sự xuất hiện trong lịch sử
        appearances = df.index[df['numbers'].str.contains(target)].tolist()
        
        if not appearances:
            gap = len(df)
            score = 0
        else:
            gap = len(df) - 1 - appearances[-1]
            # Tính nhịp trung bình (Gap analysis)
            intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
            avg_gap = sum(intervals) / len(intervals) if intervals else 5
            # Điểm tin cậy dựa trên độ nóng và nhịp rơi
            score = max(0, 100 - abs(gap - avg_gap) * 12)
            
        results.append({"Số": n, "Độ Gan (Gap)": gap, "Điểm Tin Cậy": round(score, 2)})
    return pd.DataFrame(results).sort_values("Điểm Tin Cậy", ascending=False)

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ TRỢ LÝ LOTOBET V3 - PHÂN TÍCH 2 SỐ 5 TINH")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 Cập nhật dữ liệu")
    uploaded_file = st.file_uploader("Gửi ảnh kết quả mới nhất", type=['jpg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh đã tải lên", width=250)
        if st.button("Bắt đầu quét số"):
            data = scan_results_from_image(img)
            if data:
                st.success(f"Đã tìm thấy: {data}")
                for d in data:
                    new_row = pd.DataFrame({'time': [datetime.now()], 'numbers': [d]})
                    new_row.to_csv(DATA_FILE, mode='a', header=False, index=False)
                st.rerun()

with col2:
    df_history = pd.read_csv(DATA_FILE)
    if not df_history.empty:
        analysis = calculate_trends(df_history)
        
        # Biểu đồ nhịp rơi (Sửa lỗi Plotly)
        st.subheader("📊 Biểu đồ Nhịp Rơi (Trend-line)")
        fig = px.bar(analysis, x='Số', y='Điểm Tin Cậy', color='Điểm Tin Cậy', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gợi ý dàn số (Tỷ lệ 6.61)
        top_nums = analysis.head(4)['Số'].tolist()
        st.warning(f"💡 Dàn đề xuất (Đánh 2 số 5 tinh): **{top_nums}**")
    else:
        st.info("Hãy tải ảnh kết quả hoặc nhập dữ liệu để bắt đầu phân tích.")

st.subheader("🕒 Lịch sử kỳ mở thưởng gần nhất")
st.dataframe(df_history.tail(10), use_container_width=True)
