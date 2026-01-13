import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import pytesseract
from PIL import Image
import os
from datetime import datetime

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="LOTOBET V3 PRO - ALL IN ONE", layout="wide")
DATA_FILE = "loto_data.csv"
PREDICT_FILE = "predict_history.csv"

# Khởi tạo file dữ liệu nếu chưa có
for f in [DATA_FILE, PREDICT_FILE]:
    if not os.path.exists(f):
        pd.DataFrame().to_csv(f, index=False)

# --- 1. MODULE MẮT THẦN (OCR) ---
def process_image(image_bytes):
    """Quét ảnh từ Screenshot để lấy dãy 5 số"""
    try:
        # Chuyển bytes ảnh sang định dạng OpenCV
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tiền xử lý để đọc số chính xác hơn (Thresholding)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Cấu hình OCR chỉ đọc số
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Lọc các dãy 5 số (5 tinh)
        numbers = [n for n in text.split() if len(n) == 5]
        return numbers
    except Exception as e:
        st.error(f"Lỗi OCR: {e}")
        return []

# --- 2. MODULE BỘ NÃO (ANALYZER) ---
def analyze_logic(df):
    """Phân tích nhịp rơi và cầu bệt theo biểu đồ"""
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    
    stats = []
    total_len = len(df)
    
    for n in range(10):
        target = str(n)
        # Tìm các kỳ số n xuất hiện trong chuỗi 5 số
        indices = df.index[df['numbers'].astype(str).str.contains(target)].tolist()
        
        if not indices:
            stats.append({"Số": n, "Nhịp Hiện Tại": total_len, "Điểm": 0, "Trạng Thái": "Đang Gan"})
            continue
            
        # Tính khoảng cách (Gap)
        gaps = [indices[i] - indices[i-1] for i in range(1, len(indices))]
        current_gap = (total_len - 1) - indices[-1]
        avg_gap = sum(gaps) / len(gaps) if gaps else 5
        
        # Tính điểm tin cậy (Kết hợp nhịp rơi và tần suất)
        # Nếu nhịp hiện tại gần bằng nhịp trung bình -> Điểm cao
        gap_score = max(0, 100 - abs(current_gap - avg_gap) * 15)
        freq_score = (len(indices) / total_len) * 100
        
        final_score = (gap_score * 0.7) + (freq_score * 0.3)
        
        stats.append({
            "Số": n,
            "Nhịp TB": round(avg_gap, 1),
            "Nhịp Hiện Tại": current_gap,
            "Điểm Tin Cậy": round(final_score, 2),
            "Trạng Thái": "🔥 Vào Nhịp" if current_gap >= avg_gap - 1 else "Chờ"
        })
    
    return pd.DataFrame(stats).sort_values("Điểm Tin Cậy", ascending=False)

# --- 3. GIAO DIỆN (UI/UX) ---
st.title("🛡️ LOTOBET HYBRID V3 - TRỢ LÝ DỮ LIỆU CHUYÊN NGHIỆP")
st.markdown("---")

col_input, col_view = st.columns([1, 2])

with col_input:
    st.subheader("📥 Nhập liệu thông minh")
    tab1, tab2 = st.tabs(["Quét Ảnh (OCR)", "Nhập Tay"])
    
    with tab1:
        up_img = st.file_uploader("Upload ảnh kết quả", type=['jpg', 'png'])
        if up_img:
            extracted = process_image(up_img.read())
            if extracted:
                st.success(f"Tìm thấy: {extracted}")
                if st.button("Lưu vào Data"):
                    new_data = pd.DataFrame({"time": [datetime.now()], "numbers": [",".join(extracted)]})
                    new_data.to_csv(DATA_FILE, mode='a', header=not os.path.exists(DATA_FILE), index=False)
                    st.rerun()

    with tab2:
        manual_input = st.text_input("Nhập dãy 5 số (VD: 57221)")
        if st.button("Thêm thủ công"):
            if len(manual_input) == 5:
                new_data = pd.DataFrame({"time": [datetime.now()], "numbers": [manual_input]})
                new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
                st.success("Đã thêm!")
            else: st.error("Phải đủ 5 số!")

# --- 4. HIỂN THỊ KẾT QUẢ PHÂN TÍCH ---
df_main = pd.read_csv(DATA_FILE)
if not df_main.empty:
    analysis_res = analyze_logic(df_main)
    
    with col_view:
        st.subheader("📊 Biểu đồ Nhịp rơi & Độ nóng")
        fig = px.bar(analysis_res, x='Số', y='Điểm Tin Cậy', color='Điểm Tin Cậy', 
                     color_continuous_scale='Turbo', text='Điểm Tin Cậy')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gợi ý dàn số dựa trên điểm cao nhất
        top_3 = analysis_res.head(3)['Số'].tolist()
        st.warning(f"💡 GỢI Ý DÀN (2 số 5 tinh): **{top_3}** | Tỷ lệ đề xuất: **6.61**")

    st.divider()
    
    # Bảng chi tiết
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📋 Chi tiết thông số")
        st.dataframe(analysis_res, use_container_width=True)
    with c2:
        st.subheader("🕒 Lịch sử kỳ gần nhất")
        st.table(df_main.tail(5))

else:
    st.info("Chưa có dữ liệu. Vui lòng upload ảnh hoặc nhập tay để bắt đầu phân tích.")

# --- QUẢN LÝ VỐN ---
st.sidebar.header("💰 Quản lý vốn")
balance = st.sidebar.number_input("Số dư hiện tại", value=1000)
bet_unit = st.sidebar.number_input("Tiền cược 1 đơn", value=10)
st.sidebar.info(f"Khuyến nghị cược: {round(balance * 0.02)} - {round(balance * 0.05)} (2-5% vốn)")
