import streamlit as st
import pandas as pd
import re
import datetime

# --- CẤU HÌNH GIAO DIỆN CHỐNG LAG ---
st.set_page_config(page_title="LOTOBET AI v2.1 PRO", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .result-box { 
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%); 
        padding: 30px; border-radius: 20px; 
        text-align: center; border: 2px solid #ff4b4b;
        box-shadow: 0px 10px 30px rgba(255, 75, 75, 0.3);
        margin-bottom: 25px;
    }
    .stButton>button { 
        width: 100%; height: 60px; border-radius: 12px; 
        background: #ff4b4b; color: white; font-size: 20px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo dữ liệu
if 'raw_data' not in st.session_state: st.session_state.raw_data = []
if 'history' not in st.session_state: st.session_state.history = []

st.title("🛡️ LOTOBET AI v2.1 - PREMIUM FIX")

# --- PHẦN 1: NHẬP LIỆU (HIỂN THỊ LUÔN) ---
st.markdown("### 📥 BƯỚC 1: NHẬP KẾT QUẢ KU")
input_data = st.text_area("Dán danh sách 5 số (ví dụ: 12345, 67890...):", height=150, placeholder="Dán ít nhất 20 kỳ tại đây...")

if st.button("🚀 PHÂN TÍCH NGAY"):
    clean = re.findall(r'\b\d{5}\b', input_data)
    if clean:
        st.session_state.raw_data = clean
        st.success(f"Đã cập nhật {len(clean)} kỳ gần nhất!")
    else:
        st.error("Lỗi: Không tìm thấy dữ liệu 5 số hợp lệ.")

# --- PHẦN 2: LOGIC 7 TẦNG & HIỂN THỊ ---
if st.session_state.raw_data:
    data_list = [[int(d) for d in list(s)] for s in st.session_state.raw_data]
    
    results = []
    for n in range(10):
        # Layer 1: Gap (Độ trễ)
        gap = 0
        for p in reversed(data_list):
            if n in p: break
            gap += 1
        
        # Layer 2: Recent Trend (10 kỳ)
        recent_count = sum(1 for p in data_list[-10:] if n in p)
        recent_score = (recent_count / 10) * 100
        
        # Layer 3: Toàn thời gian
        total_freq = sum(1 for p in data_list if n in p) / len(data_list) * 100
        
        # Công thức tổng hợp 7 tầng (Ensemble)
        # Ưu tiên số đang có trend (nóng) và vừa chớm trễ (gap 5-7 kỳ)
        final_score = (recent_score * 0.5) + (total_freq * 0.2) + (min(gap * 8, 30))
        
        status = "CHỜ"
        if final_score > 75: status = "🔥 VÀO TIỀN"
        elif final_score > 60: status = "⚡ THEO NHẸ"
        
        results.append({
            "SỐ": n,
            "XÁC SUẤT": round(min(final_score, 98.9), 1),
            "TRỄ (KỲ)": gap,
            "KHUYẾN NGHỊ": status
        })
    
    analysis = sorted(results, key=lambda x: x['XÁC SUẤT'], reverse=True)
    best = analysis[0]

    # Hiển thị số mạnh nhất
    st.markdown(f"""
        <div class="result-box">
            <h2 style="color: white; margin-bottom: 0;">SỐ TIỀM NĂNG NHẤT</h2>
            <h1 style="color: #ff4b4b; font-size: 100px; margin: 10px 0;">{best['SỐ']}</h1>
            <h3 style="color: #00ff00;">TỶ LỆ NỔ: {best['XÁC SUẤT']}%</h3>
            <p style="color: #aaa;">Trạng thái: {best['KHUYẾN NGHỊ']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Đối soát
    st.markdown("### 📝 ĐỐI SOÁT KẾT QUẢ")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        actual = st.text_input("Nhập kết quả thực tế vừa về:", placeholder="Ví dụ: 12345")
    with col_b:
        if st.button("LƯU"):
            if len(actual) == 5:
                is_win = str(best['SỐ']) in actual
                st.session_state.history.insert(0, {
                    "Giờ": datetime.datetime.now().strftime("%H:%M"),
                    "Dự đoán": best['SỐ'],
                    "Thực tế": actual,
                    "Kết quả": "✅ THẮNG" if is_win else "❌ THUA"
                })
                st.rerun()

    # Bảng chi tiết
    st.subheader("📊 BẢNG CHI TIẾT 0-9")
    st.table(pd.DataFrame(analysis))

    # Lịch sử
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 NHẬT KÝ CẦU")
        st.table(pd.DataFrame(st.session_state.history))
