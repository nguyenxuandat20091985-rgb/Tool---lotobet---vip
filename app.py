import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import Counter
import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="LOTOBET AI PRO v2.0", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS cho giao diện sang trọng
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-card { background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%); border-radius: 15px; padding: 20px; border: 1px solid #444; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 8px; background: linear-gradient(90deg, #ff4b4b, #ff7575); color: white; font-weight: bold; border: none; }
    .status-win { color: #00ff00; font-weight: bold; }
    .status-loss { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE PHÂN TÍCH 7 TẦNG (ADVANCED) ---
class PremiumLotoEngine:
    def __init__(self, data):
        self.data = [[int(d) for d in list(s)] for s in data]
        self.flat_data = [n for p in self.data for n in p]
        self.total = len(self.data)

    def analyze_all(self):
        results = []
        for n in range(10):
            # L1: Tần suất tổng
            f_score = (sum(1 for p in self.data if n in p) / self.total) * 100
            # L2: Độ trễ (Gap)
            gap = 0
            for p in reversed(self.data):
                if n in p: break
                gap += 1
            # L3: Trend ngắn hạn (5 kỳ)
            recent = sum(1 for p in self.data[-5:] if n in p) / 5 * 100
            # L4: Ma trận tương quan (Correlation)
            cor_score = self.get_correlation(n)
            
            # L7: Tổng hợp trọng số (Ensemble logic)
            # Công thức tối ưu: Ưu tiên Gap khi đạt ngưỡng và Trend đang lên
            final_prob = (f_score * 0.2) + (recent * 0.4) + (min(gap * 7, 35)) + (cor_score * 0.05)
            
            # Gợi ý hành động
            action = "CHỜ ĐỢI"
            if final_prob > 75: action = "🔥 VÀO TIỀN"
            elif final_prob > 60: action = "⚡ THEO NHẸ"

            results.append({
                "SỐ": n,
                "XÁC SUẤT": round(min(final_prob, 98.2), 1),
                "ĐỘ TRỄ": gap,
                "TRẠNG THÁI": action
            })
        return sorted(results, key=lambda x: x['XÁC SUẤT'], reverse=True)

    def get_correlation(self, n):
        # Giả lập tầng tương quan đơn giản
        return self.flat_data.count(n) / len(self.flat_data) * 100

# --- GIAO DIỆN CHÍNH ---
def main():
    st.title("🛡️ LOTOBET AI v2.0 - PREMIUM")
    
    if 'history' not in st.session_state: st.session_state.history = []
    if 'raw_data' not in st.session_state: st.session_state.raw_data = []

    # Sidebar quản lý dữ liệu
    with st.sidebar:
        st.header("⚙️ CÀI ĐẶT")
        input_data = st.text_area("Dán 50-100 kỳ kết quả:", height=300, help="Mỗi dòng 5 chữ số")
        if st.button("NẠP & PHÂN TÍCH"):
            clean = re.findall(r'\b\d{5}\b', input_data)
            if clean:
                st.session_state.raw_data = clean
                st.success(f"Đã xử lý {len(clean)} kỳ")
                st.rerun()
            else:
                st.error("Dữ liệu sai định dạng!")

    if not st.session_state.raw_data:
        st.info("👈 Hãy dán dữ liệu vào thanh Menu bên trái để bắt đầu phân tích.")
        return

    engine = PremiumLotoEngine(st.session_state.raw_data)
    analysis = engine.analyze_all()
    best = analysis[0]

    # Layout chính
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="main-card">
            <h3 style='text-align: center;'>SỐ KHUYÊN DÙNG</h3>
            <h1 style='text-align: center; color: #ff4b4b; font-size: 80px;'>{best['SỐ']}</h1>
            <p style='text-align: center;'>Độ tin cậy: <b>{best['XÁC SUẤT']}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        # Form chốt kết quả
        st.subheader("📝 Đối soát nhanh")
        actual_num = st.text_input("Kết quả kỳ vừa rồi (5 số):")
        if st.button("XÁC NHẬN KẾT QUẢ"):
            if len(actual_num) == 5:
                win = str(best['SỐ']) in actual_num
                st.session_state.history.insert(0, {
                    "Thời gian": datetime.datetime.now().strftime("%H:%M"),
                    "Dự đoán": best['SỐ'],
                    "Kết quả": actual_num,
                    "Kết quả": "✅ THẮNG" if win else "❌ THUA"
                })
                st.rerun()

    with col2:
        st.subheader("📊 Bảng phân tích chi tiết (0-9)")
        df = pd.DataFrame(analysis)
        st.dataframe(df.style.background_gradient(subset=['XÁC SUẤT'], cmap='OrRd'), use_container_width=True)

        # Vẽ biểu đồ Radar/Heatmap đơn giản cho xác suất
        fig = go.Figure(go.Bar(
            x=[str(x['SỐ']) for x in analysis],
            y=[x['XÁC SUẤT'] for x in analysis],
            marker_color='#ff4b4b'
        ))
        fig.update_layout(title="Trực quan hóa lực nổ", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Tab lịch sử bên dưới
    st.markdown("---")
    st.subheader("📜 Nhật ký dự đoán")
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        st.table(h_df)
        
        # Tính tỷ lệ thực tế
        wins = sum(1 for x in st.session_state.history if "✅" in x["Kết quả"])
        st.metric("TỶ LỆ THẮNG THỰC TẾ (WIN RATE)", f"{(wins/len(st.session_state.history))*100:.1f}%")

if __name__ == "__main__":
    main()
