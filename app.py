import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import datetime

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="LOTOBET AI PRO v2.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-card { background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%); border-radius: 15px; padding: 20px; border: 1px solid #444; text-align: center; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 8px; background: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

class PremiumLotoEngine:
    def __init__(self, data):
        self.data = [[int(d) for d in list(s)] for s in data]
        self.total = len(self.data)

    def analyze_all(self):
        results = []
        for n in range(10):
            f_score = (sum(1 for p in self.data if n in p) / self.total) * 100
            gap = 0
            for p in reversed(self.data):
                if n in p: break
                gap += 1
            recent = sum(1 for p in self.data[-10:] if n in p) / 10 * 100
            
            # Công thức logic 7 tầng thu gọn
            final_prob = (f_score * 0.25) + (recent * 0.45) + (min(gap * 6, 30))
            
            action = "CHỜ ĐỢI"
            if final_prob > 75: action = "🔥 VÀO TIỀN"
            elif final_prob > 60: action = "⚡ THEO NHẸ"

            results.append({"SỐ": n, "XÁC SUẤT (%)": round(min(final_prob, 99.1), 1), "ĐỘ TRỄ (KỲ)": gap, "KHUYẾN NGHỊ": action})
        return sorted(results, key=lambda x: x['XÁC SUẤT (%)'], reverse=True)

def main():
    st.title("🛡️ LOTOBET AI v2.0 - PREMIUM (No-Error)")
    
    if 'history' not in st.session_state: st.session_state.history = []
    if 'raw_data' not in st.session_state: st.session_state.raw_data = []

    with st.sidebar:
        st.header("⚙️ DỮ LIỆU")
        input_data = st.text_area("Dán kết quả Ku (5 số mỗi dòng):", height=300)
        if st.button("PHÂN TÍCH NGAY"):
            clean = re.findall(r'\b\d{5}\b', input_data)
            if clean:
                st.session_state.raw_data = clean
                st.rerun()

    if not st.session_state.raw_data:
        st.info("👈 Dán dữ liệu vào bên trái để bắt đầu.")
        return

    engine = PremiumLotoEngine(st.session_state.raw_data)
    analysis = engine.analyze_all()
    best = analysis[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div class="main-card"><h3>SỐ MẠNH NHẤT</h3><h1 style='color: #ff4b4b; font-size: 70px;'>{best['SỐ']}</h1><p>Độ tin cậy: {best['XÁC SUẤT (%)']}%</p></div>""", unsafe_allow_html=True)
        
        actual = st.text_input("Đối soát kỳ vừa về (5 số):")
        if st.button("LƯU LỊCH SỬ"):
            if len(actual) == 5:
                win = str(best['SỐ']) in actual
                st.session_state.history.insert(0, {"Giờ": datetime.datetime.now().strftime("%H:%M"), "Dự đoán": best['SỐ'], "Kết quả": actual, "Trạng thái": "✅ THẮNG" if win else "❌ THUA"})
                st.rerun()

    with col2:
        st.subheader("📊 CHI TIẾT 0-9")
        st.table(pd.DataFrame(analysis))

    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 NHẬT KÝ")
        st.table(pd.DataFrame(st.session_state.history))

if __name__ == "__main__":
    main()
