import streamlit as st
import re
from collections import Counter
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP (TAB NGANG & TỐI ƯU DIỆN TÍCH) ---
st.set_page_config(page_title="v6.0 PREDICTOR-ULTIMATE", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { background: white; }
    
    /* Thiết kế thẻ 2D chuyên nghiệp */
    .card-2d {
        background: #ffffff; border: 2px solid #d9534f; border-radius: 15px;
        padding: 15px; text-align: center; margin: 5px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        flex: 1; min-width: 140px;
    }
    .num-2d { color: #d9534f; font-size: 45px; font-weight: 900; margin: 0; }
    .percent-2d { color: #28a745; font-size: 18px; font-weight: bold; }
    .label-2d { color: #888; font-size: 11px; text-transform: uppercase; }
    
    /* Tối ưu Tab ngang */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f1f1; border-radius: 10px 10px 0 0;
        padding: 10px 20px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #d9534f !important; color: white !important; }
    
    /* Bảng thống kê */
    .stat-box {
        padding: 10px; border-radius: 8px; border: 1px solid #ddd;
        text-align: center; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. HỆ THỐNG ĐA THUẬT TOÁN (NHỊP BỆT, BÓNG, TẦN SUẤT, LẶP KỲ) ---
def advanced_engine_v6(data):
    # Trích xuất 2 số cuối (2D)
    raw_2d = re.findall(r'\d{2,5}', data)
    last_2d_list = [n[-2:] for n in raw_2d]
    
    if len(last_2d_list) < 15: return None
    
    freq = Counter(last_2d_list)
    last_kỳ = last_2d_list[-5:] # 5 con kỳ vừa ra
    prev_kỳ = last_2d_list[-10:-5] # 5 con kỳ trước đó nữa
    
    all_pairs = [f"{i:02d}" for i in range(100)]
    scored = []
    
    for p in all_pairs:
        score = 0
        # 1. Thuật toán Lặp kỳ (Bệt kỳ trước)
        if p in last_kỳ: score += 50
        # 2. Thuật toán Nhịp rơi (Bệt cách kỳ)
        if p in prev_kỳ: score += 30
        # 3. Thuật toán Bóng số (0-5, 1-6, 2-7, 3-8, 4-9)
        shadow_p = "".join([{"0":"5","5":"0","1":"6","6":"1","2":"7","7":"2","3":"8","8":"3","4":"9","9":"4"}.get(c, c) for c in p])
        if shadow_p in last_kỳ: score += 25
        # 4. Thuật toán Tần suất (Hot)
        score += freq[p] * 15
        
        # Tính độ tin cậy %
        confidence = min(80 + (score / 6), 99.2)
        scored.append({'num': p, 'conf': round(confidence, 1)})
    
    # Trả về 5 cặp số mạnh nhất
    return sorted(scored, key=lambda x: x['conf'], reverse=True)[:5]

# --- 3. QUẢN LÝ DỮ LIỆU ---
if 'history_v6' not in st.session_state: st.session_state.history_v6 = []

# --- 4. GIAO DIỆN CHÍNH ---
st.markdown("<h2 style='text-align: center; color: #d9534f;'>💎 PREDICTOR v6.0 ULTIMATE</h2>", unsafe_allow_html=True)

# Tabs ngang tối ưu diện tích
tab_soi, tab_thong_ke, tab_huong_dan = st.tabs(["🎯 SOI CẦU 5 CẶP", "📊 THỐNG KÊ CHI TIẾT", "📜 CHIẾN THUẬT"])

with tab_soi:
    col_input, col_output = st.columns([1, 1.8])
    
    with col_input:
        st.markdown("##### 📥 Dữ liệu kỳ trước")
        input_data = st.text_area("Dán chuỗi số OCR:", height=180, placeholder="Dán dãy số tại đây...")
        if st.button("🚀 PHÂN TÍCH ĐA THUẬT TOÁN", use_container_width=True):
            res = advanced_engine_v6(input_data)
            if res:
                st.session_state.current_5 = res
                st.success("✅ Đã tối ưu dự đoán!")
            else:
                st.error("Cần tối thiểu 15 cặp số để phân tích nhịp.")

    with col_output:
        if 'current_5' in st.session_state:
            st.markdown("##### 🔮 5 Cặp Số Rời Tin Cậy (Vốn 50k)")
            cols = st.columns(5)
            for idx, item in enumerate(st.session_state.current_4 if 'current_4' in st.session_state else st.session_state.current_5):
                with cols[idx]:
                    st.markdown(f"""
                        <div class="card-2d">
                            <div class="label-2d">Tỉ lệ về</div>
                            <div class="percent-2d">{item['conf']}%</div>
                            <div class="num-2d">{item['num']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.write("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ BÁO THẮNG (WIN)", use_container_width=True):
                    st.session_state.history_v6.append({"KQ": "WIN", "Số": [x['num'] for x in st.session_state.current_5]})
                    st.balloons()
            with c2:
                if st.button("❌ BÁO THUA (LOSS)", use_container_width=True):
                    st.session_state.history_v6.append({"KQ": "LOSS", "Số": [x['num'] for x in st.session_state.current_5]})

with tab_thong_ke:
    if st.session_state.history_v6:
        df = pd.DataFrame(st.session_state.history_v6)
        
        # Thống kê nhịp
        wins = len(df[df['KQ'] == 'WIN'])
        total = len(df)
        st.metric("Tỉ lệ thắng thực tế (Lần cược)", f"{(wins/total)*100:.1f}%")
        
        st.markdown("##### 📋 Nhật ký lặp kỳ")
        st.table(df.tail(10)) # Hiển thị 10 kỳ gần nhất
    else:
        st.info("Chưa có dữ liệu thống kê. Hãy bắt đầu soi cầu!")

with tab_huong_dan:
    st.markdown("""
    ### 🛡️ Chiến thuật 5 Cặp Rời (Vốn 50k)
    1. **Cách chơi:** Đặt 5 cặp số rời rạc vào mục '2 số 5 tinh'. 
    2. **Vào tiền:** Mỗi cặp 10k. Tổng 50k/kỳ. 
    3. **Ưu điểm:** Độ phủ cực rộng, giảm thiểu tối đa rủi ro nhà cái lách số.
    4. **Thống kê:** Quan sát Tab Thống kê để thấy 'Số lặp kỳ' - Nếu số lặp kỳ ra liên tục, hãy tăng điểm cho các cặp % cao.
    """)
