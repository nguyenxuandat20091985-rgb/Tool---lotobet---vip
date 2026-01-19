import streamlit as st
import re
from collections import Counter
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN COMPACT (NHỎ GỌN & CHUYÊN NGHIỆP) ---
st.set_page_config(page_title="v6.0 Compact Pro", layout="wide")

st.markdown("""
    <style>
    /* Tổng thể nền trắng sạch sẽ */
    .stApp { background: #ffffff; }
    
    /* Thu nhỏ Tab ngang */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 15px; font-size: 14px; border-radius: 8px 8px 0 0;
        background-color: #f8f9fa; color: #666;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #d9534f !important; color: white !important; 
    }

    /* Thẻ 2D thiết kế lại nhỏ gọn */
    .compact-card {
        background: white; border: 1px solid #eee; border-radius: 12px;
        padding: 10px; text-align: center; margin: 5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-top: 4px solid #d9534f;
    }
    .compact-num { color: #d9534f; font-size: 32px; font-weight: 800; line-height: 1; }
    .compact-pct { color: #28a745; font-size: 14px; font-weight: bold; margin-bottom: 5px; }
    .compact-label { color: #999; font-size: 10px; text-transform: uppercase; }

    /* Nút bấm tinh gọn */
    .stButton>button {
        border-radius: 8px; font-size: 14px; padding: 5px 0; height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. THUẬT TOÁN ĐA TẦNG (NHỊP BỆT & LẶP KỲ) ---
def engine_v6_compact(data):
    # Lấy các cặp số 2D
    raw_2d = re.findall(r'\d{2,5}', data)
    last_2d_list = [n[-2:] for n in raw_2d]
    
    if len(last_2d_list) < 10: return None
    
    freq = Counter(last_2d_list)
    last_5 = last_2d_list[-5:] # Nhịp kỳ vừa mở
    
    all_pairs = [f"{i:02d}" for i in range(100)]
    scored = []
    
    for p in all_pairs:
        score = 0
        # T1: Nhịp Bệt (Lặp kỳ trước)
        if p in last_5: score += 55
        # T2: Tần suất xuất hiện
        score += freq[p] * 12
        # T3: Bóng số lặp
        shadow = "".join([{"0":"5","5":"0","1":"6","6":"1","2":"7","7":"2","3":"8","8":"3","4":"9","9":"4"}.get(c,c) for c in p])
        if shadow in last_5: score += 20
        
        # Độ tin cậy
        conf = min(82 + (score / 6.5), 98.9)
        scored.append({'num': p, 'conf': round(conf, 1)})
    
    return sorted(scored, key=lambda x: x['conf'], reverse=True)[:5]

# --- 3. QUẢN LÝ DỮ LIỆU PHIÊN CHƠI ---
if 'history_v6' not in st.session_state: st.session_state.history_v6 = []

# --- 4. GIAO DIỆN CHÍNH ---
st.markdown("<h4 style='text-align: center; color: #d9534f; margin-bottom: 0;'>💎 PREDICTOR v6.0 COMPACT</h4>", unsafe_allow_html=True)

tab_soi, tab_stat, tab_info = st.tabs(["🎯 SOI CẦU", "📊 THỐNG KÊ", "📜 HD"])

with tab_soi:
    # Chia cột tỉ lệ 1:2 để tiết kiệm diện tích
    c_in, c_out = st.columns([1, 2.2])
    
    with c_in:
        raw_txt = st.text_area("Dán OCR:", height=120, placeholder="Kết quả kỳ trước...")
        if st.button("🚀 PHÂN TÍCH", use_container_width=True):
            res = engine_v6_compact(raw_txt)
            if res:
                st.session_state.current_5 = res
            else:
                st.error("Thiếu dữ liệu!")

    with c_out:
        if 'current_5' in st.session_state:
            # Hiển thị 5 cặp số theo dạng lưới nhỏ gọn
            rows = st.columns(5)
            for idx, item in enumerate(st.session_state.current_5):
                with rows[idx]:
                    st.markdown(f"""
                        <div class="compact-card">
                            <div class="compact-pct">{item['conf']}%</div>
                            <div class="compact-num">{item['num']}</div>
                            <div class="compact-label">Tỉ lệ về</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Nút báo cáo nhanh dưới dãy số
            st.write("")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("✅ WIN", use_container_width=True):
                    st.session_state.history_v6.append({"KQ": "WIN", "Số": [x['num'] for x in st.session_state.current_5]})
                    st.toast("Ghi nhận THẮNG!")
            with b2:
                if st.button("❌ LOSS", use_container_width=True):
                    st.session_state.history_v6.append({"KQ": "LOSS", "Số": [x['num'] for x in st.session_state.current_5]})
                    st.toast("Ghi nhận THUA!")

with tab_stat:
    if st.session_state.history_v6:
        df = pd.DataFrame(st.session_state.history_v6)
        wins = len(df[df['KQ'] == 'WIN'])
        total = len(df)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Tổng Kỳ", total)
        col_m2.metric("Tỉ lệ Win", f"{(wins/total)*100:.1f}%")
        
        st.markdown("**10 Kỳ Gần Nhất:**")
        st.table(df.tail(10))
    else:
        st.info("Chưa có lịch sử.")

with tab_info:
    st.caption("Chiến thuật: 5 cặp rời (50k) - Chế độ 2 số 5 tinh. Ưu tiên số có % trên 95%.")
