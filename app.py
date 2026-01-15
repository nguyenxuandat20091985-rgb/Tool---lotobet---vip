# ================= LOTOBET ULTRA AI PRO – V10.0 MINIMAL =================
# Phiên bản không cần matplotlib, plotly, chỉ dùng streamlit native

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET ULTRA AI PRO – V10.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS đơn giản
st.markdown("""
<style>
    .highlight-box {
        background-color: #FFA726;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FF9800;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .data-format {
        font-family: 'Courier New', monospace;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .blink {
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "lotobet_ultra_v10.db"

# ================= DATABASE =================
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS ky_quay (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE,
        so5 TEXT,
        tien_nhi TEXT,
        hau_nhi TEXT,
        tong INTEGER,
        tai_xiu TEXT,
        le_chan TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS phan_tich (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT,
        loai TEXT,
        gia_tri TEXT,
        diem_ai REAL,
        ty_le_truoc REAL,
        ket_qua_thuc TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS lich_su_danh (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT,
        loai_cuoc TEXT,
        so_danh TEXT,
        tien_cuoc REAL,
        ket_qua TEXT,
        tien_thang REAL,
        loi_nhuan REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS cai_dat (
        id INTEGER PRIMARY KEY,
        tong_von REAL DEFAULT 10000000,
        phan_tram_rui_ro REAL DEFAULT 5.0,
        ngay_bat_dau DATE DEFAULT CURRENT_DATE,
        chuoi_thua_toi_da INTEGER DEFAULT 7,
        phan_tram_lo_toi_da REAL DEFAULT 30.0
    )
    """)
    
    c.execute("INSERT OR IGNORE INTO cai_dat (id) VALUES (1)")
    
    conn.commit()
    conn.close()

init_db()

# ================= HELPER FUNCTIONS =================
def tai_xiu(tong):
    return "TÀI" if tong >= 23 else "XỈU"

def le_chan(tong):
    return "LẺ" if tong % 2 else "CHẴN"

def format_tien(tien):
    return f"{tien:,.0f} VNĐ"

def smart_parse_input(raw_text):
    if not raw_text:
        return []
    
    lines = raw_text.strip().split('\n')
    results = []
    
    for line in lines:
        line_clean = ''.join(c for c in line if c.isdigit() or c.isspace())
        numbers = line_clean.split()
        
        for num in numbers:
            if len(num) == 5 and num.isdigit():
                results.append(num)
            elif len(num) == 4 and num.isdigit():
                results.append(num)
    
    return results

def get_trend_icon(trend_type):
    icons = {
        "bệt_mạnh": "⏫",
        "bệt_yếu": "⏸️",
        "đảo_cầu": "🔀",
        "lặp_số": "🔁",
        "cầu_gãy": "❌",
        "cầu_sống": "✅",
        "đang_phân_tích": "🔄",
        "cảnh_báo": "⚠️",
        "xu_hướng_lên": "📈",
        "xu_hướng_xuống": "📉"
    }
    
    colors = {
        "bệt_mạnh": "#FF5252",
        "bệt_yếu": "#FF9800",
        "đảo_cầu": "#FFC107",
        "cầu_gãy": "#F44336",
        "cầu_sống": "#4CAF50",
        "cảnh_báo": "#FF9800"
    }
    
    icon = icons.get(trend_type, "📊")
    color = colors.get(trend_type, "#2196F3")
    
    return icon, color

# ================= AI ENGINE (Giữ nguyên) =================
class LottoAIAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        
    def analyze_2so_5tinh(self):
        if self.df.empty or len(self.df) < 10:
            return []
        
        nums = self.df["so5"].tolist()
        total_games = len(nums)
        pair_counter = defaultdict(int)
        
        for num in nums:
            unique_digits = set(num)
            for pair in itertools.combinations(unique_digits, 2):
                sorted_pair = ''.join(sorted(pair))
                pair_counter[sorted_pair] += 1
        
        results = []
        for pair, count in pair_counter.items():
            freq_score = (count / total_games) * 100
            last_seen = 0
            for i, num in enumerate(reversed(nums)):
                if all(digit in num for digit in pair):
                    last_seen = i
                    break
            
            delay_penalty = min(last_seen * 0.5, 20)
            ai_score = max(0, freq_score - delay_penalty)
            
            results.append({
                "cặp_số": pair,
                "tần_suất": count,
                "tỷ_lệ": round(freq_score, 2),
                "độ_trễ": last_seen,
                "điểm_AI": round(ai_score, 2),
                "xu_hướng": self._detect_pair_trend(pair)
            })
        
        return sorted(results, key=lambda x: x["điểm_AI"], reverse=True)[:10]
    
    def analyze_3so_5tinh(self):
        if self.df.empty or len(self.df) < 20:
            return []
        
        nums = self.df["so5"].tolist()
        total_games = len(nums)
        trio_counter = defaultdict(int)
        
        for num in nums:
            unique_digits = set(num)
            if len(unique_digits) >= 3:
                for trio in itertools.combinations(unique_digits, 3):
                    sorted_trio = ''.join(sorted(trio))
                    trio_counter[sorted_trio] += 1
        
        results = []
        for trio, count in trio_counter.items():
            freq_score = (count / total_games) * 100
            last_seen = 0
            for i, num in enumerate(reversed(nums)):
                if all(digit in num for digit in trio):
                    last_seen = i
                    break
            
            delay_penalty = min(last_seen * 0.3, 15)
            ai_score = max(0, freq_score - delay_penalty)
            
            results.append({
                "bộ_số": trio,
                "tần_suất": count,
                "tỷ_lệ": round(freq_score, 2),
                "độ_trễ": last_seen,
                "điểm_AI": round(ai_score, 2),
                "xu_hướng": self._detect_trio_trend(trio)
            })
        
        return sorted(results, key=lambda x: x["điểm_AI"], reverse=True)[:10]
    
    def _detect_pair_trend(self, pair):
        recent_games = self.df.head(20)["so5"].tolist()
        consecutive_count = 0
        for num in recent_games:
            if all(digit in num for digit in pair):
                consecutive_count += 1
            else:
                break
        
        if consecutive_count >= 3:
            return "bệt_mạnh" if consecutive_count >= 5 else "bệt_yếu"
        
        last_seen = 0
        for num in recent_games:
            if all(digit in num for digit in pair):
                last_seen = 0
            else:
                last_seen += 1
        
        if last_seen >= 5:
            return "cầu_gãy"
        
        pattern = []
        for num in recent_games[:10]:
            pattern.append(1 if all(digit in num for digit in pair) else 0)
        
        if pattern.count(1) >= 6 and pattern[-1] == 1 and pattern[-2] == 0:
            return "đảo_cầu"
        
        return "cầu_sống"
    
    def _detect_trio_trend(self, trio):
        recent_games = self.df.head(30)["so5"].tolist()
        appearances = [1 if all(digit in num for digit in trio) else 0 for num in recent_games]
        
        if sum(appearances[-3:]) == 3:
            return "bệt_mạnh"
        elif sum(appearances[-5:]) >= 4:
            return "bệt_yếu"
        elif sum(appearances) == 0:
            return "cầu_gãy"
        
        return "cầu_sống"
    
    def analyze_tai_xiu(self):
        if self.df.empty:
            return {"prediction": "TÀI", "confidence": 50}
        
        recent_tx = self.df.head(20)["tai_xiu"].tolist()
        tai_count = recent_tx.count("TÀI")
        xiu_count = recent_tx.count("XỈU")
        
        if tai_count > xiu_count:
            return {"prediction": "TÀI", "confidence": round(tai_count/20*100, 1)}
        else:
            return {"prediction": "XỈU", "confidence": round(xiu_count/20*100, 1)}
    
    def analyze_le_chan(self):
        if self.df.empty:
            return {"prediction": "LẺ", "confidence": 50}
        
        recent_lc = self.df.head(20)["le_chan"].tolist()
        le_count = recent_lc.count("LẺ")
        chan_count = recent_lc.count("CHẴN")
        
        if le_count > chan_count:
            return {"prediction": "LẺ", "confidence": round(le_count/20*100, 1)}
        else:
            return {"prediction": "CHẴN", "confidence": round(chan_count/20*100, 1)}

# ================= DATA MANAGEMENT =================
def save_ky_quay(numbers):
    conn = get_conn()
    c = conn.cursor()
    added_count = 0
    
    for num in numbers:
        if len(num) != 5 or not num.isdigit():
            continue
            
        ky_id = f"KY{int(time.time() * 1000) % 1000000:06d}"
        so5 = num
        tien_nhi = num[:2]
        hau_nhi = num[-2:]
        tong = sum(int(d) for d in num)
        
        try:
            c.execute("""
            INSERT OR IGNORE INTO ky_quay 
            (ky, so5, tien_nhi, hau_nhi, tong, tai_xiu, le_chan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ky_id, so5, tien_nhi, hau_nhi, tong,
                tai_xiu(tong), le_chan(tong)
            ))
            
            if c.rowcount > 0:
                added_count += 1
        except:
            pass
    
    conn.commit()
    conn.close()
    return added_count

def load_recent_data(limit=1000):
    conn = get_conn()
    query = f"""
    SELECT * FROM ky_quay 
    ORDER BY timestamp DESC 
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================= MAIN APP =================
def main():
    st.title("🎰 LOTOBET ULTRA AI PRO – V10.0")
    st.markdown("---")
    
    # Tabs
    tabs = st.tabs([
        "📊 DASHBOARD",
        "🎯 PHÂN TÍCH",
        "💰 QUẢN LÝ VỐN",
        "📥 NHẬP DỮ LIỆU",
        "⚙️ CÀI ĐẶT"
    ])
    
    # Load data
    df = load_recent_data(500)
    
    # ================= TAB 1: DASHBOARD =================
    with tabs[0]:
        st.subheader("📊 DASHBOARD TỔNG QUAN - 4 KHUNG RIÊNG BIỆT")
        
        # KHUNG A: TỔNG KỲ
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📌 TỔNG KỲ TRONG DB", len(df))
            st.caption(f"Cập nhật: {datetime.now().strftime('%H:%M:%S')}")
            
            # Hiển thị xu hướng đơn giản không dùng biểu đồ
            if len(df) > 5:
                recent_totals = df.head(10)["tong"].tolist()
                avg_tong = np.mean(recent_totals)
                st.caption(f"Trung bình 10 kỳ gần nhất: {avg_tong:.1f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # KHUNG B: 2 SỐ 5 TÍNH
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🔥 2 SỐ 5 TÍNH")
            
            if not df.empty:
                analyzer = LottoAIAnalyzer(df)
                results_2so = analyzer.analyze_2so_5tinh()
                
                if results_2so:
                    best_2so = results_2so[0]
                    icon, color = get_trend_icon(best_2so.get("xu_hướng", "cầu_sống"))
                    
                    # Hiển thị icon
                    st.markdown(f'<span style="font-size:24px;color:{color}">{icon}</span> **{best_2so["cặp_số"]}**', unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Điểm AI", f"{best_2so['điểm_AI']}%")
                    with cols[1]:
                        st.metric("Tần suất", best_2so['tần_suất'])
                    
                    # Progress bar
                    st.progress(min(best_2so['điểm_AI']/100, 1.0))
                    
                    # Format data
                    st.markdown('<div class="data-format">', unsafe_allow_html=True)
                    st.text(f"( 2 tinh: {best_2so['cặp_số'][0]}{best_2so['cặp_số'][1]}• )")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⏳ Đang phân tích...")
            else:
                st.info("📥 Vui lòng nhập dữ liệu")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # KHUNG C: 3 SỐ 5 TÍNH
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🔥 3 SỐ 5 TÍNH")
            
            if not df.empty:
                analyzer = LottoAIAnalyzer(df)
                results_3so = analyzer.analyze_3so_5tinh()
                
                if results_3so:
                    best_3so = results_3so[0]
                    icon, color = get_trend_icon(best_3so.get("xu_hướng", "cầu_sống"))
                    
                    st.markdown(f'<span style="font-size:24px;color:{color}">{icon}</span> **{best_3so["bộ_số"]}**', unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Điểm AI", f"{best_3so['điểm_AI']}%")
                    with cols[1]:
                        st.metric("Tần suất", best_3so['tần_suất'])
                    
                    st.progress(min(best_3so['điểm_AI']/100, 1.0))
                    
                    st.markdown('<div class="data-format">', unsafe_allow_html=True)
                    st.text(f"( 3 tinh: {best_3so['bộ_số']}• )")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⏳ Đang phân tích...")
            else:
                st.info("📥 Vui lòng nhập dữ liệu")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # KHUNG D: KẾT LUẬN SỐ ĐÁNH
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("## 🎯 SỐ CẦN ĐÁNH KỲ TIẾP THEO")
        
        if not df.empty:
            analyzer = LottoAIAnalyzer(df)
            results_2so = analyzer.analyze_2so_5tinh()
            results_3so = analyzer.analyze_3so_5tinh()
            
            if results_2so and results_3so:
                best_2so = results_2so[0]
                best_3so = results_3so[0]
                tx_analysis = analyzer.analyze_tai_xiu()
                lc_analysis = analyzer.analyze_le_chan()
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.markdown("### 🔥 2 SỐ")
                    st.markdown(f"# `{best_2so['cặp_số']}`")
                    st.caption(f"Điểm AI: {best_2so['điểm_AI']}%")
                
                with col_b:
                    st.markdown("### 🔥 3 SỐ")
                    st.markdown(f"# `{best_3so['bộ_số']}`")
                    st.caption(f"Điểm AI: {best_3so['điểm_AI']}%")
                
                with col_c:
                    st.markdown("### 🎲 TÀI/XỈU")
                    st.markdown(f"# `{tx_analysis['prediction']}`")
                    st.caption(f"Độ tin cậy: {tx_analysis['confidence']}%")
                
                with col_d:
                    st.markdown("### 🎲 LẺ/CHẴN")
                    st.markdown(f"# `{lc_analysis['prediction']}`")
                    st.caption(f"Độ tin cậy: {lc_analysis['confidence']}%")
                
                st.markdown("---")
                st.caption("✅ Dựa trên phân tích AI từ dữ liệu lịch sử")
            else:
                st.info("🔄 Đang phân tích dữ liệu...")
        else:
            st.info("📥 Vui lòng nhập dữ liệu trước")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ================= TAB 2: PHÂN TÍCH =================
    with tabs[1]:
        st.subheader("🎯 PHÂN TÍCH CHI TIẾT")
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 TOP 5 CẶP 2 SỐ")
                analyzer = LottoAIAnalyzer(df)
                results_2so = analyzer.analyze_2so_5tinh()[:5]
                
                if results_2so:
                    for i, result in enumerate(results_2so, 1):
                        icon, color = get_trend_icon(result["xu_hướng"])
                        cols = st.columns([1, 3, 2])
                        with cols[0]:
                            st.markdown(f"**{i}.**")
                        with cols[1]:
                            st.markdown(f'<span style="color:{color};font-size:20px">{icon}</span> **`{result["cặp_số"]}`**', unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f"{result['điểm_AI']}%")
                
                st.markdown("---")
                st.markdown("#### 🎯 TIỀN NHỊ")
                if len(df) > 0:
                    recent_tn = df.head(10)[["ky", "tien_nhi"]]
                    st.dataframe(recent_tn, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 TOP 5 BỘ 3 SỐ")
                results_3so = analyzer.analyze_3so_5tinh()[:5]
                
                if results_3so:
                    for i, result in enumerate(results_3so, 1):
                        icon, color = get_trend_icon(result["xu_hướng"])
                        cols = st.columns([1, 3, 2])
                        with cols[0]:
                            st.markdown(f"**{i}.**")
                        with cols[1]:
                            st.markdown(f'<span style="color:{color};font-size:20px">{icon}</span> **`{result["bộ_số"]}`**', unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f"{result['điểm_AI']}%")
                
                st.markdown("---")
                st.markdown("#### 🎯 HẬU NHỊ")
                if len(df) > 0:
                    recent_hn = df.head(10)[["ky", "hau_nhi"]]
                    st.dataframe(recent_hn, use_container_width=True)
        else:
            st.info("📥 Vui lòng nhập dữ liệu trước")
    
    # ================= TAB 3: QUẢN LÝ VỐN =================
    with tabs[2]:
        st.subheader("💰 QUẢN LÝ VỐN THÔNG MINH")
        
        # Load settings
        conn = get_conn()
        settings = pd.read_sql("SELECT * FROM cai_dat WHERE id = 1", conn)
        conn.close()
        
        if not settings.empty:
            current = settings.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                tong_von = st.number_input(
                    "💰 TỔNG VỐN (VNĐ)",
                    min_value=100000,
                    value=int(current["tong_von"]),
                    step=100000
                )
            
            with col2:
                rui_ro = st.slider(
                    "📉 RỦI RO / KỲ (%)",
                    min_value=1,
                    max_value=20,
                    value=int(current["phan_tram_rui_ro"])
                )
            
            if st.button("🎯 TÍNH PHÂN BỔ", type="primary"):
                if not df.empty:
                    analyzer = LottoAIAnalyzer(df)
                    results_2so = analyzer.analyze_2so_5tinh()
                    results_3so = analyzer.analyze_3so_5tinh()
                    
                    if results_2so and results_3so:
                        best_2so = results_2so[0]
                        best_3so = results_3so[0]
                        
                        tien_toi_da = tong_von * (rui_ro / 100)
                        diem_tong = best_2so['điểm_AI'] + best_3so['điểm_AI']
                        
                        if diem_tong > 0:
                            tien_2so = (best_2so['điểm_AI'] / diem_tong) * tien_toi_da
                            tien_3so = (best_3so['điểm_AI'] / diem_tong) * tien_toi_da
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("### 📊 PHÂN BỔ VỐN")
                            
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("TỔNG CƯỢC", format_tien(tien_toi_da))
                            
                            with col_b:
                                st.metric("2 SỐ", format_tien(tien_2so))
                            
                            with col_c:
                                st.metric("3 SỐ", format_tien(tien_3so))
                            
                            st.markdown("---")
                            st.markdown(f"**2 Số `{best_2so['cặp_số']}`:** {format_tien(tien_2so)} ({best_2so['điểm_AI']:.1f}%)")
                            st.markdown(f"**3 Số `{best_3so['bộ_số']}`:** {format_tien(tien_3so)} ({best_3so['điểm_AI']:.1f}%)")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Lưu cài đặt
                            if st.button("💾 LƯU CÀI ĐẶT"):
                                conn = get_conn()
                                c = conn.cursor()
                                c.execute("""
                                UPDATE cai_dat 
                                SET tong_von = ?, phan_tram_rui_ro = ?
                                WHERE id = 1
                                """, (tong_von, rui_ro))
                                conn.commit()
                                conn.close()
                                st.success("✅ Đã lưu!")
                    else:
                        st.warning("Chưa đủ dữ liệu phân tích")
                else:
                    st.info("Vui lòng nhập dữ liệu trước")
    
    # ================= TAB 4: NHẬP DỮ LIỆU =================
    with tabs[3]:
        st.subheader("📥 NHẬP DỮ LIỆU")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            raw = st.text_area(
                "Dán kết quả (mỗi dòng 1 số hoặc nhiều số cách nhau):",
                height=200,
                placeholder="""12345
67890
54321

Hoặc: 12345 67890 54321

Hoặc: 2 tinh: 5264 3 tinh: 5289
"""
            )
        
        with col2:
            st.markdown("#### 📁 TỪ FILE")
            uploaded = st.file_uploader("TXT/CSV", type=['txt', 'csv'])
            
            if uploaded:
                content = uploaded.getvalue().decode()
                st.text_area("Nội dung:", content, height=150, disabled=True)
                
                if st.button("📥 NHẬP FILE"):
                    nums = smart_parse_input(content)
                    added = save_ky_quay(nums)
                    st.success(f"✅ Đã thêm {added} kỳ")
        
        if raw:
            nums = smart_parse_input(raw)
            
            if nums:
                st.markdown(f"**Tìm thấy {len(nums)} số:**")
                st.markdown('<div class="data-format">', unsafe_allow_html=True)
                for num in nums[:10]:
                    st.text(f"• {num}")
                if len(nums) > 10:
                    st.text(f"... và {len(nums)-10} số khác")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("💾 LƯU VÀO DB", type="primary"):
                    added = save_ky_quay(nums)
                    st.success(f"✅ Đã lưu {added} kỳ mới!")
                    time.sleep(1)
                    st.rerun()
        
        # Hiển thị dữ liệu hiện có
        st.markdown("---")
        st.subheader("📊 DỮ LIỆU HIỆN CÓ")
        
        if not df.empty:
            st.dataframe(
                df[["ky", "so5", "tai_xiu", "le_chan"]].head(20),
                use_container_width=True,
                height=300
            )
            st.caption(f"Hiển thị 20/{len(df)} kỳ gần nhất")
        else:
            st.info("📭 Chưa có dữ liệu")
    
    # ================= TAB 5: CÀI ĐẶT =================
    with tabs[4]:
        st.subheader("⚙️ CÀI ĐẶT HỆ THỐNG")
        
        conn = get_conn()
        settings = pd.read_sql("SELECT * FROM cai_dat WHERE id = 1", conn)
        conn.close()
        
        if not settings.empty:
            s = settings.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                chuoi_thua = st.number_input(
                    "Chuỗi thua cảnh báo:",
                    min_value=1,
                    max_value=20,
                    value=int(s["chuoi_thua_toi_da"])
                )
            
            with col2:
                lo_toi_da = st.slider(
                    "Lỗ tối đa/ngày (%):",
                    min_value=10,
                    max_value=50,
                    value=int(s["phan_tram_lo_toi_da"])
                )
            
            if st.button("💾 LƯU CÀI ĐẶT", type="primary"):
                conn = get_conn()
                c = conn.cursor()
                c.execute("""
                UPDATE cai_dat 
                SET chuoi_thua_toi_da = ?, phan_tram_lo_toi_da = ?
                WHERE id = 1
                """, (chuoi_thua, lo_toi_da))
                conn.commit()
                conn.close()
                st.success("✅ Đã lưu cài đặt!")
            
            st.markdown("---")
            st.markdown("#### ⚠️ CẢNH BÁO AN TOÀN")
            st.markdown("""
            - **Dừng ngay** khi thua 5 kỳ liên tiếp
            - **Không đánh** quá 5% vốn/kỳ
            - **Nghỉ ngơi** khi lỗ 20% trong ngày
            - **Tool chỉ hỗ trợ**, quyết định cuối cùng là của bạn
            """)
            
            st.markdown('<div style="background-color:#E3F2FD;padding:20px;border-radius:10px">', unsafe_allow_html=True)
            st.markdown("### 🧠 **KỶ LUẬT LÀ CHÌA KHÓA - DỪNG LẠI ĐÚNG LÚC**")
            st.markdown('</div>', unsafe_allow_html=True)

# ================= RUN =================
if __name__ == "__main__":
    main()
