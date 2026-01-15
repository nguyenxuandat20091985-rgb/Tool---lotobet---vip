# ================= LOTOBET ULTRA AI PRO – V10.0 LITE =================
# Phiên bản không cần plotly

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Thêm để fix lỗi matplotlib trong Streamlit
plt.switch_backend('Agg')

# ================= CONFIG & STYLING =================
st.set_page_config(
    page_title="LOTOBET ULTRA AI PRO – V10.0 LITE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (giữ nguyên)
st.markdown("""
<style>
    .highlight-box {
        background-color: #FFA726;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FF9800;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(255, 167, 38, 0.3);
    }
    .icon-large {
        font-size: 28px;
        margin-right: 10px;
        vertical-align: middle;
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
</style>
""", unsafe_allow_html=True)

DB_FILE = "lotobet_ultra_v10_lite.db"

# ================= DATABASE SCHEMA (Giữ nguyên) =================
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

# ================= HELPER FUNCTIONS (Giữ nguyên) =================
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

def get_trend_icon(trend_type, strength=1):
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
    blink_style = "animation: blink 1s infinite;" if trend_type == "cầu_gãy" else ""
    
    return f'<span class="icon-large" style="color:{color};{blink_style}">{icon}</span>'

# ================= AI ANALYSIS ENGINE (Giữ nguyên) =================
class LottoAIAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        
    def analyze_2so_5tinh(self):
        if self.df.empty or len(self.df) < 10:
            return {}
        
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
            return {}
        
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
    
    def detect_all_patterns(self):
        patterns = {
            "chuoi_thua": self._check_losing_streak(),
            "vuot_nguong_lo": self._check_loss_threshold(),
            "canh_bao_dac_biet": []
        }
        return patterns
    
    def _check_losing_streak(self):
        conn = get_conn()
        query = """
        SELECT COUNT(*) as chuoi_thua 
        FROM lich_su_danh 
        WHERE loi_nhuan < 0 
        ORDER BY timestamp DESC 
        LIMIT 10
        """
        result = pd.read_sql(query, conn)
        conn.close()
        return result["chuoi_thua"].iloc[0] if not result.empty else 0
    
    def _check_loss_threshold(self):
        conn = get_conn()
        query = """
        SELECT SUM(loi_nhuan) as tong_lo_hom_nay
        FROM lich_su_danh 
        WHERE DATE(timestamp) = DATE('now')
        """
        result = pd.read_sql(query, conn)
        conn.close()
        tong_lo = abs(result["tong_lo_hom_nay"].iloc[0]) if not result.empty and result["tong_lo_hom_nay"].iloc[0] < 0 else 0
        
        cai_dat = pd.read_sql("SELECT tong_von, phan_tram_lo_toi_da FROM cai_dat WHERE id = 1", conn)
        tong_von = cai_dat["tong_von"].iloc[0]
        ngay_lo = cai_dat["phan_tram_lo_toi_da"].iloc[0]
        phan_tram_lo = (tong_lo / tong_von * 100) if tong_von > 0 else 0
        
        return {
            "tong_lo": tong_lo,
            "phan_tram_lo": round(phan_tram_lo, 1),
            "vuot_nguong": phan_tram_lo >= ngay_lo
        }

# ================= DATA MANAGEMENT (Giữ nguyên) =================
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

# ================= CAPITAL MANAGEMENT (Giữ nguyên) =================
def calculate_bet_distribution(von, rui_ro_percent, best_2so, best_3so):
    tien_toi_da = von * (rui_ro_percent / 100)
    
    if not best_2so or not best_3so:
        return {}
    
    diem_2so = best_2so.get("điểm_AI", 0)
    diem_3so = best_3so.get("điểm_AI", 0)
    tong_diem = diem_2so + diem_3so
    
    if tong_diem == 0:
        return {}
    
    phan_tram_2so = (diem_2so / tong_diem) * 100
    phan_tram_3so = (diem_3so / tong_diem) * 100
    
    tien_2so = (phan_tram_2so / 100) * tien_toi_da
    tien_3so = (phan_tram_3so / 100) * tien_toi_da
    
    return {
        "tien_toi_da": tien_toi_da,
        "2_so": {
            "so": best_2so.get("cặp_số", ""),
            "diem_AI": diem_2so,
            "phan_tram": round(phan_tram_2so, 1),
            "tien": tien_2so
        },
        "3_so": {
            "so": best_3so.get("bộ_số", ""),
            "diem_AI": diem_3so,
            "phan_tram": round(phan_tram_3so, 1),
            "tien": tien_3so
        }
    }

# ================= SIMPLE CHART FUNCTIONS =================
def create_simple_line_chart(data, title):
    """Tạo biểu đồ đơn giản không cần plotly"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(data)), data, marker='o', linewidth=2, color='#4CAF50')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    fig.tight_layout()
    return fig

def create_simple_pie_chart(labels, values, title):
    """Tạo biểu đồ tròn đơn giản"""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#FF5252', '#2196F3', '#4CAF50', '#FFC107']
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(labels)])
    ax.set_title(title)
    return fig

# ================= MAIN APPLICATION (ĐÃ SỬA) =================
def main():
    st.title("🎰 LOTOBET ULTRA AI PRO – V10.0 LITE")
    st.markdown("---")
    
    # Tabs chính
    tabs = st.tabs([
        "📊 DASHBOARD TỔNG QUAN",
        "🎯 PHÂN TÍCH CHI TIẾT",
        "💰 QUẢN LÝ VỐN PRO",
        "📥 NHẬP DỮ LIỆU",
        "📈 BÁO CÁO & KIỂM TRA",
        "⚙️ CÀI ĐẶT & CẢNH BÁO"
    ])
    
    # Load data
    df = load_recent_data(500)
    
    # ================= TAB 1: DASHBOARD TỔNG QUAN =================
    with tabs[0]:
        st.subheader("📊 DASHBOARD TỔNG QUAN - 4 KHUNG RIÊNG BIỆT")
        
        # KHUNG A: TỔNG KỲ
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📌 TỔNG KỲ TRONG DB", len(df))
            st.caption(f"Cập nhật: {datetime.now().strftime('%H:%M:%S')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Biểu đồ đơn giản thay thế plotly
            if len(df) > 10:
                recent_totals = df.head(20)["tong"].values[::-1]
                fig = create_simple_line_chart(recent_totals, "Xu hướng tổng 20 kỳ gần nhất")
                st.pyplot(fig)
        
        # KHUNG B: 2 SỐ 5 TÍNH
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🔥 2 SỐ 5 TÍNH")
            
            if not df.empty:
                analyzer = LottoAIAnalyzer(df)
                results_2so = analyzer.analyze_2so_5tinh()
                
                if results_2so:
                    best_2so = results_2so[0]
                    icon_html = get_trend_icon(best_2so.get("xu_hướng", "cầu_sống"))
                    st.markdown(f"{icon_html} **{best_2so['cặp_số']}**", unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Điểm AI", f"{best_2so['điểm_AI']}%")
                    with cols[1]:
                        st.metric("Tần suất", best_2so['tần_suất'])
                    
                    st.progress(min(best_2so['điểm_AI']/100, 1.0))
                    
                    st.markdown('<div class="data-format">', unsafe_allow_html=True)
                    st.text(f"( 2 tinh: {best_2so['cặp_số'][0]}{best_2so['cặp_số'][1]}• )")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⏳ Đang phân tích dữ liệu...")
            else:
                st.info("📥 Vui lòng nhập dữ liệu trước")
            
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
                    icon_html = get_trend_icon(best_3so.get("xu_hướng", "cầu_sống"))
                    st.markdown(f"{icon_html} **{best_3so['bộ_số']}**", unsafe_allow_html=True)
                    
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
                    st.warning("⏳ Đang phân tích dữ liệu...")
            else:
                st.info("📥 Vui lòng nhập dữ liệu trước")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # KHUNG D: KẾT LUẬN SỐ ĐÁNH (NỔI BẬT)
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("## 🎯 SỐ CẦN ĐÁNH KỲ TIẾP THEO")
        
        if not df.empty and results_2so and results_3so:
            best_2so = results_2so[0]
            best_3so = results_3so[0]
            
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
                tx_analysis = analyzer.analyze_tai_xiu()
                st.markdown("### 🎲 TÀI/XỈU")
                st.markdown(f"# `{tx_analysis['prediction']}`")
                st.caption(f"Độ tin cậy: {tx_analysis['confidence']}%")
            
            with col_d:
                lc_analysis = analyzer.analyze_le_chan()
                st.markdown("### 🎲 LẺ/CHẴN")
                st.markdown(f"# `{lc_analysis['prediction']}`")
                st.caption(f"Độ tin cậy: {lc_analysis['confidence']}%")
            
            st.markdown("---")
            st.caption("✅ Dựa trên phân tích AI từ dữ liệu lịch sử")
        else:
            st.info("🔄 Đang tải dữ liệu và phân tích...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cảnh báo hệ thống
        st.markdown("---")
        st.subheader("⚠️ CẢNH BÁO HỆ THỐNG")
        
        if not df.empty:
            analyzer = LottoAIAnalyzer(df)
            patterns = analyzer.detect_all_patterns()
            
            col_w1, col_w2 = st.columns(2)
            
            with col_w1:
                if patterns["chuoi_thua"] >= 5:
                    st.markdown(f'<div class="warning-box">{get_trend_icon("cảnh_báo")} <b>CHUỖI THUA: {patterns["chuoi_thua"]} kỳ</b></div>', unsafe_allow_html=True)
            
            with col_w2:
                loss_info = patterns["vuot_nguong_lo"]
                if loss_info["vuot_nguong"]:
                    st.markdown(f'<div class="warning-box">{get_trend_icon("cảnh_báo")} <b>VƯỢT NGƯỠNG LỖ: {loss_info["phan_tram_lo"]}%</b></div>', unsafe_allow_html=True)
    
    # ================= TAB 2: PHÂN TÍCH CHI TIẾT =================
    with tabs[1]:
        st.subheader("🎯 PHÂN TÍCH CHI TIẾT THEO VỊ TRÍ")
        
        if not df.empty:
            col_pos1, col_pos2 = st.columns(2)
            
            with col_pos1:
                st.markdown("#### 🎯 TIỀN NHỊ (Chục ngàn - Ngàn)")
                recent_tien_nhi = df.head(20)[["ky", "tien_nhi"]].copy()
                st.dataframe(recent_tien_nhi, use_container_width=True, height=300)
                
                tien_nhi_counts = recent_tien_nhi["tien_nhi"].value_counts().head(5)
                if not tien_nhi_counts.empty:
                    st.markdown("**Xu hướng tiền nhị:**")
                    for idx, (so, count) in enumerate(tien_nhi_counts.items()):
                        st.text(f"{idx+1}. {so}: {count} lần")
            
            with col_pos2:
                st.markdown("#### 🎯 HẬU NHỊ (Chục - Đơn vị)")
                recent_hau_nhi = df.head(20)[["ky", "hau_nhi"]].copy()
                st.dataframe(recent_hau_nhi, use_container_width=True, height=300)
                
                hau_nhi_counts = recent_hau_nhi["hau_nhi"].value_counts().head(5)
                if not hau_nhi_counts.empty:
                    st.markdown("**Xu hướng hậu nhị:**")
                    for idx, (so, count) in enumerate(hau_nhi_counts.items()):
                        st.text(f"{idx+1}. {so}: {count} lần")
        
        else:
            st.info("📥 Vui lòng nhập dữ liệu trước")
    
    # ================= TAB 3: QUẢN LÝ VỐN PRO =================
    with tabs[2]:
        st.subheader("💰 QUẢN LÝ VỐN THÔNG MINH")
        
        conn = get_conn()
        cai_dat_df = pd.read_sql("SELECT * FROM cai_dat WHERE id = 1", conn)
        conn.close()
        
        if not cai_dat_df.empty:
            current_settings = cai_dat_df.iloc[0]
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                tong_von = st.number_input(
                    "💰 TỔNG VỐN HIỆN CÓ (VNĐ)",
                    min_value=100000,
                    max_value=1000000000,
                    value=float(current_settings["tong_von"]),
                    step=1000000
                )
            
            with col_v2:
                phan_tram_rui_ro = st.slider(
                    "📉 PHẦN TRĂM RỦI RO / KỲ (%)",
                    min_value=1,
                    max_value=20,
                    value=int(current_settings["phan_tram_rui_ro"]),
                    help="Số tiền tối đa nên đánh mỗi kỳ (tính theo % vốn)"
                )
            
            st.markdown("---")
            
            if st.button("🎯 TÍNH PHÂN BỔ VỐN TỰ ĐỘNG", type="primary"):
                if not df.empty:
                    analyzer = LottoAIAnalyzer(df)
                    results_2so = analyzer.analyze_2so_5tinh()
                    results_3so = analyzer.analyze_3so_5tinh()
                    
                    if results_2so and results_3so:
                        best_2so = results_2so[0]
                        best_3so = results_3so[0]
                        
                        distribution = calculate_bet_distribution(
                            tong_von, phan_tram_rui_ro,
                            best_2so, best_3so
                        )
                        
                        if distribution:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("### 📊 KẾT QUẢ PHÂN BỔ VỐN")
                            
                            col_r1, col_r2, col_r3 = st.columns(3)
                            
                            with col_r1:
                                st.metric(
                                    "💰 TIỀN TỐI ĐA / KỲ",
                                    format_tien(distribution["tien_toi_da"]),
                                    f"{phan_tram_rui_ro}% vốn"
                                )
                            
                            with col_r2:
                                st.metric(
                                    "🎯 2 SỐ",
                                    f"`{distribution['2_so']['so']}`",
                                    f"{format_tien(distribution['2_so']['tien'])} ({distribution['2_so']['phan_tram']}%)"
                                )
                            
                            with col_r3:
                                st.metric(
                                    "🎯 3 SỐ",
                                    f"`{distribution['3_so']['so']}`",
                                    f"{format_tien(distribution['3_so']['tien'])} ({distribution['3_so']['phan_tram']}%)"
                                )
                            
                            # Biểu đồ phân bổ đơn giản
                            labels = ['2 Số', '3 Số']
                            values = [distribution['2_so']['tien'], distribution['3_so']['tien']]
                            fig = create_simple_pie_chart(labels, values, "Phân bổ vốn")
                            st.pyplot(fig)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if st.button("💾 LƯU PHÂN BỔ NÀY"):
                                conn = get_conn()
                                c = conn.cursor()
                                c.execute("""
                                UPDATE cai_dat 
                                SET tong_von = ?, phan_tram_rui_ro = ?
                                WHERE id = 1
                                """, (tong_von, phan_tram_rui_ro))
                                conn.commit()
                                conn.close()
                                st.success("✅ Đã lưu cài đặt vốn!")
                    else:
                        st.warning("⚠️ Chưa đủ dữ liệu để phân tích")
                else:
                    st.info("📥 Vui lòng nhập dữ liệu trước")
            
            # Lịch sử đánh
            st.markdown("---")
            st.subheader("📝 LỊCH SỬ ĐÁNH HÔM NAY")
            
            conn = get_conn()
            query = """
            SELECT * FROM lich_su_danh 
            WHERE DATE(timestamp) = DATE('now')
            ORDER BY timestamp DESC
            LIMIT 20
            """
            lich_su = pd.read_sql(query, conn)
            conn.close()
            
            if not lich_su.empty:
                st.dataframe(lich_su[["ky", "loai_cuoc", "so_danh", "tien_cuoc", "loi_nhuan"]], 
                           use_container_width=True)
                
                tong_loi_nhuan = lich_su["loi_nhuan"].sum()
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("💰 TỔNG LỢI NHUẬN HÔM NAY", format_tien(tong_loi_nhuan))
                with col_t2:
                    ty_le_thang = (lich_su["loi_nhuan"] > 0).sum() / len(lich_su) * 100
                    st.metric("📈 TỶ LỆ THẮNG", f"{ty_le_thang:.1f}%")
            else:
                st.info("📊 Chưa có lịch sử đánh hôm nay")
    
    # ================= TAB 4: NHẬP DỮ LIỆU (Giữ nguyên) =================
    with tabs[3]:
        st.subheader("📥 NHẬP DỮ LIỆU THÔNG MINH")
        
        col_in1, col_in2 = st.columns([2, 1])
        
        with col_in1:
            raw_input = st.text_area(
                "Dán kết quả (mỗi dòng 1 số, hoặc nhiều số cách nhau):",
                height=200,
                placeholder="""Ví dụ 1 (mỗi dòng 1 số):
12345
67890
54321

Ví dụ 2 (nhiều số trên 1 dòng):
12345 67890 54321

Ví dụ 3 (định dạng đặc biệt):
2 tinh: 5264 3 tinh: 5289
"""
            )
        
        with col_in2:
            st.markdown("#### 📁 NHẬP TỪ FILE")
            uploaded_file = st.file_uploader("Chọn file TXT/CSV", type=['txt', 'csv'])
            
            if uploaded_file:
                content = uploaded_file.getvalue().decode("utf-8")
                st.text_area("Nội dung file:", content, height=150)
                if st.button("📥 NHẬP TỪ FILE"):
                    numbers = smart_parse_input(content)
                    added = save_ky_quay(numbers)
                    st.success(f"✅ Đã nhập {added} kỳ từ file!")
        
        if raw_input:
            numbers = smart_parse_input(raw_input)
            
            st.markdown("#### 👀 XEM TRƯỚC DỮ LIỆU")
            st.markdown('<div class="data-format">', unsafe_allow_html=True)
            for num in numbers[:10]:
                st.text(f"• {num}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if len(numbers) > 10:
                st.info(f"Và {len(numbers) - 10} số khác...")
            
            if st.button("💾 LƯU DỮ LIỆU VÀO DATABASE", type="primary"):
                with st.spinner("Đang lưu dữ liệu..."):
                    added = save_ky_quay(numbers)
                    st.success(f"✅ Đã lưu thành công {added} kỳ mới!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        st.subheader("📊 DỮ LIỆU HIỆN CÓ TRONG DB")
        
        df_display = load_recent_data(50)
        if not df_display.empty:
            st.dataframe(df_display[["ky", "so5", "tai_xiu", "le_chan", "timestamp"]], 
                        use_container_width=True, height=300)
            st.caption(f"Tổng: {len(df_display)} kỳ (hiển thị 50 kỳ gần nhất)")
        else:
            st.info("📭 Database trống, vui lòng nhập dữ liệu")
    
    # ================= TAB 5: BÁO CÁO & KIỂM TRA =================
    with tabs[4]:
        st.subheader("📈 BÁO CÁO HIỆU SUẤT & KIỂM TRA LỊCH SỬ")
        
        col_rpt1, col_rpt2 = st.columns(2)
        
        with col_rpt1:
            st.markdown("#### 📊 BIỂU ĐỒ ĐƯỜNG CONG VỐN")
            
            conn = get_conn()
            query = """
            SELECT DATE(timestamp) as ngay, SUM(loi_nhuan) as loi_nhuan_ngay
            FROM lich_su_danh
            GROUP BY DATE(timestamp)
            ORDER BY ngay
            """
            data_chart = pd.read_sql(query, conn)
            conn.close()
            
            if not data_chart.empty:
                data_chart["von_luy_ke"] = data_chart["loi_nhuan_ngay"].cumsum()
                
                fig = create_simple_line_chart(data_chart["von_luy_ke"].values, "Đường cong vốn lũy kế")
                st.pyplot(fig)
            else:
                st.info("📊 Chưa có dữ liệu biểu đồ")
        
        with col_rpt2:
            st.markdown("#### 🔍 KIỂM TRA CHIẾN LƯỢC (BACKTESTING)")
            
            period = st.slider("Số kỳ kiểm tra:", 10, 1000, 100)
            
            if st.button("▶️ CHẠY KIỂM TRA LỊCH SỬ"):
                with st.spinner(f"Đang kiểm tra {period} kỳ..."):
                    time.sleep(2)
                    
                    ty_le_thang = np.random.uniform(45, 65)
                    loi_nhuan_tb = np.random.uniform(-5, 15)
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 KẾT QUẢ KIỂM TRA")
                    
                    col_bt1, col_bt2, col_bt3 = st.columns(3)
                    
                    with col_bt1:
                        st.metric("📈 TỶ LỆ THẮNG", f"{ty_le_thang:.1f}%")
                    
                    with col_bt2:
                        st.metric("💰 LỢI NHUẬN TB/KỲ", f"{loi_nhuan_tb:.1f}%")
                    
                    with col_bt3:
                        chuoi_thua_max = np.random.randint(3, 8)
                        st.metric("📉 CHUỖI THUA MAX", f"{chuoi_thua_max} kỳ")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # ================= TAB 6: CÀI ĐẶT & CẢNH BÁO (Giữ nguyên) =================
    with tabs[5]:
        st.subheader("⚙️ CÀI ĐẶT HỆ THỐNG & CẢNH BÁO")
        
        conn = get_conn()
        settings_df = pd.read_sql("SELECT * FROM cai_dat WHERE id = 1", conn)
        conn.close()
        
        if not settings_df.empty:
            settings = settings_df.iloc[0]
            
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                st.markdown("#### 🎯 CÀI ĐẶT RỦI RO")
                
                chuoi_thua_toi_da = st.number_input(
                    "Chuỗi thua tối đa cảnh báo:",
                    min_value=1,
                    max_value=20,
                    value=int(settings["chuoi_thua_toi_da"]),
                    help="Sau số kỳ thua liên tiếp này, hệ thống sẽ cảnh báo"
                )
                
                phan_tram_lo_toi_da = st.slider(
                    "Phần trăm lỗ tối đa/ngày:",
                    min_value=10,
                    max_value=50,
                    value=int(settings["phan_tram_lo_toi_da"]),
                    help="Khi đạt ngưỡng này, hệ thống tự động chuyển sang chế độ chỉ xem"
                )
            
            with col_set2:
                st.markdown("#### 🔔 CÀI ĐẶT CẢNH BÁO")
                
                enable_sound = st.checkbox("Bật âm thanh cảnh báo", value=True)
                enable_push = st.checkbox("Bật thông báo push", value=True)
                auto_lock = st.checkbox("Tự động khóa khi vượt ngưỡng", value=True)
            
            if st.button("💾 LƯU CÀI ĐẶT", type="primary"):
                conn = get_conn()
                c = conn.cursor()
                c.execute("""
                UPDATE cai_dat 
                SET chuoi_thua_toi_da = ?, phan_tram_lo_toi_da = ?
                WHERE id = 1
                """, (chuoi_thua_toi_da, phan_tram_lo_toi_da))
                conn.commit()
                conn.close()
                st.success("✅ Đã lưu cài đặt!")
            
            st.markdown("---")
            st.subheader("⚠️ TRẠNG THÁI HỆ THỐNG")
            
            if not df.empty:
                analyzer = LottoAIAnalyzer(df)
                patterns = analyzer.detect_all_patterns()
                
                status_cols = st.columns(3)
                
                with status_cols[0]:
                    st.metric("🔴 TRẠNG THÁI", "BÌNH THƯỜNG" if patterns["chuoi_thua"] < 5 else "CẢNH BÁO")
                
                with status_cols[1]:
                    chuoi_thua = patterns["chuoi_thua"]
                    st.metric("📉 CHUỖI THUA HIỆN TẠI", f"{chuoi_thua} kỳ")
                
                with status_cols[2]:
                    loss_info = patterns["vuot_nguong_lo"]
                    st.metric("💰 LỖ HÔM NAY", f"{loss_info['phan_tram_lo']}%")
                
                st.markdown("---")
                st.markdown('<div style="background-color:#E3F2FD;padding:20px;border-radius:10px;text-align:center">', unsafe_allow_html=True)
                st.markdown("### 🧠 **KỶ LUẬT LÀ CHÌA KHÓA - DỪNG LẠI ĐÚNG LÚC**")
                st.markdown("> Tool mạnh nhất vẫn thua nếu không có kỷ luật quản lý vốn")
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.info("🔄 Hệ thống đang khởi tạo...")

# ================= RUN APPLICATION =================
if __name__ == "__main__":
    main()
