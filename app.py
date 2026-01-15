import streamlit as st
import pandas as pd
import sqlite3
import re
import numpy as np
from datetime import datetime
from collections import Counter

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET ULTRA AI – V10.0",
    layout="wide",
    page_icon="🎯"
)

DB_FILE = "lotobet_ultra.db"

# ================== DATABASE ==================
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        ky INTEGER,
        number5 TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS pair2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky INTEGER,
        pair TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS pair3 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky INTEGER,
        pair TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================== UTIL ==================
def next_ky():
    conn = get_conn()
    df = pd.read_sql("SELECT MAX(ky) ky FROM raw_data", conn)
    conn.close()
    if df.iloc[0]["ky"] is None:
        return 1
    return int(df.iloc[0]["ky"]) + 1

def normalize_input(text):
    nums = re.findall(r"\d{5}", text)
    return nums

def save_numbers(nums):
    if not nums:
        return 0

    conn = get_conn()
    ky = next_ky()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for n in nums:
        # chống trùng tuyệt đối
        check = pd.read_sql(
            "SELECT 1 FROM raw_data WHERE number5 = ?",
            conn, params=(n,)
        )
        if not check.empty:
            continue

        conn.execute(
            "INSERT INTO raw_data (time, ky, number5) VALUES (?, ?, ?)",
            (now, ky, n)
        )

        p2 = n[-2:]
        p3 = n[-3:]

        conn.execute("INSERT INTO pair2 (ky, pair) VALUES (?, ?)", (ky, p2))
        conn.execute("INSERT INTO pair3 (ky, pair) VALUES (?, ?)", (ky, p3))

        ky += 1
        added += 1

    conn.commit()
    conn.close()
    return added

def load_df(table):
    conn = get_conn()
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

# ================== AI CORE ==================
def ai_analyze(df, label):
    if df.empty:
        return pd.DataFrame()

    seq = df["pair"].tolist()
    total = len(seq)
    last20 = seq[-20:]

    cnt_all = Counter(seq)
    cnt20 = Counter(last20)

    rows = []
    for pair in cnt_all:
        freq = cnt20[pair] / 20 if pair in cnt20 else 0
        freq_all = cnt_all[pair] / total

        # cầu lặp
        pos = [i for i, p in enumerate(seq) if p == pair]
        gap_score = 0
        if len(pos) >= 3:
            gaps = np.diff(pos[-3:])
            avg_gap = np.mean(gaps)
            last_gap = total - pos[-1]
            if abs(last_gap - avg_gap) <= 1:
                gap_score = 20
            elif last_gap < avg_gap:
                gap_score = -10
            else:
                gap_score = -15

        score = round((freq * 60 + freq_all * 40) + gap_score, 2)

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Tỷ lệ trúng (%)": round(freq_all * 100, 2)
        })

    out = pd.DataFrame(rows)
    out = out[out["Điểm AI (%)"] > 0]
    out = out.sort_values("Điểm AI (%)", ascending=False)
    return out

def tai_xiu(num):
    s = sum(int(x) for x in num)
    return "TÀI" if s >= 23 else "XỈU"

# ================== UI ==================
st.title("🎯 LOTOBET ULTRA AI – V10.0")

# -------- INPUT --------
st.subheader("📥 NHẬP DỮ LIỆU (TỰ ĐỘNG)")
raw = st.text_area(
    "Dán kết quả (mỗi số 5 chữ số – dán cột hay dòng đều được)",
    height=120
)

if st.button("💾 LƯU DỮ LIỆU"):
    nums = normalize_input(raw)
    added = save_numbers(nums)
    st.success(f"Đã lưu {added} kỳ mới")

# -------- LOAD DATA --------
raw_df = load_df("raw_data")
pair2_df = load_df("pair2")
pair3_df = load_df("pair3")

# ================== DASHBOARD ==================
st.divider()

colA, colB, colC, colD = st.columns(4)

# ===== KHUNG A =====
with colA:
    st.markdown("## 📊 TỔNG KỲ")
    st.metric("Tổng kỳ", len(raw_df))
    if not raw_df.empty:
        last = raw_df.iloc[-1]
        st.caption(f"Kỳ gần nhất: #{last['ky']}")

# ===== KHUNG B =====
with colB:
    st.markdown("## 🔁 2 TINH")
    st.caption(f"( 2 tinh: {len(pair2_df)} • 3 tinh: {len(pair3_df)} )")
    analysis2 = ai_analyze(pair2_df, "2")
    if analysis2.empty:
        st.warning("Chưa đủ dữ liệu")
    else:
        best2 = analysis2.iloc[0]
        st.metric("ĐÁNH 2 SỐ", best2["Cặp"])
        st.write("Điểm AI:", best2["Điểm AI (%)"], "%")

# ===== KHUNG C =====
with colC:
    st.markdown("## 🔁 3 TINH")
    st.caption(f"( 2 tinh: {len(pair2_df)} • 3 tinh: {len(pair3_df)} )")
    analysis3 = ai_analyze(pair3_df, "3")
    if analysis3.empty:
        st.warning("Chưa đủ dữ liệu")
    else:
        best3 = analysis3.iloc[0]
        st.metric("ĐÁNH 3 SỐ", best3["Cặp"])
        st.write("Điểm AI:", best3["Điểm AI (%)"], "%")

# ===== KHUNG D =====
with colD:
    st.markdown("## 🎯 SỐ CẦN ĐÁNH")
    if not analysis2.empty:
        st.success(f"2 SỐ: {analysis2.iloc[0]['Cặp']}")
    if not analysis3.empty:
        st.success(f"3 SỐ: {analysis3.iloc[0]['Cặp']}")
    st.caption("Dựa trên AI lịch sử – không phải may rủi")

# ================== PHÂN TÍCH NÂNG CAO ==================
st.divider()
st.subheader("📊 PHÂN TÍCH BỔ SUNG")

if not raw_df.empty:
    last_num = raw_df.iloc[-1]["number5"]
    st.write("🎲 Tài / Xỉu kỳ gần nhất:", tai_xiu(last_num))
    st.write("🔄 Trạng thái:", "🔁 Đang phân tích cầu...")

st.caption("⚠️ Tool hỗ trợ xác suất – quản lý vốn & kỷ luật là bắt buộc")
