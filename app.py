import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.5",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
RESULT_LOG = "result_log.csv"
MIN_DATA = 40

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

# ================= SAVE DATA (ANTI DUPLICATE + 2/3 TỈNH) =================
def save_pairs_unique(pairs_2, pairs_3):
    df = load_csv(DATA_FILE, ["time", "pair", "kind"])
    existing = list(zip(df["pair"], df["kind"]))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_count = 0

    for p in pairs_2:
        p = str(p).zfill(2)
        key = (p, "2")
        if len(existing) > 0 and key == existing[-1]:
            continue
        df.loc[len(df)] = [now, p, "2"]
        existing.append(key)
        new_count += 1

    for p in pairs_3:
        p = str(p).zfill(3)
        key = (p, "3")
        if len(existing) > 0 and key == existing[-1]:
            continue
        df.loc[len(df)] = [now, p, "3"]
        existing.append(key)
        new_count += 1

    save_csv(df, DATA_FILE)
    return new_count

# ================= RESULT TRACK =================
def log_result(pair, kind, hit):
    df = load_csv(RESULT_LOG, ["time", "pair", "kind", "result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        kind,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    save_csv(df, RESULT_LOG)

def win_rate(pair, kind, lookback=30):
    df = load_csv(RESULT_LOG, ["time", "pair", "kind", "result"])
    df = df[(df["pair"] == pair) & (df["kind"] == kind)].tail(lookback)
    if len(df) == 0:
        return 0
    return round((df["result"] == "TRÚNG").mean() * 100, 2)

# ================= CYCLE / REPEAT =================
def cycle_score(df, pair):
    seq = df["pair"].tolist()
    pos = [i for i, p in enumerate(seq) if p == pair]
    if len(pos) < 3:
        return -5, "Thiếu dữ liệu"
    gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
    avg_gap = sum(gaps[-3:]) / len(gaps[-3:])
    last_gap = len(seq) - 1 - pos[-1]

    if abs(last_gap - avg_gap) <= 1:
        return 20, "🎯 Đúng nhịp"
    elif last_gap < avg_gap:
        return -10, "⏳ Vừa ra"
    else:
        return -15, "⚠️ Quá hạn"

# ================= CORE AI =================
def analyze_v35(df, kind="2"):
    df = df[df["kind"] == kind]
    total = len(df)

    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    rows = []

    for pair in cnt_all:
        freq_score = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / total) * 0.2
        ) * 100

        c_score, c_note = cycle_score(df, pair)

        biet = -20 if cnt10[pair] >= 4 else 0

        score = round(freq_score + c_score + biet, 2)
        rate = win_rate(pair, kind)

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": c_note,
            "Tỷ lệ trúng (%)": rate
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("Điểm AI (%)", ascending=False)
    out = out[out["Điểm AI (%)"] > 0]

    return out

# ================= STORAGE V3.6 (FOR V3.5 UI) =================
RAW_FILE = "raw_input.csv"
PAIR2_FILE = "pair_2.csv"
PAIR3_FILE = "pair_3.csv"

def load_df(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]
    return pd.DataFrame(columns=cols)

def next_ky(df):
    if df.empty:
        return 1
    return int(df["ky"].max()) + 1

def save_numbers_v36(numbers):
    """
    numbers: list số 5 chữ số
    """
    raw_df = load_df(RAW_FILE, ["time", "ky", "number5"])
    pair2_df = load_df(PAIR2_FILE, ["time", "ky", "pair"])
    pair3_df = load_df(PAIR3_FILE, ["time", "ky", "pair"])

    ky = next_ky(raw_df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    added = 0

    for num in numbers:
        # ❌ TRÙNG TUYỆT ĐỐI
        if not raw_df[raw_df["number5"] == num].empty:
            continue

        raw_df.loc[len(raw_df)] = [now, ky, num]

        # ===== 2 TINH =====
        p2 = num[-2:]
        if pair2_df.empty or pair2_df.iloc[-1]["pair"] != p2:
            pair2_df.loc[len(pair2_df)] = [now, ky, p2]

        # ===== 3 TINH =====
        p3 = num[-3:]
        if pair3_df.empty or pair3_df.iloc[-1]["pair"] != p3:
            pair3_df.loc[len(pair3_df)] = [now, ky, p3]

        added += 1
        ky += 1

    raw_df.to_csv(RAW_FILE, index=False)
    pair2_df.to_csv(PAIR2_FILE, index=False)
    pair3_df.to_csv(PAIR3_FILE, index=False)

    return added

# ===== 2 TỈNH =====
st.subheader("🔥 TOP 2 TỈNH")
analysis_2 = analyze_v35(df, "2")
st.dataframe(analysis_2.head(5), use_container_width=True, hide_index=True)

best2 = analysis_2.iloc[0]
st.markdown(f"""
### 🧠 KẾT LUẬN 2 TỈNH
- 🎯 **Cặp:** `{best2['Cặp']}`
- 📊 **Điểm AI:** `{best2['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best2['Cầu']}
- ✅ **Tỷ lệ trúng:** `{best2['Tỷ lệ trúng (%)']}%`
""")
st.divider()
st.subheader("🔥 TOP 3 TỈNH")

analysis_3 = analyze_v35(df, "3")

if analysis_3.empty:
    st.warning("⚠️ Chưa có đủ dữ liệu hoặc cầu 3 tỉnh chưa đạt điều kiện")
else:
    st.dataframe(
        analysis_3.head(5),
        use_container_width=True,
        hide_index=True
    )

    best3 = analysis_3.iloc[0]

    st.markdown(f"""
    ### 🧠 KẾT LUẬN 3 TỈNH
    - 🎯 **Cặp:** `{best3['Cặp']}`
    - 📊 **Điểm AI:** `{best3['Điểm AI (%)']}%`
    - 🔁 **Cầu:** {best3['Cầu']}
    - ✅ **Tỷ lệ trúng:** `{best3['Tỷ lệ trúng (%)']}%`
    """)
# ===== 3 TỈNH =====
st.subheader("🔥 TOP 3 TỈNH")
analysis_3 = analyze_v35(df, "3")
st.dataframe(analysis_3.head(5), use_container_width=True, hide_index=True)

best3 = analysis_3.iloc[0]
st.markdown(f"""
### 🧠 KẾT LUẬN 3 TỈNH
- 🎯 **Cặp:** `{best3['Cặp']}`
- 📊 **Điểm AI:** `{best3['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best3['Cầu']}
- ✅ **Tỷ lệ trúng:** `{best3['Tỷ lệ trúng (%)']}%`
""")

st.caption("⚠️ AI hỗ trợ xác suất – quản lý vốn & kỷ luật là bắt buộc")
