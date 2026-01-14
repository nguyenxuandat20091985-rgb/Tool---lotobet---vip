import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.4",
    layout="centered",
    page_icon="🎯"
)

MIN_DATA = 40

RAW_FILE  = "raw_5so.csv"
PAIR2_FILE = "pair_2.csv"
PAIR3_FILE = "pair_3.csv"
RESULT_LOG = "result_log.csv"

# ================= UTIL =================
def load_df(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def next_ky(df):
    if df.empty:
        return 1
    return int(df["ky"].astype(int).max()) + 1

# ================= STORAGE V3.4++ =================
def save_numbers_v34(numbers):
    raw_df = load_df(RAW_FILE, ["time", "ky", "number5"])
    p2_df  = load_df(PAIR2_FILE, ["time", "ky", "pair"])
    p3_df  = load_df(PAIR3_FILE, ["time", "ky", "pair"])

    ky  = next_ky(raw_df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for num in numbers:
        if not raw_df[raw_df["number5"] == num].empty:
            continue

        raw_df.loc[len(raw_df)] = [now, ky, num]

        p2 = num[-2:]
        if p2_df.empty or p2_df.iloc[-1]["pair"] != p2:
            p2_df.loc[len(p2_df)] = [now, ky, p2]

        p3 = num[-3:]
        if p3_df.empty or p3_df.iloc[-1]["pair"] != p3:
            p3_df.loc[len(p3_df)] = [now, ky, p3]

        ky += 1
        added += 1

    raw_df.to_csv(RAW_FILE, index=False)
    p2_df.to_csv(PAIR2_FILE, index=False)
    p3_df.to_csv(PAIR3_FILE, index=False)

    return added

# ================= RESULT TRACK =================
def log_result(pair, hit):
    df = load_df(RESULT_LOG, ["time", "pair", "result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    df.to_csv(RESULT_LOG, index=False)

def win_rate(pair, lookback=30):
    df = load_df(RESULT_LOG, ["time", "pair", "result"])
    df = df[df["pair"] == pair].tail(lookback)
    if df.empty:
        return 0
    return round((df["result"] == "TRÚNG").mean() * 100, 2)

# ================= CYCLE =================
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

# ================= AI CORE =================
def analyze_v34(df):
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

        bet_penalty = -20 if cnt10[pair] >= 4 else 0

        score = round(freq_score + c_score + bet_penalty, 2)
        rate = win_rate(pair)

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": c_note,
            "Tỷ lệ trúng (%)": rate
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out[df_out["Điểm AI (%)"] > 0]
    return df_out.sort_values("Điểm AI (%)", ascending=False)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.4")

raw = st.text_area("📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)", height=120)

if st.button("💾 LƯU KỲ"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        added = save_numbers_v34(nums)
        st.success(f"Đã lưu {added} kỳ (2 & 3 tinh không cố định)")
    else:
        st.error("Sai định dạng dữ liệu")

df = load_df(PAIR2_FILE, ["time", "ky", "pair"])
st.info(f"📊 Tổng dữ liệu 2 tinh: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI phân tích")
    st.stop()

st.divider()
analysis = analyze_v34(df)

st.subheader("🔥 TOP 5 CẶP ĐỀ XUẤT")
st.dataframe(analysis.head(5), use_container_width=True, hide_index=True)

best = analysis.iloc[0]

st.subheader("🧠 KẾT LUẬN AI")
st.markdown(f"""
### 🎯 Cặp đề xuất: **{best['Cặp']}**
- 📊 **Điểm AI:** `{best['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best['Cầu']}
- ✅ **Tỷ lệ trúng (30 kỳ):** `{best['Tỷ lệ trúng (%)']}%`
""")

if best["Điểm AI (%)"] >= 65 and best["Tỷ lệ trúng (%)"] >= 25:
    st.success("✅ ĐỦ ĐIỀU KIỆN VÀO TIỀN")
else:
    st.warning("⚠️ NÊN THEO DÕI – CHƯA AN TOÀN")

st.divider()
st.subheader("🧾 GHI NHẬN KẾT QUẢ KỲ NÀY")

c1, c2 = st.columns(2)
with c1:
    if st.button("✅ TRÚNG"):
        log_result(best["Cặp"], True)
        st.success("Đã ghi TRÚNG")
with c2:
    if st.button("❌ TRƯỢT"):
        log_result(best["Cặp"], False)
        st.warning("Đã ghi TRƯỢT")

st.caption("⚠️ AI hỗ trợ xác suất – quản lý vốn & kỷ luật là bắt buộc")
