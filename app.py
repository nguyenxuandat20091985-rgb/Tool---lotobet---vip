import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.2",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
RESULT_LOG = "result_log.csv"

MIN_DATA = 40

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

# ================= SAVE DATA =================
def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in pairs:
        df.loc[len(df)] = [now, p]
    save_csv(df, DATA_FILE)

# ================= REPEAT CYCLE =================
def repeat_cycle_score(df, pair):
    seq = df["pair"].tolist()
    pos = [i for i, p in enumerate(seq) if p == pair]

    if len(pos) < 3:
        return 0, "Thiếu dữ liệu"

    gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
    avg_gap = sum(gaps[-3:]) / len(gaps[-3:])
    last_gap = len(seq) - 1 - pos[-1]

    if abs(last_gap - avg_gap) <= 1:
        return 15, "🎯 Đúng nhịp"
    elif last_gap < avg_gap:
        return -5, "⏳ Vừa ra"
    else:
        return -10, "⚠️ Quá hạn"

# ================= RESULT TRACK =================
def log_result(pair, hit):
    df = load_csv(RESULT_LOG, ["time", "pair", "result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    save_csv(df, RESULT_LOG)

def win_rate(pair, lookback=30):
    df = load_csv(RESULT_LOG, ["time", "pair", "result"])
    df = df[df["pair"] == pair].tail(lookback)
    if len(df) == 0:
        return 0
    return round((df["result"] == "TRÚNG").mean() * 100, 2)

# ================= ANALYSIS =================
def analyze_v32(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    results = []

    for pair in cnt_all:
        base = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / total) * 0.2
        ) * 100

        cycle_score, cycle_note = repeat_cycle_score(df, pair)
        score = round(base + cycle_score, 2)

        rate = win_rate(pair)

        results.append({
            "pair": pair,
            "score": score,
            "cycle": cycle_note,
            "win_rate": rate
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.2")

raw = st.text_area("📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)")

if st.button("💾 LƯU KỲ"):
    nums = re.findall(r"\d{5}", raw)
    pairs = [n[-2:] for n in nums]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Sai định dạng")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI phân tích")
    st.stop()

st.divider()

analysis = analyze_v32(df)

st.subheader("🔥 TOP CẶP ĐỀ XUẤT")
st.table(analysis[:5])

best = analysis[0]

st.subheader("🧠 KẾT LUẬN AI")
st.markdown(f"""
- **Cặp đề xuất:** `{best['pair']}`
- **Điểm AI:** `{best['score']}%`
- **Cầu lặp:** {best['cycle']}
- **Tỷ lệ trúng (30 kỳ):** `{best['win_rate']}%`
""")

if best["score"] >= 60 and best["win_rate"] >= 25:
    st.success("✅ ĐỦ ĐIỀU KIỆN VÀO TIỀN")
else:
    st.warning("⚠️ NÊN THEO DÕI – CHƯA AN TOÀN")

# ================= LOG RESULT =================
st.subheader("🧾 GHI NHẬN KẾT QUẢ KỲ NÀY")
col1, col2 = st.columns(2)

with col1:
    if st.button("✅ TRÚNG"):
        log_result(best["pair"], True)
        st.success("Đã ghi TRÚNG")

with col2:
    if st.button("❌ TRƯỢT"):
        log_result(best["pair"], False)
        st.warning("Đã ghi TRƯỢT")

st.caption("⚠️ AI hỗ trợ xác suất – kỷ luật & quản lý vốn quyết định lợi nhuận")
