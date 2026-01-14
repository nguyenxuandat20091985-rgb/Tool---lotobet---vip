import streamlit as st
import pandas as pd
import re, os
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.8",
    layout="centered",
    page_icon="🧠"
)

MIN_DATA = 40
LOOKBACK_WR = 30
STOP_LOSS_STREAK = 3
TAKE_PROFIT_DAY = 15
BASE_BANKROLL = 100

RAW_FILE   = "raw_numbers.csv"
PAIR2_FILE = "pair2.csv"
PAIR3_FILE = "pair3.csv"
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

def save_df(df, path):
    df.to_csv(path, index=False)

def next_ky(df):
    return 1 if df.empty else int(df["ky"].astype(int).max()) + 1

# ================= STORAGE (DELTA) =================
def save_numbers(numbers):
    raw = load_df(RAW_FILE, ["time","ky","number"])
    p2  = load_df(PAIR2_FILE,["time","ky","pair"])
    p3  = load_df(PAIR3_FILE,["time","ky","pair"])

    ky = next_ky(raw)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for num in numbers:
        # ❌ trùng tuyệt đối
        if not raw[raw["number"] == num].empty:
            continue

        raw.loc[len(raw)] = [now, ky, num]

        pair2 = num[-2:]
        pair3 = num[-3:]

        # ❌ trùng liên tiếp
        if p2.empty or p2.iloc[-1]["pair"] != pair2:
            p2.loc[len(p2)] = [now, ky, pair2]

        if p3.empty or p3.iloc[-1]["pair"] != pair3:
            p3.loc[len(p3)] = [now, ky, pair3]

        ky += 1
        added += 1

    save_df(raw, RAW_FILE)
    save_df(p2, PAIR2_FILE)
    save_df(p3, PAIR3_FILE)
    return added

# ================= RESULT TRACK =================
def log_result(pair, hit):
    df = load_df(RESULT_LOG, ["time","pair","result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    save_df(df, RESULT_LOG)

def win_rate(pair, lookback=LOOKBACK_WR):
    df = load_df(RESULT_LOG, ["time","pair","result"])
    d = df[df["pair"] == pair].tail(lookback)
    if d.empty:
        return 0.0
    return round((d["result"] == "TRÚNG").mean() * 100, 2)

def loss_streak_recent(n=STOP_LOSS_STREAK):
    df = load_df(RESULT_LOG, ["time","pair","result"]).tail(n)
    if len(df) < n:
        return False
    return all(df["result"] == "TRƯỢT")

def profit_today():
    df = load_df(RESULT_LOG, ["time","pair","result"])
    if df.empty:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    d = df[df["time"].str.startswith(today)]
    win = (d["result"] == "TRÚNG").sum()
    lose = (d["result"] == "TRƯỢT").sum()
    return win - lose

# ================= AI CORE =================
def cycle_score(seq, pair):
    pos = [i for i,p in enumerate(seq) if p == pair]
    if len(pos) < 3:
        return -5, "Thiếu dữ liệu"

    gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
    avg = sum(gaps[-3:]) / len(gaps[-3:])
    last = len(seq) - 1 - pos[-1]

    if abs(last - avg) <= 1:
        return 20, "🎯 Đúng nhịp"
    elif last < avg:
        return -10, "⏳ Vừa ra"
    else:
        return 15, "⚠️ Quá hạn"

def analyze(pair_df):
    if len(pair_df) < MIN_DATA:
        return pd.DataFrame()

    seq = pair_df["pair"].tolist()
    last10 = seq[-10:]
    last20 = seq[-20:]

    cnt_all = Counter(seq)
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    rows = []
    for pair in cnt_all:
        freq = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / len(seq)) * 0.2
        ) * 100

        c_score, c_note = cycle_score(seq, pair)
        biet = -20 if cnt10[pair] >= 4 else 0

        score = round(freq + c_score + biet, 2)
        if score <= 0:
            continue

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": c_note,
            "WR(30)": win_rate(pair)
        })

    return pd.DataFrame(rows).sort_values("Điểm AI (%)", ascending=False)

def make_dan(df):
    strong = df[(df["Điểm AI (%)"] >= 65) & (df["WR(30)"] >= 25)]
    return {
        "Dàn 1": strong.head(1)["Cặp"].tolist(),
        "Dàn 3": strong.head(3)["Cặp"].tolist(),
        "Dàn 5": strong.head(5)["Cặp"].tolist()
    }

def stake_suggestion(conf):
    if conf >= 85:
        return 3
    if conf >= 75:
        return 2
    if conf >= 65:
        return 1
    return 0

# ================= UI =================
st.title("🧠 LOTOBET AUTO PRO – AI V3.8")

st.subheader("📥 NẠP DỮ LIỆU (Real-time)")
tabs = st.tabs(["✍️ Dán tay", "📂 Upload"])
nums = []

with tabs[0]:
    raw = st.text_area("Dán số 5 chữ số", height=120)
    nums += re.findall(r"\d{5}", raw)

with tabs[1]:
    f = st.file_uploader("Upload .txt / .csv", type=["txt","csv"])
    if f:
        nums += re.findall(r"\d{5}", f.read().decode("utf-8"))

if st.button("💾 LƯU & PHÂN TÍCH"):
    if not nums:
        st.error("Không có dữ liệu hợp lệ")
    else:
        with st.spinner("🧠 AI đang phân tích DELTA..."):
            added = save_numbers(nums)
        st.success(f"✅ Đã lưu {added} số mới")

st.divider()

# ================= ANALYSIS =================
pair2 = load_df(PAIR2_FILE, ["time","ky","pair"])
pair3 = load_df(PAIR3_FILE, ["time","ky","pair"])

for label, dfp in [("2 TỈNH", pair2), ("3 TỈNH", pair3)]:
    st.subheader(f"🔥 TOP {label}")

    a = analyze(dfp)
    if a.empty:
        st.warning("Chưa đủ dữ liệu")
        continue

    st.dataframe(a.head(5), use_container_width=True, hide_index=True)
    best = a.iloc[0]

    st.markdown(f"""
    **🎯 Cặp:** `{best['Cặp']}`  
    **📊 Điểm AI:** `{best['Điểm AI (%)']}%`  
    **🔁 Cầu:** {best['Cầu']}  
    **📈 WR(30):** `{best['WR(30)']}%`
    """)

    # ===== BIỂU ĐỒ NHỊP =====
    st.line_chart(dfp["pair"].astype("category").cat.codes.tail(30))

    # ===== DÀN =====
    dan = make_dan(a)
    st.write("🎯 **DÀN ĐỀ XUẤT**")
    st.write("• Dàn 1:", dan["Dàn 1"])
    st.write("• Dàn 3:", dan["Dàn 3"])
    st.write("• Dàn 5:", dan["Dàn 5"])

    # ===== BANKROLL / RISK =====
    conf = best["Điểm AI (%)"]
    stake = stake_suggestion(conf)

    if loss_streak_recent():
        st.error("⛔ Chuỗi thua gần đây – BẮT BUỘC DỪNG")
    elif profit_today() >= TAKE_PROFIT_DAY:
        st.warning("💰 Đã đạt take-profit hôm nay – NÊN DỪNG")
    elif stake == 0:
        st.warning("⚠️ Độ tin cậy thấp – THEO DÕI")
    else:
        st.success(f"✅ GỢI Ý ĐÁNH: {stake} tay")

    # ===== GHI NHẬN =====
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"✅ TRÚNG ({label})"):
            log_result(best["Cặp"], True)
            st.success("Đã ghi TRÚNG")
    with c2:
        if st.button(f"❌ TRƯỢT ({label})"):
            log_result(best["Cặp"], False)
            st.warning("Đã ghi TRƯỢT")

st.caption("⚠️ AI hỗ trợ xác suất – thắng thua phụ thuộc kỷ luật & quản lý vốn")
