import streamlit as st
import pandas as pd
import re, os
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.7 FULL",
    layout="centered",
    page_icon="🎯"
)

RAW_FILE   = "raw_5so.csv"
PAIR2_FILE = "pair_2.csv"
PAIR3_FILE = "pair_3.csv"
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

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ================= SAVE DATA =================
def save_numbers(numbers):
    raw = load_csv(RAW_FILE, ["time","number"])
    p2  = load_csv(PAIR2_FILE, ["time","pair"])
    p3  = load_csv(PAIR3_FILE, ["time","pair"])

    added = 0

    for n in numbers:
        if n in raw["number"].values:
            continue

        raw.loc[len(raw)] = [now(), n]

        d2 = n[-2:]
        d3 = n[-3:]

        if p2.empty or p2.iloc[-1]["pair"] != d2:
            p2.loc[len(p2)] = [now(), d2]

        if p3.empty or p3.iloc[-1]["pair"] != d3:
            p3.loc[len(p3)] = [now(), d3]

        added += 1

    save_csv(raw, RAW_FILE)
    save_csv(p2, PAIR2_FILE)
    save_csv(p3, PAIR3_FILE)

    return added

# ================= RESULT MEMORY =================
def log_result(pair, hit):
    df = load_csv(RESULT_LOG, ["time","pair","result"])
    df.loc[len(df)] = [now(), pair, "TRÚNG" if hit else "TRƯỢT"]
    save_csv(df, RESULT_LOG)

def win_rate(pair, lookback=30):
    df = load_csv(RESULT_LOG, ["time","pair","result"])
    df = df[df["pair"] == pair].tail(lookback)
    if df.empty:
        return 0
    return round((df["result"] == "TRÚNG").mean()*100,2)

# ================= AI CORE =================
def analyze_pair(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    rows = []

    for pair in cnt_all:
        score = (
            (cnt10[pair]/10)*0.5 +
            (cnt20[pair]/20)*0.3 +
            (cnt_all[pair]/total)*0.2
        ) * 100

        rate = win_rate(pair)

        if cnt10[pair] >= 4:
            score -= 25
        elif cnt10[pair] == 3:
            score += 10

        score = round(score,2)

        if score <= 0:
            continue

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Tỷ lệ trúng (%)": rate
        })

    if not rows:
        return pd.DataFrame(columns=["Cặp","Điểm AI (%)","Tỷ lệ trúng (%)"])

    return pd.DataFrame(rows).sort_values("Điểm AI (%)", ascending=False)

# ================= TÀI XỈU =================
def tai_xiu_stats(raw):
    nums = raw["number"].astype(str)
    tx = []
    for n in nums:
        s = sum(int(x) for x in n)
        tx.append("TÀI" if s >= 23 else "XỈU")
    return Counter(tx)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.7 FULL")

raw_input = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)", height=120)

if st.button("💾 LƯU DỮ LIỆU"):
    nums = re.findall(r"\d{5}", raw_input)
    if nums:
        added = save_numbers(nums)
        st.success(f"Đã lưu {added} kỳ (tự loại trùng)")
    else:
        st.error("Sai định dạng")

raw_df  = load_csv(RAW_FILE, ["time","number"])
pair2_df = load_csv(PAIR2_FILE, ["time","pair"])
pair3_df = load_csv(PAIR3_FILE, ["time","pair"])

st.info(f"""
📊 Tổng kỳ: {len(raw_df)}  
• 2 tinh: {len(pair2_df)}  
• 3 tinh: {len(pair3_df)}
""")

if len(pair2_df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI phân tích")
    st.stop()

# ================= 2 TINH =================
st.divider()
st.subheader("🔥 TOP 2 TINH")

analysis2 = analyze_pair(pair2_df)

if analysis2.empty:
    st.warning("Không có cầu 2 tinh phù hợp")
else:
    st.dataframe(analysis2.head(5), use_container_width=True, hide_index=True)
    best2 = analysis2.iloc[0]

    st.markdown(f"""
    **Cặp 2 tinh:** `{best2['Cặp']}`  
    **Điểm AI:** `{best2['Điểm AI (%)']}%`  
    **Tỷ lệ trúng:** `{best2['Tỷ lệ trúng (%)']}%`
    """)

# ================= 3 TINH =================
st.divider()
st.subheader("🔥 TOP 3 TINH")

analysis3 = analyze_pair(pair3_df)

if analysis3.empty:
    st.warning("Không có cầu 3 tinh phù hợp")
else:
    st.dataframe(analysis3.head(5), use_container_width=True, hide_index=True)
    best3 = analysis3.iloc[0]

    st.markdown(f"""
    **Cặp 3 tinh:** `{best3['Cặp']}`  
    **Điểm AI:** `{best3['Điểm AI (%)']}%`  
    **Tỷ lệ trúng:** `{best3['Tỷ lệ trúng (%)']}%`
    """)

# ================= TÀI XỈU =================
st.divider()
st.subheader("⚖️ TÀI / XỈU")

tx = tai_xiu_stats(raw_df)
st.write(dict(tx))

# ================= SỔ ĐỀ =================
st.divider()
st.subheader("📘 SỔ ĐỀ")

log_df = load_csv(RESULT_LOG, ["time","pair","result"])
if log_df.empty:
    st.info("Chưa có lịch sử trúng/trượt")
else:
    st.dataframe(log_df.tail(10), use_container_width=True, hide_index=True)

st.caption("⚠️ AI hỗ trợ xác suất – kỷ luật & quản lý vốn quyết định kết quả")
