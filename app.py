import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V4",
    layout="centered",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
AI_FILE = "ai_weight.csv"

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([{"time": now, "pair": p} for p in pairs])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def log_prediction(pair, score, advice):
    df = load_csv(LOG_FILE, ["time", "pair", "score", "advice"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [now, pair, score, advice]
    df.to_csv(LOG_FILE, index=False)

# ================= AI =================
def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, good=True):
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.15 if good else -0.1
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    ai.to_csv(AI_FILE, index=False)

# ================= ANALYSIS =================
def analyze_v4(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    ai = load_ai()
    ai_map = dict(zip(ai["pair"], ai["weight"]))

    result = []
    for pair in cnt_all:
        if cnt10[pair] == 0 and cnt20[pair] <= 1:
            continue  # quá cold → loại

        base = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / total) * 0.2
        )
        weight = ai_map.get(pair, 1.0)
        score = round(base * weight * 100, 2)

        result.append({
            "pair": pair,
            "10k": cnt10[pair],
            "20k": cnt20[pair],
            "score": score
        })

    return sorted(result, key=lambda x: x["score"], reverse=True)

def backtest(df, pair, lookback=30):
    hits = 0
    total = min(lookback, len(df))
    for i in range(1, total+1):
        if df.iloc[-i]["pair"] == pair:
            hits += 1
    rate = round(hits / total * 100, 2) if total else 0
    return rate

# ================= UI =================
st.title("🟢 LOTOBET AUTO PRO – V4")
st.caption("AI + Backtest + Dàn thông minh (bản thực chiến)")

raw = st.text_area("📥 Dán kết quả 5 tỉnh", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2] + r[-1]) for r in rows if len(r) == 5]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Không đọc được dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ================= CORE =================
if len(df) >= 40:
    analysis = analyze_v4(df)

    # TOP CẶP
    best = analysis[0]
    rate = backtest(df, best["pair"])

    st.subheader("🏆 TOP CẶP MẠNH NHẤT")
    st.metric("Cặp đề xuất", best["pair"], f"{best['score']}%")
    st.progress(min(rate/50, 1.0))

    if rate >= 25:
        advice = "🟢 NÊN ĐÁNH"
        st.success(f"Backtest {rate}% – {advice}")
        update_ai(best["pair"], True)
    else:
        advice = "🟡 THEO DÕI"
        st.warning(f"Backtest {rate}% – {advice}")
        update_ai(best["pair"], False)

    log_prediction(best["pair"], best["score"], advice)

    # ================= DÀN 5 =================
    st.subheader("🎯 DÀN 5 SỐ (ĐÁNH CHÍNH)")
    dan5 = [x["pair"] for x in analysis[:5]]
    st.markdown(f"### 🔢 {dan5}")

    # ================= DÀN 3 TỈNH =================
    st.subheader("🎯 DÀN 3 TỈNH – DÀN 3 SỐ")
    dan3tinh = [x["pair"] for x in analysis[:3]]
    st.markdown(f"### 🔢 {dan3tinh}")

# ================= LOG =================
st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pair", "score", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))
