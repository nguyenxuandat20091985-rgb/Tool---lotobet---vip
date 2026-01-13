import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – FINAL",
    page_icon="🎯",
    layout="centered"
)

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
AI_FILE = "ai_weight.csv"

# ==================================================
# CORE UTILITIES
# ==================================================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{"time": now, "pair": p} for p in pairs])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def log_prediction(pair, score, rate, advice):
    df = load_csv(LOG_FILE, ["time", "pair", "score", "rate", "advice"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [now, pair, score, rate, advice]
    df.to_csv(LOG_FILE, index=False)

# ==================================================
# AI WEIGHT
# ==================================================
def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, win):
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.2 if win else -0.1
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    ai.to_csv(AI_FILE, index=False)

# ==================================================
# ANALYSIS ENGINE
# ==================================================
def analyze_pairs(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    ai = load_ai()
    ai_map = dict(zip(ai["pair"], ai["weight"]))

    results = []
    for pair in cnt_all:
        base = (
            cnt10[pair]/10 * 0.5 +
            cnt20[pair]/20 * 0.3 +
            cnt_all[pair]/total * 0.2
        )
        score = round(base * ai_map.get(pair, 1.0) * 100, 2)

        if cnt10[pair] >= 3:
            level = "🔥 HOT"
            advice = "ĐÁNH CHÍNH"
        elif cnt10[pair] == 2:
            level = "🌤 ỔN ĐỊNH"
            advice = "ĐÁNH PHỤ"
        elif cnt20[pair] >= 2:
            level = "🎯 BÙNG LẠI"
            advice = "GÀI NHẸ"
        else:
            level = "❄️ COLD"
            advice = "BỎ"

        results.append({
            "pair": pair,
            "10k": cnt10[pair],
            "20k": cnt20[pair],
            "score": score,
            "level": level,
            "advice": advice
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

def backtest(df, pair, lookback=30):
    hits = 0
    total = min(lookback, len(df)-1)
    for i in range(total):
        if df.iloc[-(i+2)]["pair"] == pair:
            hits += 1
    rate = round(hits/total*100, 2) if total else 0
    return rate

def analyze_digits(df):
    digits = []
    for p in df["pair"]:
        digits.extend([p//10, p%10])

    last10 = df.tail(10)["pair"]
    last20 = df.tail(20)["pair"]

    def ext(ps):
        out = []
        for p in ps:
            out.extend([p//10, p%10])
        return out

    cnt_all = Counter(digits)
    cnt10 = Counter(ext(last10))
    cnt20 = Counter(ext(last20))

    scores = {}
    for d in range(10):
        s = (
            cnt10[d]/(len(last10)*2) * 0.5 +
            cnt20[d]/(len(last20)*2) * 0.3 +
            cnt_all[d]/len(digits) * 0.2
        )
        scores[d] = round(s*100, 2)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ==================================================
# UI
# ==================================================
st.markdown("<h2 style='text-align:center;color:#00ff99'>🎯 LOTOBET AUTO PRO – FINAL</h2>", unsafe_allow_html=True)

raw = st.text_area("📥 Dán kết quả 5 tỉnh", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2]+r[-1]) for r in rows if len(r)==5]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("❌ Không đọc được dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ==================================================
# DASHBOARD
# ==================================================
if len(df) >= 40:
    analysis = analyze_pairs(df)
    top = analysis[0]
    rate = backtest(df, top["pair"])

    st.subheader("🏆 CẶP AI MẠNH NHẤT")
    st.success(
        f"🎯 **{top['pair']}** | Score: **{top['score']}%** | Backtest: **{rate}%**\n\n"
        f"👉 Khuyến nghị: **{top['advice']}**"
    )

    if st.button("📌 LƯU & AI HỌC"):
        log_prediction(top["pair"], top["score"], rate, top["advice"])
        update_ai(top["pair"], win=(rate >= 25))
        st.success("AI đã cập nhật")

    st.subheader("🔥 TOP 5 CẶP NÊN ĐÁNH")
    st.table(pd.DataFrame(analysis[:5]))

    # ================= DÀN 5 =================
    st.subheader("🎯 DÀN 5 SỐ TINH NHẤT")
    dan5 = [x["pair"] for x in analysis[:5]]
    st.success("👉 " + " – ".join(map(str, dan5)))

    # ================= DIGIT =================
    st.subheader("🔢 3 CHỮ SỐ MẠNH (5 TỈNH)")
    digits = analyze_digits(df)[:3]
    st.info(", ".join(f"{d} ({s}%)" for d, s in digits))

    # ================= 5 SỐ KHÔNG CỐ ĐỊNH =================
    st.subheader("🚀 5 SỐ XÁC SUẤT CAO NHẤT")
    selected = []
    for d, _ in digits:
        for x in analysis[:10]:
            if str(d) in str(x["pair"]) and x["pair"] not in selected:
                selected.append(x["pair"])
            if len(selected) == 5:
                break
        if len(selected) == 5:
            break

    st.success("🎯 " + " – ".join(map(str, selected)))

# ==================================================
# LOG
# ==================================================
st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pair", "score", "rate", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))
