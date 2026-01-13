import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="LOTOBET AUTO PRO – V3", layout="centered")

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
AI_FILE = "ai_weight.csv"

# ================= LOAD / SAVE =================
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

def log_prediction(pair, score, advice, status):
    df = load_csv(LOG_FILE, ["time", "pair", "score", "status", "advice"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [now, pair, score, status, advice]
    df.to_csv(LOG_FILE, index=False)

# ================= AI LEARNING =================
def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, win=True):
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.2 if win else -0.1
    ai.loc[idx, "weight"] = max(0.1, ai.loc[idx, "weight"])
    ai.to_csv(AI_FILE, index=False)

# ================= ANALYSIS =================
def analyze_v3(df):
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
        base = (cnt10[pair]/10)*0.5 + (cnt20[pair]/20)*0.3 + (cnt_all[pair]/total)*0.2
        weight = ai_map.get(pair, 1.0)
        score = round(base * weight * 100, 2)

        if cnt10[pair] >= 3:
            status = "🔥 HOT"
            advice = "🟢 ĐÁNH MẠNH"
        elif cnt10[pair] == 2:
            status = "🌤 WARM"
            advice = "🟡 ĐÁNH NHẸ"
        else:
            status = "❄️ COLD"
            advice = "🔴 BỎ"

        results.append({
            "pair": pair,
            "10k": cnt10[pair],
            "20k": cnt20[pair],
            "score": score,
            "status": status,
            "advice": advice
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ================= BACKTEST =================
def backtest(df, test_pair, lookback=30):
    hits = 0
    total = min(lookback, len(df)-1)
    for i in range(total):
        if df.iloc[-(i+2)]["pair"] == test_pair:
            hits += 1
    rate = round(hits/total*100, 2) if total else 0
    return hits, rate

# ================= UI =================
st.title("🟢 LOTOBET AUTO PRO – V3")

raw = st.text_area("📥 Dán kết quả 5 tỉnh", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2]+r[-1]) for r in rows if len(r)==5]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Không nhận diện được dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ================= ANALYZE =================
if len(df) >= 40:
    analysis = analyze_v3(df)
    st.subheader("🔥 TOP 5 CẶP AI ĐỀ XUẤT")
    st.table(analysis[:5])

    best = analysis[0]
    hits, rate = backtest(df, best["pair"])

    st.subheader("🚦 KẾT LUẬN AI")
    st.markdown(f"""
    **Cặp:** `{best['pair']}`  
    **Score AI:** `{best['score']}%`  
    **Backtest trúng:** `{rate}%`  
    **Trạng thái:** {best['status']}  
    **Khuyến nghị:** {best['advice']}
    """)

    if rate >= 25:
        st.success("✅ Đủ điều kiện xuống tiền")
    else:
        st.warning("⚠️ Độ tin cậy thấp – nên theo dõi")

    if st.button("📌 LƯU & HỌC AI"):
        log_prediction(best["pair"], best["score"], best["advice"], best["status"])
        update_ai(best["pair"], win=(rate >= 25))
        st.success("AI đã học xong kỳ này")

# ================= DÀN =================
st.subheader("🎯 DÀN THÔNG MINH")
if len(df) >= 40:
    st.write("Dàn 1:", [x["pair"] for x in analysis[:1]])
    st.write("Dàn 3:", [x["pair"] for x in analysis[:3]])
    st.write("Dàn 5:", [x["pair"] for x in analysis[:5]])

# ================= LOG =================
st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pair", "score", "status", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))
