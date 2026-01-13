import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

st.set_page_config(page_title="LOTOBET AUTO PRO – CẤP 1", layout="centered")

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"

# ---------- LOAD / SAVE ----------
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame(
        [{"time": now, "pair": p} for p in pairs]
    )
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def log_prediction(pairs, advice):
    df = load_csv(LOG_FILE, ["time", "pairs", "advice"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [now, ",".join(map(str, pairs)), advice]
    df.to_csv(LOG_FILE, index=False)

# ---------- ANALYSIS ----------
def analyze(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()

    counter_total = Counter(df["pair"])
    counter_10 = Counter(last10)

    results = []

    for pair, cnt10 in counter_10.items():
        p_recent = cnt10 / 10
        p_total = counter_total[pair] / total
        score = round((p_recent * 0.6 + p_total * 0.4) * 100, 2)

        results.append({
            "pair": pair,
            "10_kỳ": cnt10,
            "tổng": counter_total[pair],
            "score_%": score
        })

    return sorted(results, key=lambda x: x["score_%"], reverse=True)

# ---------- UI ----------
st.title("🟢 LOTOBET AUTO PRO – CẤP 1")

raw = st.text_area("📥 Dán kết quả 5 tinh", height=120)

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

if len(df) >= 20:
    analysis = analyze(df)

    st.subheader("📊 PHÂN TÍCH CẶP SỐ (TOP 5)")
    st.table(analysis[:5])

    best = analysis[0]
    advice = "🟢 NÊN ĐÁNH" if best["10_kỳ"] >= 3 else "🟡 CÂN NHẮC"

    st.subheader("🚦 KHUYẾN NGHỊ")
    st.markdown(f"""
    **Cặp đề xuất:** `{best['pair']}`  
    **Xác suất tương đối:** `{best['score_%']}%`  
    **Khuyến nghị:** {advice}
    """)

    if st.button("📌 LƯU DỰ ĐOÁN KỲ NÀY"):
        log_prediction([best["pair"]], advice)
        st.success("Đã lưu dự đoán")

st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pairs", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))
