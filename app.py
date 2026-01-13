import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

st.set_page_config(page_title="LOTOBET AUTO PRO – CẤP 1 (V9.2)", layout="centered")

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"

# ---------- LOAD / INIT ----------
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return df

# ---------- SAVE DATA ----------
def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in pairs:
        df.loc[len(df)] = [now, p]
    df.to_csv(DATA_FILE, index=False)

def log_prediction(pair, advice, score):
    df = load_csv(LOG_FILE, ["time", "pair", "score", "advice"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [now, pair, score, advice]
    df.to_csv(LOG_FILE, index=False)

# ---------- ANALYSIS ----------
def analyze(df):
    total = len(df)
    counter = Counter(df["pair"])
    results = []

    for pair, cnt in counter.items():
        # gap
        last_idx = df[df["pair"] == pair].index
        gap = total - 1 - last_idx[-1] if len(last_idx) > 0 else total

        freq = cnt / total
        score = round((freq * 100) - gap * 1.5, 2)

        results.append({
            "pair": pair,
            "số_lần": cnt,
            "gap": gap,
            "tần_suất_%": round(freq * 100, 2),
            "score_%": score
        })

    results = [r for r in results if r["score_%"] > 0]
    return sorted(results, key=lambda x: x["score_%"], reverse=True)

# ---------- UI ----------
st.title("🤖 LOTOBET AUTO PRO – CẤP 1 (V9.2)")
st.caption("Phân tích an toàn • Không ép đánh • Có quyền nghỉ")

raw = st.text_area("📥 Dán kết quả 5 tinh (VD: 57221)", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2] + r[-1]) for r in rows if len(r) == 5]

    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
        st.rerun()
    else:
        st.error("❌ Cần đúng định dạng 5 chữ số")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ---------- RESULT ----------
if len(df) >= 20:
    analysis = analyze(df)

    if analysis:
        st.subheader("📊 TOP 5 CẶP TIỀM NĂNG")
        st.table(analysis[:5])

        best = analysis[0]

        if best["gap"] <= 1:
            advice = "🛑 CẦU NÓNG – NÊN NGHỈ"
        elif best["score_%"] < 5:
            advice = "🟡 KHÔNG RÕ RÀNG"
        else:
            advice = "🟢 CÓ THỂ ĐÁNH NHỎ"

        st.subheader("🚦 KHUYẾN NGHỊ")
        st.markdown(f"""
        **Cặp đề xuất:** `{best['pair']}`  
        **Score:** `{best['score_%']}%`  
        **Gap:** `{best['gap']}`  
        **Khuyến nghị:** **{advice}**
        """)

        if st.button("📌 LƯU DỰ ĐOÁN"):
            log_prediction(best["pair"], advice, best["score_%"])
            st.success("Đã lưu dự đoán")

# ---------- LOG ----------
st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pair", "score", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))
