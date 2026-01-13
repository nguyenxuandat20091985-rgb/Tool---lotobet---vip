import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="LOTOBET AUTO PRO – V2", layout="centered")

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"

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

# ================= ANALYSIS =================
def analyze_v2(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_total = Counter(df["pair"])
    cnt_10 = Counter(last10)
    cnt_20 = Counter(last20)

    results = []
    for pair in cnt_total:
        p10 = cnt_10[pair] / 10
        p20 = cnt_20[pair] / 20
        p_all = cnt_total[pair] / total

        score = round((p10*0.5 + p20*0.3 + p_all*0.2)*100, 2)

        if cnt_10[pair] >= 3:
            status = "🔥 HOT"
            advice = "🟢 ĐÁNH MẠNH"
        elif cnt_10[pair] == 2:
            status = "🌤 WARM"
            advice = "🟡 ĐÁNH NHẸ"
        else:
            status = "❄️ COLD"
            advice = "🔴 BỎ – THEO DÕI"

        results.append({
            "pair": pair,
            "10_kỳ": cnt_10[pair],
            "20_kỳ": cnt_20[pair],
            "tổng": cnt_total[pair],
            "score_%": score,
            "trạng thái": status,
            "khuyến nghị": advice
        })

    return sorted(results, key=lambda x: x["score_%"], reverse=True)

# ================= UI =================
st.title("🟢 LOTOBET AUTO PRO – V2")

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
if len(df) >= 30:
    analysis = analyze_v2(df)

    st.subheader("🔥 TOP 5 CẶP ĐÁNG CHÚ Ý")
    st.table(analysis[:5])

    best = analysis[0]

    st.subheader("🚦 KHUYẾN NGHỊ KỲ TỚI")
    st.markdown(f"""
    **Cặp đề xuất:** `{best['pair']}`  
    **Score:** `{best['score_%']}%`  
    **Trạng thái:** {best['trạng thái']}  
    **Khuyến nghị:** {best['khuyến nghị']}
    """)

    if st.button("📌 LƯU DỰ ĐOÁN"):
        log_prediction(
            best["pair"],
            best["score_%"],
            best["khuyến nghị"],
            best["trạng thái"]
        )
        st.success("Đã lưu dự đoán")

# ================= LOG =================
st.subheader("🧾 LỊCH SỬ DỰ ĐOÁN")
log_df = load_csv(LOG_FILE, ["time", "pair", "score", "status", "advice"])
if not log_df.empty:
    st.table(log_df.tail(10))

# ================= RESET =================
if st.button("🗑 RESET TOÀN BỘ DỮ LIỆU"):
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    st.warning("Đã reset toàn bộ dữ liệu")
