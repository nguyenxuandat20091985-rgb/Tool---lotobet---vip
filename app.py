import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from collections import Counter
import plotly.express as px

# ================== CONFIG ==================
st.set_page_config(page_title="LOTObet V9", layout="wide")
DATA_FILE = "lotobet_data.csv"
PREDICT_FILE = "predict_history.csv"

# ================== INIT ==================
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["time", "result"]).to_csv(DATA_FILE, index=False)

if not os.path.exists(PREDICT_FILE):
    pd.DataFrame(columns=["time", "pairs", "advice"]).to_csv(PREDICT_FILE, index=False)

# ================== CORE ANALYSIS ==================
def analyze_digits(df):
    total = len(df)
    rows = []

    for d in range(10):
        digit = str(d)
        idx = df[df["result"].str.contains(digit)].index.tolist()

        if not idx:
            gap = total
            streak = 0
        else:
            gap = total - 1 - idx[-1]
            streak = 1
            for i in range(len(idx)-1, 0, -1):
                if idx[i] - idx[i-1] == 1:
                    streak += 1
                else:
                    break

        score = max(0, 100 - gap * 10 - streak * 8)

        rows.append({
            "Số": d,
            "Gap": gap,
            "Bệt": streak,
            "Điểm": round(score, 2),
            "Cảnh báo": "⚠️ Bệt sâu" if streak >= 3 else ""
        })

    return pd.DataFrame(rows).sort_values("Điểm", ascending=False)

def analyze_pairs(df):
    pairs = []
    for r in df["result"]:
        u = list(set(r))
        for i in range(len(u)):
            for j in range(i+1, len(u)):
                pairs.append("".join(sorted([u[i], u[j]])))

    c = Counter(pairs)
    total = sum(c.values())

    data = []
    for k, v in c.most_common(10):
        data.append({
            "Cặp": k,
            "Số lần": v,
            "Tỷ lệ %": round(v / total * 100, 2)
        })

    return pd.DataFrame(data)

def assistant_advice(df, ana):
    if len(df) < 20:
        return "🛑 Dữ liệu ít – KHÔNG NÊN ĐÁNH"

    if ana["Bệt"].max() >= 4:
        return "⚠️ Cầu bệt sâu – NÊN NGHỈ, tránh đuổi"

    return "✅ Cầu ổn – Đánh nhỏ, 1 tay"

# ================== UI ==================
st.title("🤖 LOTOBET V9 – TRỢ LÝ KIẾM TIỀN AN TOÀN")
st.caption("Không đuổi cầu • Không all-in • Ưu tiên sống sót")

col1, col2 = st.columns([1, 2])

# ================== INPUT ==================
with col1:
    st.subheader("📥 Nhập kết quả 5 tinh")
    raw = st.text_input("Ví dụ: 57221")
    raw = re.sub(r"\D", "", raw)

    if st.button("💾 Lưu kỳ"):
        if len(raw) == 5:
            pd.DataFrame([{
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": raw
            }]).to_csv(DATA_FILE, mode="a", header=False, index=False)
            st.success(f"Đã lưu: {raw}")
            st.rerun()
        else:
            st.error("❌ Cần đúng 5 chữ số")

    st.info("🎯 Luật: Chọn 2 số – xuất hiện trong 5 tinh là thắng")

# ================== ANALYSIS ==================
with col2:
    df = pd.read_csv(DATA_FILE)

    if df.empty:
        st.warning("Chưa có dữ liệu")
    else:
        ana = analyze_digits(df)
        pair_df = analyze_pairs(df)
        advice = assistant_advice(df, ana)

        st.subheader("📊 Phân tích số (Cầu bệt – Gap)")
        fig = px.bar(
            ana,
            x="Số",
            y="Điểm",
            color="Bệt",
            text="Cảnh báo",
            color_continuous_scale="Turbo"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧮 Cặp 2 số tiềm năng")
        st.dataframe(pair_df, use_container_width=True)

        # GỢI Ý
        top = ana[ana["Bệt"] < 3].head(4)["Số"].astype(str).tolist()
        if len(top) >= 4:
            pairs = [top[0]+top[1], top[2]+top[3]]
        else:
            pairs = []

        st.success(f"🎯 Gợi ý: {pairs if pairs else 'KHÔNG CHỐT'}")
        st.warning(f"🤖 Trợ lý: {advice}")

        pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pairs": ", ".join(pairs),
            "advice": advice
        }]).to_csv(PREDICT_FILE, mode="a", header=False, index=False)

# ================== HISTORY ==================
st.subheader("🕒 10 kỳ gần nhất")
if not df.empty:
    st.dataframe(df.tail(10), use_container_width=True)

st.subheader("📌 Nhật ký trợ lý")
pred = pd.read_csv(PREDICT_FILE)
if not pred.empty:
    st.dataframe(pred.tail(10), use_container_width=True)
