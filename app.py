import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

st.set_page_config(page_title="LOTOBET AUTO – CẤP 1", layout="centered")

DATA_FILE = "data.csv"

# ---------- DATA ----------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "pair"])

def save_pairs(pairs):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame(
        [{"time": now, "pair": p} for p in pairs]
    )
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ---------- ANALYSIS ----------
def detect_signal(df):
    recent = df.tail(10)["pair"]
    counter = Counter(recent)

    strong = [k for k, v in counter.items() if v >= 3]

    if strong:
        return "🟢 NÊN ĐÁNH", strong

    if len(df) < 30:
        return "🟡 CHƯA ĐỦ DỮ LIỆU", []

    return "🔴 KHÔNG NÊN ĐÁNH", []

# ---------- UI ----------
st.title("🟢 LOTOBET AUTO – CẤP 1")

raw = st.text_area(
    "📥 Dán kết quả 5 tinh (vd: 71829 00384 55921)",
    height=120
)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2] + r[-1]) for r in rows if len(r) == 5]

    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Không nhận diện được dữ liệu")

df = load_data()
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) >= 10:
    signal, nums = detect_signal(df)
    st.subheader("🚦 TÍN HIỆU TỰ ĐỘNG")
    st.markdown(f"### {signal}")
    if nums:
        st.write("🎯 Cặp đang có cầu:", nums)

st.caption("Tool tự động cấp 1 – Không hack – Không đảm bảo trúng")
