import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

st.set_page_config(page_title="NUMCORE AI v6.6 – 2 TÍNH", layout="centered")

DATA_FILE = "data.csv"

# ================= DATA =================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["numbers"] = df["numbers"].astype(str)
        df = df[df["numbers"].str.match(r"^\d{5}$", na=False)]
        return df
    return pd.DataFrame(columns=["time", "numbers"])

def save_data(rows):
    df = load_data()
    df = pd.concat([df, rows], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================= ANALYSIS =================
def split_digits(data):
    digits = []
    for n in data:
        if isinstance(n, str) and len(n) == 5 and n.isdigit():
            digits.extend(list(n))
    return digits

def analyze(data):
    digits = split_digits(data)
    if len(digits) < 50:
        return None

    freq = Counter(digits)

    max_f = max(freq.values())
    hot = [d for d, c in freq.items() if c >= max_f * 0.75]
    cold = [d for d in "0123456789" if d not in freq]

    last_seen = {}
    for i, n in enumerate(reversed(data)):
        for d in n:
            if d not in last_seen:
                last_seen[d] = i + 1

    bias = [d for d, v in last_seen.items() if v <= 3]

    core = list(set(hot) & set(bias))
    core = core[:2]

    confidence = len(core)

    return {
        "freq": freq,
        "hot": hot,
        "cold": cold,
        "bias": bias,
        "core": core,
        "confidence": confidence
    }

# ================= UI =================
st.title("NUMCORE AI v6.6 – 2 TÍNH")
st.caption("Ưu tiên an toàn – Không ảo – Không gỡ")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

with tab1:
    st.subheader("Nhập kết quả (5 số)")
    raw = st.text_area("Nhập nhiều kỳ – mỗi dòng 1 kết quả", height=160)

    if st.button("Lưu dữ liệu"):
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if line.isdigit() and len(line) == 5:
                rows.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "numbers": line
                })

        if rows:
            save_data(pd.DataFrame(rows))
            st.success(f"Đã lưu {len(rows)} kỳ hợp lệ")
        else:
            st.error("Không có dòng nào đúng chuẩn 5 số")

    df = load_data()
    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(30), use_container_width=True)

with tab2:
    df = load_data()

    if len(df) < 30:
        st.error("⚠️ Dữ liệu quá ít – KHÔNG ĐÁNH")
    else:
        result = analyze(df["numbers"].tolist())

        if result is None:
            st.error("⚠️ Cầu nhiễu – KHÔNG ĐÁNH")
        else:
            st.subheader("🎯 SỐ TRUNG TÂM (AI)")
            if result["core"]:
                if len(result["core"]) >= 1:
                    st.metric("Tổ hợp A", result["core"][0])
                if len(result["core"]) >= 2:
                    st.metric("Tổ hợp B", result["core"][1])
            else:
                st.warning("Không có số trung tâm đủ tin cậy")

            st.subheader("🧠 SỐ CHIẾN LƯỢC")
            if result["confidence"] >= 2:
                st.success("Chỉ nên quan sát – không all-in")
                st.write("Nhóm số ưu tiên:", result["core"])
            else:
                st.error("KHÔNG ĐÁNH – cầu yếu")

            st.subheader("📊 Thống kê nhanh")
            freq_df = pd.DataFrame(
                result["freq"].items(), columns=["Số", "Tần suất"]
            ).sort_values("Tần suất", ascending=False)
            st.dataframe(freq_df, use_container_width=True)

st.caption("⚠️ Tool phân tích xác suất – KHÔNG cam kết lợi nhuận.")
