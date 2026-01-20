import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os
import re

# ================= CONFIG =================
st.set_page_config(page_title="NUMCORE AI v6.6", layout="centered")
DATA_FILE = "data.csv"

# ================= DATA =================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["numbers"] = df["numbers"].astype(str)
        return df
    return pd.DataFrame(columns=["time", "numbers"])

def save_numbers(nums):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"time": now, "numbers": n} for n in nums]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def clean_numbers(df):
    return [
        n for n in df["numbers"]
        if isinstance(n, str) and n.isdigit() and len(n) == 5
    ]

# ================= AI CORE =================
def ai_center_numbers(df):
    valid = clean_numbers(df)

    if len(valid) < 10:
        return None

    all_digits = []
    recent_digits = []

    for n in valid:
        all_digits.extend(list(n))

    for n in valid[-15:]:
        recent_digits.extend(list(n))

    freq_all = Counter(all_digits)
    freq_recent = Counter(recent_digits)

    score = {}
    for d in "0123456789":
        score[d] = freq_all.get(d, 0) * 0.4 + freq_recent.get(d, 0) * 0.6

    # phá lặp 2 kỳ cuối
    last_overlap = set(valid[-1]) & set(valid[-2])
    for d in last_overlap:
        score[d] *= 0.3

    top = sorted(score, key=score.get, reverse=True)[:5]
    return top

def ai_strategy(center):
    return Counter(center).most_common(1)[0][0]

# ================= UI =================
st.title("🧠 NUMCORE AI v6.6")
st.caption("Phân tích chuỗi số – Ưu tiên hiệu quả – Không nhiễu")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🎯 Phân tích & Dự đoán"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Nhập nhiều kỳ (mỗi dòng 5 số)")
    raw = st.text_area("Ví dụ:\n12345\n67890\n90876")

    if st.button("💾 Lưu"):
        lines = raw.splitlines()
        valid = [l.strip() for l in lines if re.fullmatch(r"\d{5}", l.strip())]
        if valid:
            save_numbers(valid)
            st.success(f"Đã lưu {len(valid)} kỳ")
        else:
            st.warning("Không có dữ liệu hợp lệ")

    df = load_data()
    st.markdown(f"📊 **Tổng kỳ hợp lệ:** {len(clean_numbers(df))}")
    st.dataframe(df.tail(20), use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    df = load_data()
    valid = clean_numbers(df)

    if len(valid) < 10:
        st.warning("Chưa đủ dữ liệu sạch để AI phân tích")
    else:
        center = ai_center_numbers(df)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        c1, c2 = st.columns(2)
        c1.metric("Tổ hợp A", "".join(center[:3]))
        c2.metric("Tổ hợp B", "".join(center[2:]))

        st.divider()

        st.subheader("🧠 SỐ CHIẾN LƯỢC")
        st.metric("AI chọn lọc", ai_strategy(center))

        st.divider()

        st.subheader("📈 Thống kê nhanh")
        freq = Counter("".join(valid))
        stat = pd.DataFrame(freq.items(), columns=["Số", "Tần suất"]).sort_values(
            "Tần suất", ascending=False
        )
        st.dataframe(stat, use_container_width=True)

st.caption("⚠ Công cụ phân tích xác suất – không cam kết trúng")
