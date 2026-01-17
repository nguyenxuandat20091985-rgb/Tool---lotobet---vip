import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTObet AI v1.0",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA UTILS ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["number"])

def save_data(numbers):
    df = load_data()
    new_df = pd.DataFrame({"number": numbers})
    df = pd.concat([df, new_df]).drop_duplicates().reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    return df

def clean_input(text):
    raw = text.replace("\n", " ").split(" ")
    nums = []
    for x in raw:
        x = x.strip()
        if x.isdigit() and len(x) == 5:
            nums.append(x)
    return nums

# ================== ANALYSIS CORE ==================
def analyze_numbers(df):
    """
    Phân tích theo LOGIC:
    - 1 số (0–9) xuất hiện ở BẤT KỲ vị trí nào trong 5 số
    """
    total_draws = len(df)
    if total_draws == 0:
        return None

    appear_count = Counter()

    for value in df["number"]:
        unique_digits = set(value)
        for d in unique_digits:
            appear_count[d] += 1

    results = []
    for d in range(10):
        count = appear_count.get(str(d), 0)
        percent = round((count / total_draws) * 100, 2)
        results.append({
            "Số": d,
            "% Xuất hiện trong 5 số": percent,
            "Khuyến nghị": "NÊN ĐÁNH" if percent >= 50 else "KHÔNG NÊN"
        })

    return pd.DataFrame(results).sort_values(
        by="% Xuất hiện trong 5 số",
        ascending=False
    )

# ================== UI ==================
st.title("🧠 LOTOBET AI v1.0")
st.caption("Dự đoán 1 con số có khả năng xuất hiện trong giải đặc biệt (5 số)")

tabs = st.tabs([
    "📥 Thu thập dữ liệu",
    "⚡ Phân tích nhanh",
    "📊 Phân tích số"
])

# ---------- TAB 1 ----------
with tabs[0]:
    st.subheader("📥 Nhập & nạp dữ liệu")

    input_text = st.text_area(
        "Nhập số (5 chữ số, không cần cách nhau):",
        placeholder="12345 54321\n56789\n98765",
        height=150
    )

    if st.button("➕ Nạp dữ liệu"):
        numbers = clean_input(input_text)
        if numbers:
            df = save_data(numbers)
            st.success(f"Đã nạp {len(numbers)} số hợp lệ")
            st.write("Tổng dữ liệu:", len(df))
        else:
            st.error("Không có số hợp lệ (phải đủ 5 chữ số)")

    st.divider()

    uploaded = st.file_uploader("Import TXT / CSV", type=["txt", "csv"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            if "number" in df.columns:
                save_data(df["number"].astype(str).tolist())
                st.success("Import CSV thành công")
        else:
            content = uploaded.read().decode("utf-8")
            nums = clean_input(content)
            save_data(nums)
            st.success("Import TXT thành công")

# ---------- TAB 2 ----------
with tabs[1]:
    st.subheader("⚡ Phân tích nhanh")

    df = load_data()
    result = analyze_numbers(df)

    if result is not None:
        best = result.iloc[0]
        st.metric(
            label="🎯 SỐ ĐỀ XUẤT",
            value=int(best["Số"]),
            delta=f'{best["% Xuất hiện trong 5 số"]}% khả năng xuất hiện'
        )
    else:
        st.info("Chưa có dữ liệu để phân tích")

# ---------- TAB 3 ----------
with tabs[2]:
    st.subheader("📊 Phân tích chi tiết từng số")

    df = load_data()
    result = analyze_numbers(df)

    if result is not None:
        st.dataframe(result, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu")
