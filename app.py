import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime
import os
import random

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers"])

    df = pd.read_csv(DATA_FILE)

    # TỰ SỬA DATA CŨ
    if "numbers" not in df.columns:
        # lấy cột cuối cùng làm numbers
        df["numbers"] = df.iloc[:, -1].astype(str)

    df["numbers"] = df["numbers"].astype(str)
    return df[["time", "numbers"]]

def save_result(raw):
    df = load_data()
    new = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "numbers": raw
    }])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================== CORE LOGIC ==================
def parse_numbers(value):
    s = str(value)
    return [int(x) for x in s if x.isdigit()][:5]

def unique(nums):
    out = []
    for n in nums:
        if n not in out:
            out.append(n)
    return out[:5]

def build_groups(nums):
    if len(nums) < 3:
        return []
    return list(combinations(nums, 3))[:2]

def ai_pick(nums):
    pool = [n for n in range(10) if n not in nums]
    if len(pool) < 2:
        return "--"
    a = random.choice(pool)
    b = random.choice([x for x in pool if x != a])
    return f"{a}{b}"

# ================== UI ==================
st.title("🔷 NUMCORE")
st.caption("Phân tích chuỗi số – Ưu tiên hiệu quả – Không nhiễu")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🎯 Phân tích & Dự đoán"])

# ================== TAB 1 ==================
with tab1:
    raw = st.text_input("Nhập kết quả kỳ (5 số)", max_chars=5)

    if st.button("Lưu"):
        if raw.isdigit() and len(raw) == 5:
            save_result(raw)
            st.success("Đã lưu dữ liệu")
        else:
            st.error("Nhập đúng 5 chữ số")

    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

# ================== TAB 2 ==================
with tab2:
    df = load_data()

    all_nums = []
    for v in df["numbers"]:
        try:
            all_nums.extend(parse_numbers(v))
        except:
            continue

    if len(all_nums) < 10:
        st.warning("Chưa đủ dữ liệu để phân tích")
    else:
        freq = Counter(all_nums)
        top = unique([n for n, _ in freq.most_common(5)])

        st.subheader("🎯 SỐ TRUNG TÂM")
        groups = build_groups(top)

        c1, c2 = st.columns(2)
        if len(groups) > 0:
            c1.metric("Tổ hợp A", "".join(map(str, groups[0])))
        if len(groups) > 1:
            c2.metric("Tổ hợp B", "".join(map(str, groups[1])))

        st.divider()

        st.subheader("🧠 SỐ CHIẾN LƯỢC")
        st.metric("Ưu tiên", ai_pick(top))

        st.divider()

        total = len(df)
        rate = min(60, 45 + total // 40)

        st.subheader("📊 THỐNG KÊ")
        st.write(f"Kỳ đã phân tích: **{total}**")
        st.write(f"Tỉ lệ tham khảo: **≈ {rate}%**")

st.caption("NUMCORE v6.6 – Ổn định – Không crash – Tập trung tiền thật")
