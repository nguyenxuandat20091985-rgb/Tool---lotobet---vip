import streamlit as st
import pandas as pd
from collections import Counter
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI v6.9 – 2 TINH",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== LOAD DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["numbers"] = df["numbers"].astype(str)
        return df
    return pd.DataFrame(columns=["time", "numbers"])

df = load_data()

# ================== UI ==================
st.title("🎯 NUMCORE AI v6.9")
st.caption("AI lọc rủi ro – Chuyên 2 tinh an toàn")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Nhập kết quả (5 số)")
    nums = st.text_input("Ví dụ: 30945")

    if st.button("Lưu"):
        if nums.isdigit() and len(nums) == 5:
            df = pd.concat([
                df,
                pd.DataFrame([{
                    "time": pd.Timestamp.now(),
                    "numbers": nums
                }])
            ], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("Đã lưu dữ liệu")
        else:
            st.error("Sai định dạng – phải đúng 5 số")

    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(25), use_container_width=True)

# ================== CORE AI ==================
def split_digits(series):
    digits = []
    for x in series:
        digits.extend(list(x))
    return digits

def score_numbers(df):
    recent = df.tail(30)
    digits = split_digits(recent["numbers"])
    freq = Counter(digits)
    total = sum(freq.values())

    scores = {}
    for d in map(str, range(10)):
        f = freq.get(d, 0) / total if total else 0
        penalty = 0.18 if f > 0.23 else 0  # tránh số quá nóng
        scores[d] = f - penalty
    return scores

def pick_safe_numbers(scores):
    return sorted(scores, key=scores.get, reverse=True)[:5]

def pair_ai(df, safe_nums):
    recent = df.tail(40)
    pairs = {}

    for i in range(len(safe_nums)):
        for j in range(i + 1, len(safe_nums)):
            a, b = safe_nums[i], safe_nums[j]
            count = 0
            for n in recent["numbers"]:
                if a in n and b in n:
                    count += 1
            pairs[a + b] = 1 / (count + 1)

    return sorted(pairs, key=pairs.get, reverse=True)

# ================== TAB 2 ==================
with tab2:
    if len(df) < 15:
        st.warning("Cần tối thiểu 15 kỳ dữ liệu")
    else:
        scores = score_numbers(df)
        safe_nums = pick_safe_numbers(scores)
        best_pairs = pair_ai(df, safe_nums)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        st.metric("Tổ hợp A", best_pairs[0])
        st.metric("Tổ hợp B", best_pairs[1])

        st.subheader("🧠 SỐ CHIẾN LƯỢC (ĐÁNH)")
        st.success(f"{best_pairs[0]}  –  {best_pairs[1]}")

        st.subheader("📊 Thống kê nhanh")
        stat = Counter(split_digits(df["numbers"]))
        stat_df = pd.DataFrame(stat.items(), columns=["Số", "Tần suất"])
        st.dataframe(stat_df.sort_values("Tần suất", ascending=False),
                     use_container_width=True)

st.caption("⚠️ AI hỗ trợ xác suất – không gấp thếp – không all-in")
