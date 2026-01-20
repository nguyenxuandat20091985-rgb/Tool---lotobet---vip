import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI v6.6 – 2 TINH",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_number(num):
    df = load_data()
    df.loc[len(df)] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "numbers": str(num)
    }
    df.to_csv(DATA_FILE, index=False)

# ================== AI CORE ==================
def split_digits(df):
    digits = []
    for n in df["numbers"]:
        s = str(n).zfill(5)
        digits.extend(list(s))
    return digits

def ai_frequency(digits):
    c = Counter(digits)
    return sorted(c.items(), key=lambda x: x[1], reverse=True)

def ai_cycle_filter(df, digit, lookback=25):
    recent = df.tail(lookback)["numbers"].astype(str)
    count = sum(digit in x for x in recent)
    return count <= lookback * 0.4  # không quá nóng

def ai_select_centers(df):
    digits = split_digits(df)
    freq = ai_frequency(digits)

    stable = []
    for d, _ in freq:
        if ai_cycle_filter(df, d):
            stable.append(d)
        if len(stable) >= 6:
            break

    A = stable[:2]
    B = stable[2:4]
    return A, B

def ai_strategy(A, B):
    # chọn 2 số có khoảng cách & khác nhóm
    if len(A) >= 2:
        return [A[0], B[0]] if len(B) > 0 else A[:2]
    return A + B

# ================== UI ==================
st.title("🎯 NUMCORE AI v6.6 – 2 TINH")
st.caption("Ưu tiên hiệu quả – Không nhiễu – Đánh được")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Nhập kết quả (5 số)")
    nums = st.text_area(
        "Nhập nhiều kỳ – mỗi dòng 1 kết quả",
        placeholder="Ví dụ:\n30945\n69763\n91573",
        height=150
    )

    if st.button("Lưu dữ liệu"):
        lines = [x.strip() for x in nums.splitlines() if x.strip().isdigit()]
        for l in lines:
            if len(l) == 5:
                save_number(l)
        st.success(f"Đã lưu {len(lines)} kỳ")

    df = load_data()
    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(20), use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    df = load_data()

    if len(df) < 10:
        st.warning("Cần ít nhất 10 kỳ để AI phân tích chuẩn")
    else:
        A, B = ai_select_centers(df)
        strategy = ai_strategy(A, B)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tổ hợp A", "".join(A))
        with col2:
            st.metric("Tổ hợp B", "".join(B))

        st.subheader("🧠 SỐ CHIẾN LƯỢC (ĐÁNH)")
        st.success(" – ".join(strategy))

        st.subheader("📊 Thống kê nhanh")
        digits = split_digits(df)
        freq = Counter(digits)
        stat_df = pd.DataFrame(freq.items(), columns=["Số", "Tần suất"]).sort_values(
            "Tần suất", ascending=False
        )
        st.dataframe(stat_df, use_container_width=True)
