import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI – 2 TINH",
    layout="centered"
)

DATA_FILE = "data_2tinh.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_number(num):
    df = load_data()
    df.loc[len(df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), num]
    df.to_csv(DATA_FILE, index=False)

# ================== CORE AI ==================
def extract_2tinh(df):
    all_nums = []
    for n in df["numbers"]:
        if len(str(n)) == 5:
            all_nums.append(str(n)[-2:])
    return all_nums

def ai_analyze(df):
    nums = extract_2tinh(df)

    if len(nums) < 20:
        return None, None, None

    # ---- Thuật toán 1: Tần suất ----
    freq = Counter(nums)

    # ---- Thuật toán 2: Gần đây (momentum) ----
    recent = nums[-20:]
    recent_freq = Counter(recent)

    # ---- Thuật toán 3: Gan ----
    last_seen = {}
    for i, n in enumerate(nums):
        last_seen[n] = i
    gan_score = {n: len(nums) - idx for n, idx in last_seen.items()}

    # ---- Chấm điểm tổng ----
    score = {}
    for n in freq:
        score[n] = (
            freq[n] * 1.0 +
            recent_freq.get(n, 0) * 1.5 +
            gan_score.get(n, 0) * 0.5
        )

    top = sorted(score.items(), key=lambda x: x[1], reverse=True)

    trung_tam_A = top[0][0]
    trung_tam_B = top[1][0]

    # ---- AI chốt 1 số đánh ----
    chien_luoc = top[0][0]

    return trung_tam_A, trung_tam_B, chien_luoc

# ================== UI ==================
st.title("🎯 NUMCORE AI – 2 TINH")
st.caption("Ưu tiên hiệu quả – Không nhiễu – Đánh được")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Nhập kết quả (5 số)")
    num = st.text_input("Ví dụ: 30945", max_chars=5)

    if st.button("Lưu"):
        if num.isdigit() and len(num) == 5:
            save_number(num)
            st.success("Đã lưu dữ liệu")
        else:
            st.error("Sai định dạng")

    df = load_data()
    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(20), use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    df = load_data()

    if len(df) < 20:
        st.warning("Cần tối thiểu 20 kỳ để AI phân tích chuẩn")
    else:
        A, B, CL = ai_analyze(df)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        col1, col2 = st.columns(2)
        col1.metric("Tổ hợp A", A)
        col2.metric("Tổ hợp B", B)

        st.subheader("🧠 SỐ CHIẾN LƯỢC (ĐÁNH)")
        st.success(f"{CL}")

        st.subheader("📊 Thống kê nhanh")
        two_digits = extract_2tinh(df)
        tk = Counter(two_digits).most_common(10)
        st.table(pd.DataFrame(tk, columns=["Số", "Số lần xuất hiện"]))
