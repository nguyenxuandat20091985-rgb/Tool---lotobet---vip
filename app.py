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

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_number(num):
    df = load_data()
    df.loc[len(df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), num]
    df.to_csv(DATA_FILE, index=False)

# ================== AI CORE ==================
def split_digits(df):
    all_digits = []
    for n in df["numbers"]:
        all_digits.extend(list(str(n)))
    return all_digits

def ai_core(df):
    digits = split_digits(df)
    freq = Counter(digits)

    # Lấy 6 số mạnh nhất
    top = freq.most_common(6)
    nums = [int(n[0]) for n in top]

    # Trung tâm A – B (2 cụm 2 số)
    center_a = f"{nums[0]}{nums[1]}"
    center_b = f"{nums[2]}{nums[3]}"

    # AI chiến lược: 2 số mạnh + ổn định
    strategy = sorted([nums[0], nums[2]])

    return center_a, center_b, strategy, freq

# ================== UI ==================
st.title("🎯 NUMCORE AI – 2 TINH")
st.caption("Ưu tiên hiệu quả – Không nhiễu – Đánh được")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Nhập kết quả (5 số)")

    if "input_num" not in st.session_state:
        st.session_state.input_num = ""

    num = st.text_input(
        "Ví dụ: 30945",
        max_chars=5,
        key="input_num"
    )

    if st.button("Lưu"):
        if num.isdigit() and len(num) == 5:
            save_number(num)
            st.session_state.input_num = ""
            st.success("Đã lưu 1 kỳ mới ✅")
            st.rerun()
        else:
            st.error("Sai định dạng – cần đúng 5 số")

    df = load_data()
    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(20), use_container_width=True)

# ================== TAB 2 ==================
with tab2:
    df = load_data()

    if len(df) < 20:
        st.warning("Cần ít nhất 20 kỳ để AI phân tích chính xác")
    else:
        center_a, center_b, strategy, freq = ai_core(df)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        col1, col2 = st.columns(2)
        col1.metric("Tổ hợp A", center_a)
        col2.metric("Tổ hợp B", center_b)

        st.subheader("🧠 SỐ CHIẾN LƯỢC (ĐÁNH)")
        st.success(f"{strategy[0]}  –  {strategy[1]}")

        st.subheader("📊 Thống kê nhanh")
        stat_df = pd.DataFrame(freq.most_common(), columns=["Số", "Tần suất"])
        st.dataframe(stat_df, use_container_width=True)
