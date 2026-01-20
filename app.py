import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI v6.6 – 2 TÍNH",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== INIT ==================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "numbers"])

# ================== UTILS ==================
def save_data():
    st.session_state.data.to_csv(DATA_FILE, index=False)

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def clean_number(n):
    n = str(n)
    return "".join([c for c in n if c.isdigit()])

def split_digits(series):
    digits = []
    for n in series:
        n = clean_number(n)
        if len(n) >= 5:
            digits.extend(list(n))
    return digits

# ================== AI CORE ==================
def analyze(df):
    digits = split_digits(df["numbers"])

    if len(digits) < 20:
        return None

    freq = Counter(digits)

    # Trung tâm A – B
    top = freq.most_common(3)
    A = top[0][0]
    B = top[1][0]

    # Độ lệch
    diff = abs(freq[A] - freq[B])

    # Chiến lược
    if diff <= 2:
        strategy = "ĐÁNH NHẸ – 2 số"
        risk = "THẤP"
        bet = f"{A}{B} – {B}{A}"
    elif diff <= 5:
        strategy = "ĐÁNH THĂM DÒ"
        risk = "TRUNG BÌNH"
        bet = f"{A}{B}"
    else:
        strategy = "CHỈ QUAN SÁT"
        risk = "CAO"
        bet = "KHÔNG NÊN VÀO"

    return {
        "A": A,
        "B": B,
        "bet": bet,
        "strategy": strategy,
        "risk": risk,
        "freq": freq
    }

# ================== LOAD DATA ==================
st.session_state.data = load_data()

# ================== UI ==================
st.title("🎯 NUMCORE AI v6.6 – 2 TÍNH")
st.caption("Ưu tiên an toàn – Không ảo – Không gỡ liều")

tabs = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ================== TAB 1 ==================
with tabs[0]:
    st.subheader("Nhập kết quả (5 số)")
    num = st.text_input("Ví dụ: 30945")

    if st.button("Lưu"):
        num = clean_number(num)
        if len(num) == 5:
            new_row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "numbers": num
            }
            st.session_state.data = pd.concat(
                [st.session_state.data, pd.DataFrame([new_row])],
                ignore_index=True
            )
            save_data()
            st.success("Đã lưu dữ liệu")
        else:
            st.error("Phải nhập đúng 5 số")

    st.divider()
    st.subheader("Dữ liệu đã nhập")
    st.dataframe(st.session_state.data.tail(20), use_container_width=True)

# ================== TAB 2 ==================
with tabs[1]:
    st.subheader("🎯 SỐ TRUNG TÂM (AI)")

    result = analyze(st.session_state.data)

    if result is None:
        st.warning("Chưa đủ dữ liệu để phân tích (tối thiểu ~20 chữ số)")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tổ hợp A", result["A"])
        with col2:
            st.metric("Tổ hợp B", result["B"])

        st.divider()
        st.subheader("🧠 SỐ CHIẾN LƯỢC")

        if result["risk"] == "CAO":
            st.error("🚫 KHÔNG NÊN VÀO – Cầu xấu")
        elif result["risk"] == "TRUNG BÌNH":
            st.warning("⚠️ ĐÁNH THĂM DÒ – Tiền nhỏ")
        else:
            st.success("✅ ĐÁNH ĐƯỢC – Ưu tiên an toàn")

        st.markdown(f"""
**Chiến lược:** {result["strategy"]}  
**Số đề xuất:** `{result["bet"]}`  
**Mức rủi ro:** **{result["risk"]}**
""")

        st.divider()
        st.subheader("📊 Thống kê nhanh")
        freq_df = pd.DataFrame(
            result["freq"].most_common(),
            columns=["Số", "Tần suất"]
        )
        st.dataframe(freq_df, use_container_width=True)
