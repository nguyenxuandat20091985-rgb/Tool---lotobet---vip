import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os
import re

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI v6.6",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_numbers(nums):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = [{"time": now, "numbers": n} for n in nums]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================== AI CORE ==================
def ai_center_numbers(df):
    valid = []

    for n in df["numbers"]:
        if isinstance(n, str) and n.isdigit() and len(n) == 5:
            valid.append(n)

    if len(valid) < 10:
        return ["?", "?", "?", "?", "?"]

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
        score[d] = freq_all.get(d, 0) * 0.3 + freq_recent.get(d, 0) * 0.4

    # phá bệt
    if len(valid) >= 2:
        bad = set(valid[-1]) & set(valid[-2])
        for b in bad:
            score[b] *= 0.3

    top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:5]
    return [x[0] for x in top]

def ai_strategy_number(center):
    c = Counter(center)
    return c.most_common(1)[0][0]

# ================== UI ==================
st.title("🧠 NUMCORE AI v6.6")
st.caption("Phân tích chuỗi số – Ưu tiên hiệu quả – Không nhiễu")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🎯 Phân tích & Dự đoán"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Nhập kết quả (mỗi dòng 5 số)")
    raw = st.text_area("Ví dụ:\n12345\n67890\n90876")

    if st.button("💾 Lưu dữ liệu"):
        lines = raw.splitlines()
        valid = []
        invalid = 0

        for l in lines:
            l = l.strip()
            if re.fullmatch(r"\d{5}", l):
                valid.append(l)
            else:
                invalid += 1

        if valid:
            save_numbers(valid)
            st.success(f"Đã lưu {len(valid)} kỳ")
        if invalid:
            st.warning(f"Bỏ qua {invalid} dòng sai định dạng")

    df = load_data()
    st.markdown(f"📊 **Tổng kỳ:** {len(df)}")
    st.dataframe(df.tail(20), use_container_width=True)

# ================== TAB 2 ==================
with tab2:
    df = load_data()

    if len(df) < 10:
        st.warning("Cần tối thiểu 10 kỳ để AI hoạt động")
    else:
        center = ai_center_numbers(df)

        st.subheader("🎯 SỐ TRUNG TÂM (AI)")
        col1, col2 = st.columns(2)
        col1.metric("Tổ hợp A", "".join(center[:3]))
        col2.metric("Tổ hợp B", "".join(center[2:]))

        st.divider()

        st.subheader("🧠 SỐ CHIẾN LƯỢC")
        strategy = ai_strategy_number(center)
        st.metric("AI chọn lọc", strategy)

        st.divider()

        st.subheader("📈 Thống kê nhanh")
        freq = Counter("".join(df["numbers"].dropna()))
        stat = pd.DataFrame(freq.items(), columns=["Số", "Tần suất"]).sort_values("Tần suất", ascending=False)
        st.dataframe(stat, use_container_width=True)

st.caption("⚠ Công cụ phân tích – Không cam kết trúng")
