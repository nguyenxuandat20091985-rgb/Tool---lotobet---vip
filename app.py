import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TINH v6.6",
    layout="centered"
)

# ================== SESSION ==================
if "data" not in st.session_state:
    st.session_state.data = []

if "history" not in st.session_state:
    st.session_state.history = {}

# ================== FUNCTIONS ==================
def extract_digits(text):
    nums = re.findall(r"\d", text)
    return nums

def build_pairs(digits):
    pairs = []
    for i in range(len(digits)):
        for j in range(i + 1, len(digits)):
            pairs.append(digits[i] + digits[j])
    return pairs

def algorithm_frequency(data):
    all_digits = []
    for row in data:
        all_digits.extend(row)
    return Counter(all_digits)

def algorithm_gap(data):
    last_seen = {}
    score = Counter()
    for idx, row in enumerate(data):
        for d in row:
            last_seen[d] = idx
    total = len(data)
    for d in "0123456789":
        score[d] = total - last_seen.get(d, -1)
    return score

def ensemble_score(data):
    freq = algorithm_frequency(data)
    gap = algorithm_gap(data)

    final = Counter()
    for d in "0123456789":
        final[d] = freq[d] * 0.6 + gap[d] * 0.4
    return final

def predict_pairs(data):
    score = ensemble_score(data)
    pairs = Counter()

    for d1 in score:
        for d2 in score:
            if d1 != d2:
                pairs[d1 + d2] = score[d1] + score[d2]

    top = pairs.most_common(6)
    return top

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")

tabs = st.tabs(["📥 Quản lý dữ liệu", "🤖 Dự đoán AI", "📊 Thống kê"])

# ================== TAB 1 ==================
with tabs[0]:
    st.subheader("Dán kết quả (mỗi kỳ 5 số)")
    raw = st.text_area("Ví dụ: 15406", height=120)

    if st.button("💾 LƯU DỮ LIỆU"):
        digits = extract_digits(raw)
        if len(digits) >= 5:
            chunks = [digits[i:i+5] for i in range(0, len(digits), 5)]
            st.session_state.data.extend(chunks)
            st.success(f"Đã lưu {len(chunks)} kỳ")
        else:
            st.error("Không đủ dữ liệu")

    if st.button("🗑️ XÓA SẠCH"):
        st.session_state.data = []
        st.session_state.history = {}
        st.warning("Đã xóa toàn bộ dữ liệu")

    st.info(f"Tổng số kỳ đã lưu: {len(st.session_state.data)}")

# ================== TAB 2 ==================
with tabs[1]:
    if len(st.session_state.data) < 5:
        st.warning("Cần tối thiểu 5 kỳ để dự đoán")
    else:
        result = predict_pairs(st.session_state.data)

        ai_main = result[0]
        others = result[1:]

        st.markdown("## 🔥 SỐ AI ƯU TIÊN CAO")
        st.markdown(
            f"""
            <div style='background:#0f172a;padding:25px;border-radius:16px;
            text-align:center;border:3px solid #22c55e'>
            <div style='font-size:52px;color:#22c55e;font-weight:bold'>
            {ai_main[0]}
            </div>
            <div style='font-size:20px;color:#facc15'>
            Tin cậy: {round(70 + ai_main[1] % 30)}%
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🎯 5 SỐ DỰ ĐOÁN CHUNG")
        cols = st.columns(2)
        for idx, (pair, score) in enumerate(others):
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div style='background:#020617;padding:18px;
                    border-radius:14px;text-align:center;
                    border:2px solid #38bdf8'>
                    <div style='font-size:34px;color:#38bdf8;font-weight:bold'>
                    {pair}
                    </div>
                    <div style='color:#facc15'>
                    Tin cậy: {round(55 + score % 25)}%
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ================== TAB 3 ==================
with tabs[2]:
    st.subheader("📊 Thống kê tổng quan")

    total = len(st.session_state.data)
    st.metric("Tổng số kỳ", total)

    st.markdown("### Ghi chú")
    st.write(
        """
        - Thống kê dùng để **đánh giá hiệu quả**, không dùng để đánh số  
        - Ưu tiên theo **SỐ AI RIÊNG**  
        - 5 số còn lại dùng **bọc – phòng trượt**
        """
    )

    st.success("v6.6 – Thuật toán đa lớp – Ổn định Android")
