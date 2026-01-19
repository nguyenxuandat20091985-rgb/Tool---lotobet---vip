import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os
import random

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TINH v6.6",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_data(nums):
    df = load_data()
    df = pd.concat([
        df,
        pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "numbers": nums
        }])
    ], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def parse_numbers(df):
    all_nums = []
    for row in df["numbers"]:
        all_nums.extend([int(x) for x in str(row)])
    return all_nums

# ================== ALGORITHMS ==================
def algo_frequency(nums):
    return Counter(nums)

def algo_gap(df):
    gap = {}
    for n in range(10):
        last = df[df["numbers"].str.contains(str(n))]
        gap[n] = len(df) if last.empty else len(df) - last.index[-1]
    return gap

def algo_pair_support(df):
    pair_count = Counter()
    for row in df["numbers"]:
        digits = list(set(str(row)))
        for d in digits:
            pair_count[int(d)] += 1
    return pair_count

# ================== CORE AI ==================
def predict(df):
    nums = parse_numbers(df)

    freq = algo_frequency(nums)
    gap = algo_gap(df)
    pair = algo_pair_support(df)

    score = {}
    for n in range(10):
        score[n] = (
            freq.get(n, 0) * 0.4 +
            gap.get(n, 0) * 0.35 +
            pair.get(n, 0) * 0.25
        )

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)

    # ===== 5 số dự đoán chung =====
    common = [x[0] for x in ranked[:5]]

    # ===== SỐ PHÁ KỲ (AI RIÊNG) =====
    # điều kiện: gap cao + không nằm trong top tần suất
    freq_rank = [x[0] for x in freq.most_common(5)]
    break_candidates = [
        n for n in range(10)
        if gap.get(n, 0) >= sum(gap.values()) / 10 and n not in freq_rank
    ]

    if break_candidates:
        ai_break = max(break_candidates, key=lambda x: gap[x])
    else:
        ai_break = ranked[5][0]

    return common, ai_break, score

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")
st.caption("Phân tích đủ 5 số – Vá SỐ PHÁ KỲ – Thực chiến")

tab1, tab2, tab3 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🤖 Dự đoán AI",
    "📊 Thống kê"
])

# -------- TAB 1 --------
with tab1:
    st.subheader("Nhập kết quả (mỗi kỳ 5 số)")
    txt = st.text_area("Ví dụ: 12345", height=120)

    if st.button("💾 LƯU DỮ LIỆU"):
        lines = [x.strip() for x in txt.splitlines() if len(x.strip()) == 5]
        for line in lines:
            save_data(line)
        st.success(f"Đã lưu {len(lines)} kỳ")

    df = load_data()
    st.info(f"Tổng số kỳ: {len(df)}")

    if st.button("🗑 XÓA SẠCH"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.warning("Đã xóa toàn bộ dữ liệu")

# -------- TAB 2 --------
with tab2:
    df = load_data()
    if len(df) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
    else:
        common, ai_break, score = predict(df)

        st.markdown("## 🎯 5 SỐ DỰ ĐOÁN CHUNG")
        for n in common:
            st.markdown(
                f"""
                <div style="background:#0b1220;padding:15px;border-radius:15px;
                border:2px solid #00ffc6;margin-bottom:10px;text-align:center">
                <h1 style="color:#00e0ff">{n}</h1>
                <p style="color:gold">Tin cậy: {min(90, int(score[n]))}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("## 🤖 SỐ AI RIÊNG – ƯU TIÊN CAO")
        st.markdown(
            f"""
            <div style="background:#001a0f;padding:20px;border-radius:20px;
            border:3px solid #00ff66;text-align:center">
            <h1 style="color:red">{ai_break}</h1>
            <p style="color:gold;font-size:20px">SỐ PHÁ KỲ – KHẢ NĂNG CAO</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- TAB 3 --------
with tab3:
    df = load_data()
    if df.empty:
        st.info("Chưa có dữ liệu")
    else:
        nums = parse_numbers(df)
        c = Counter(nums)
        st.subheader("Tần suất xuất hiện")
        st.dataframe(pd.DataFrame(c.items(), columns=["Số", "Số lần"]).sort_values("Số"))

        st.subheader("Gợi ý sử dụng")
        st.markdown("""
        - Ưu tiên **SỐ AI RIÊNG**
        - Kết hợp 1–2 số trong **5 số chung**
        - Tránh đánh dàn rộng
        """)
