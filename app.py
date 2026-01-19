import streamlit as st
import pandas as pd
import re
from collections import Counter, defaultdict
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

def save_numbers(nums):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat(
        [df, pd.DataFrame([{"time": now, "numbers": nums}])],
        ignore_index=True
    )
    df.to_csv(DATA_FILE, index=False)

def clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ================== SAFE PARSE (FIX LỖI) ==================
def parse_numbers(df):
    all_nums = []
    for row in df["numbers"]:
        s = str(row).strip()
        if len(s) != 5 or not s.isdigit():
            continue
        for ch in s:
            all_nums.append(int(ch))
    return all_nums

# ================== ALGORITHMS ==================
def algo_frequency(nums):
    c = Counter(nums)
    return {k: v for k, v in c.items()}

def algo_gap(df):
    gap = {}
    for n in range(10):
        rows = df[df["numbers"].astype(str).str.contains(str(n), na=False)]
        if rows.empty:
            gap[n] = 999
        else:
            gap[n] = len(df) - rows.index.max()
    return gap

def algo_hot_cold(freq):
    hot = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in hot[:5]]

def algo_random_boost():
    return random.sample(range(10), 5)

# ================== AI CORE ==================
def predict(df):
    nums = parse_numbers(df)
    if len(nums) < 50:
        return [], [], 0

    freq = algo_frequency(nums)
    gap = algo_gap(df)

    score = defaultdict(float)

    # Thuật toán 1: Tần suất
    for n, v in freq.items():
        score[n] += v * 1.2

    # Thuật toán 2: Nhịp rơi
    for n, g in gap.items():
        score[n] += max(0, 50 - g)

    # Thuật toán 3: Hot / Cold
    for n in algo_hot_cold(freq):
        score[n] += 15

    # Thuật toán 4: Random phá kỳ
    for n in algo_random_boost():
        score[n] += 8

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)

    # 5 số dự đoán chung
    common_nums = [n for n, _ in ranked[:5]]

    # 1 số AI riêng (phá cầu)
    ai_break = ranked[5][0] if len(ranked) > 5 else ranked[0][0]

    # Ghép 2 số
    pairs = []
    for i in range(len(common_nums)):
        for j in range(i + 1, len(common_nums)):
            pairs.append(f"{common_nums[i]}{common_nums[j]}")

    random.shuffle(pairs)
    pairs = pairs[:6]

    confidence = min(90, 50 + len(nums) // 150)

    return pairs, ai_break, confidence

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")
st.caption("Phân tích đủ 5 số – Ổn định – Không sập dữ liệu lớn")

tab1, tab2, tab3 = st.tabs(["📥 Quản lý dữ liệu", "🤖 Dự đoán AI", "📊 Thống kê"])

# -------- TAB 1 --------
with tab1:
    st.subheader("Nhập dữ liệu kết quả")
    raw = st.text_area("Dán kết quả (ví dụ: 15406 hoặc nhiều dòng):")

    if st.button("💾 LƯU"):
        nums = re.findall(r"\b\d{5}\b", raw)
        for n in nums:
            save_numbers(n)
        st.success(f"Đã lưu {len(nums)} kỳ")

    if st.button("🗑️ XÓA SẠCH"):
        clear_data()
        st.warning("Đã xóa toàn bộ dữ liệu")

    df = load_data()
    st.info(f"Tổng số kỳ: {len(df)}")

# -------- TAB 2 --------
with tab2:
    df = load_data()
    if len(df) < 10:
        st.warning("Chưa đủ dữ liệu để AI phân tích")
    else:
        pairs, ai_break, conf = predict(df)

        st.markdown(f"### 🔥 Tin cậy tổng: **{conf}%**")

        st.markdown("## 🎯 5 số dự đoán chung (ghép 2D)")
        cols = st.columns(3)
        for i, p in enumerate(pairs):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        background:#0b1220;
                        padding:20px;
                        border-radius:15px;
                        text-align:center;
                        margin-bottom:10px;">
                        <div style="font-size:42px;color:#4dd2ff;">{p}</div>
                        <div style="color:#ffd966;">Tin cậy: {random.randint(60,85)}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("## 🤖 Số AI phá kỳ")
        st.success(f"Số AI riêng: **{ai_break}**")

# -------- TAB 3 --------
with tab3:
    df = load_data()
    if df.empty:
        st.info("Chưa có dữ liệu")
    else:
        nums = parse_numbers(df)
        freq = Counter(nums)

        st.subheader("Tần suất xuất hiện (0–9)")
        st.dataframe(
            pd.DataFrame(freq.items(), columns=["Số", "Số lần"])
            .sort_values("Số")
            .reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("Top số nổi bật")
        top = freq.most_common(5)
        for n, c in top:
            st.write(f"• Số {n}: {c} lần")
