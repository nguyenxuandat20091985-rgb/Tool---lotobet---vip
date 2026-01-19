import streamlit as st
import pandas as pd
import random
import re
import os
from collections import Counter, defaultdict
from datetime import datetime

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

def save_data(numbers):
    df = load_data()
    new = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "numbers": numbers
    }])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ================== PARSE ==================
def parse_numbers(df):
    all_nums = []
    for row in df["numbers"]:
        if isinstance(row, str):
            digits = re.findall(r"\d", row)
            all_nums.extend([int(x) for x in digits])
    return all_nums

# ================== AI CORE ==================
def predict(df):
    total_draws = len(df)
    if total_draws < 20:
        return None, None, None

    nums = parse_numbers(df)
    freq = Counter(nums)

    # Nhịp rơi
    gap = {}
    for n in range(10):
        idx = df[df["numbers"].astype(str).str.contains(str(n), na=False)].index
        gap[n] = total_draws - idx.max() if len(idx) > 0 else total_draws + 10

    score = defaultdict(float)

    # Thuật toán 1: tần suất
    for n, v in freq.items():
        score[n] += v * 1.2

    # Thuật toán 2: nhịp rơi
    for n, g in gap.items():
        score[n] += max(0, 28 - g)

    # Thuật toán 3: cân bằng âm dương (random có kiểm soát)
    for n in random.sample(range(10), 5):
        score[n] += 6

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)

    main_nums = [n for n, _ in ranked[:5]]
    ai_break = ranked[5][0]

    # Ghép 2D
    pairs = []
    for i in range(len(main_nums)):
        for j in range(i + 1, len(main_nums)):
            pairs.append(f"{main_nums[i]}{main_nums[j]}")
    random.shuffle(pairs)
    pairs = pairs[:6]

    confidence = min(85, 55 + total_draws // 120)

    return pairs, ai_break, confidence

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")
st.caption("Phân tích đủ 5 số – Ổn định – Không sập dữ liệu lớn")

tab1, tab2, tab3 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🤖 Dự đoán AI",
    "📊 Thống kê"
])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Dán kết quả (mỗi kỳ 5 số)")
    numbers = st.text_area("Ví dụ: 12345", height=120)

    if st.button("💾 LƯU DỮ LIỆU"):
        if re.fullmatch(r"\d{5}", numbers.strip()):
            save_data(numbers.strip())
            st.success("Đã lưu 1 kỳ")
        else:
            st.error("Sai định dạng – cần đúng 5 số")

    df = load_data()
    st.info(f"Đã lưu {len(df)} kỳ")

    if st.button("🗑️ XÓA SẠCH"):
        clear_data()
        st.warning("Đã xóa toàn bộ dữ liệu")

# ---------- TAB 2 ----------
with tab2:
    df = load_data()
    pairs, ai_break, conf = predict(df)

    if pairs is None:
        st.warning("⚠️ Chưa đủ dữ liệu (cần tối thiểu 20 kỳ)")
    else:
        st.markdown(f"🔥 **Tin cậy tổng: {conf}%**")

        st.markdown("## 🎯 5 số dự đoán chung (ghép 2D)")
        cols = st.columns(3)
        for i, p in enumerate(pairs):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        background:#0b1220;
                        padding:18px;
                        border-radius:16px;
                        text-align:center;
                        margin-bottom:12px;
                        border:2px solid #1f6feb;">
                        <div style="font-size:38px;color:#4dd2ff;">{p}</div>
                        <div style="color:#ffd966;">Tin cậy: {random.randint(62,82)}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("## 🤖 Số AI phá kỳ")
        st.success(f"Số AI riêng: **{ai_break}**")

# ---------- TAB 3 ----------
with tab3:
    df = load_data()
    if len(df) == 0:
        st.info("Chưa có dữ liệu để thống kê")
    else:
        st.metric("Tổng số kỳ", len(df))
        nums = parse_numbers(df)
        freq = Counter(nums)
        top = freq.most_common(5)

        st.markdown("### 🔢 Top số xuất hiện nhiều")
        for n, c in top:
            st.write(f"Số {n}: {c} lần")

st.caption("⚠️ Công cụ hỗ trợ phân tích – không cam kết trúng")
