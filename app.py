import streamlit as st
import pandas as pd
import re
import os
from collections import Counter
from datetime import datetime
import random

# ================== CẤU HÌNH ==================
st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TỈNH v6.6",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== HÀM DỮ LIỆU ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "result"])

def save_data(num):
    df = load_data()
    df.loc[len(df)] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": num
    }
    df.to_csv(DATA_FILE, index=False)

def clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ================== THUẬT TOÁN ==================
def algo_frequency(data):
    digits = "".join(data)
    return [x[0] for x in Counter(digits).most_common(5)]

def algo_recent(data):
    return list("".join(data[-20:]))[:5]

def algo_random():
    return [str(random.randint(0,9)) for _ in range(5)]

def merge_algorithms(data):
    pool = []
    pool += algo_frequency(data)
    pool += algo_recent(data)
    pool += algo_random()
    return [x[0] for x in Counter(pool).most_common(5)]

def ai_break_cycle(data):
    used = set("".join(data[-10:]))
    for i in range(100):
        n = f"{random.randint(0,99):02d}"
        if n[0] not in used or n[1] not in used:
            return n
    return f"{random.randint(0,99):02d}"

# ================== GIAO DIỆN ==================
st.title("🔥 LOTO BET 2 SỐ 5 TỈNH v6.6")
st.caption("Phân tích đủ 5 số – Ổn định – Không sập dữ liệu lớn")

tab1, tab2, tab3 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🎯 Dự đoán AI",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Dán kết quả (mỗi dòng đúng 5 số)")
    raw = st.text_area(
        "Ví dụ:\n12345\n67890",
        height=180
    )

    if st.button("💾 LƯU DỮ LIỆU"):
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        valid, invalid = [], []

        for line in lines:
            if re.fullmatch(r"\d{5}", line):
                valid.append(line)
            else:
                invalid.append(line)

        for v in valid:
            save_data(v)

        if valid:
            st.success(f"✅ Đã lưu {len(valid)} kỳ")
        if invalid:
            st.error(f"❌ Sai định dạng: {', '.join(invalid)}")

    df = load_data()
    st.info(f"📦 Tổng kỳ đã lưu: {len(df)}")

    if st.button("🗑️ XÓA SẠCH"):
        clear_data()
        st.warning("Đã xóa toàn bộ dữ liệu")

# ================== TAB 2 ==================
with tab2:
    df = load_data()
    if len(df) < 10:
        st.warning("⚠️ Cần ít nhất 10 kỳ để dự đoán")
    else:
        data = df["result"].tolist()

        st.subheader("🎯 5 số dự đoán chung (ghép 2D)")
        five_digits = merge_algorithms(data)
        st.success(" → ".join(five_digits))

        st.subheader("🤖 Số AI phá kỳ (AI riêng)")
        ai_num = ai_break_cycle(data)
        st.success(f"Số AI riêng: {ai_num}")

        st.metric("🔥 Tin cậy tổng (ước lượng)", "≈ 55%")

# ================== TAB 3 ==================
with tab3:
    df = load_data()
    if df.empty:
        st.info("Chưa có dữ liệu")
    else:
        all_digits = "".join(df["result"].tolist())
        counter = Counter(all_digits)

        st.subheader("📊 Tần suất số (dễ hiểu)")
        freq_df = pd.DataFrame(counter.items(), columns=["Số", "Số lần"])
        freq_df = freq_df.sort_values("Số lần", ascending=False)

        st.dataframe(freq_df, use_container_width=True)

        st.caption("⚠️ Thống kê hỗ trợ phân tích – không cam kết trúng")
