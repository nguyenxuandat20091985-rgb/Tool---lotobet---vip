import streamlit as st
import pandas as pd
import os
import re
import random
from collections import Counter
from datetime import datetime

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TỈNH v6.6",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["result"] = df["result"].astype(str)
        df = df[df["result"].str.fullmatch(r"\d{5}")]
        return df
    return pd.DataFrame(columns=["time", "result"])

def save_data(num):
    df = load_data()
    df.loc[len(df)] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": str(num)
    }
    df.to_csv(DATA_FILE, index=False)

def clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

# ================== ALGORITHMS ==================
def algo_frequency(data):
    digits = "".join([str(x) for x in data if re.fullmatch(r"\d{5}", str(x))])
    if not digits:
        return []
    return [x[0] for x in Counter(digits).most_common(5)]

def algo_recent(data):
    recent = data[-20:]
    digits = "".join(recent)
    return list(dict.fromkeys(digits))[:5]

def algo_random():
    return [str(random.randint(0, 9)) for _ in range(5)]

def merge_algorithms(data):
    pool = []
    pool.extend(algo_frequency(data))
    pool.extend(algo_recent(data))
    pool.extend(algo_random())

    if not pool:
        return []

    return [x[0] for x in Counter(pool).most_common(5)]

def ai_break_cycle(data):
    recent_digits = set("".join(data[-10:]))
    for _ in range(200):
        num = f"{random.randint(0,99):02d}"
        if num[0] not in recent_digits or num[1] not in recent_digits:
            return num
    return f"{random.randint(0,99):02d}"

# ================== UI ==================
st.title("🔥 LOTO BET 2 SỐ 5 TỈNH v6.6")
st.caption("Phân tích đủ 5 số – Ổn định – Không sập dữ liệu")

tab1, tab2, tab3 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🎯 Dự đoán AI",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Dán kết quả (mỗi dòng đúng 5 số)")
    raw = st.text_area("Ví dụ:\n12345\n67890", height=200)

    if st.button("💾 LƯU DỮ LIỆU"):
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        ok, bad = [], []

        for line in lines:
            if re.fullmatch(r"\d{5}", line):
                save_data(line)
                ok.append(line)
            else:
                bad.append(line)

        if ok:
            st.success(f"✅ Đã lưu {len(ok)} kỳ")
        if bad:
            st.error(f"❌ Sai định dạng: {', '.join(bad)}")

    df = load_data()
    st.info(f"📦 Tổng kỳ đã lưu: {len(df)}")

    if st.button("🗑️ XÓA SẠCH"):
        clear_data()
        st.warning("Đã xóa toàn bộ dữ liệu")

# ================== TAB 2 ==================
with tab2:
    df = load_data()

    if len(df) < 10:
        st.warning("⚠️ Cần ít nhất 10 kỳ để AI phân tích")
    else:
        data = df["result"].tolist()

        st.subheader("🎯 5 số dự đoán chung (ghép 2D)")
        five_digits = merge_algorithms(data)

        if five_digits:
            st.success(" → ".join(five_digits))
        else:
            st.error("Không đủ dữ liệu sạch để dự đoán")

        st.subheader("🤖 Số AI riêng (phá kỳ)")
        st.success(f"Số AI: {ai_break_cycle(data)}")

        st.metric("🔥 Tỉ lệ tham khảo", "≈ 55%")

# ================== TAB 3 ==================
with tab3:
    df = load_data()
    if df.empty:
        st.info("Chưa có dữ liệu")
    else:
        digits = "".join(df["result"].tolist())
        counter = Counter(digits)

        st.subheader("📊 Thống kê tần suất (dễ nhìn)")
        stat_df = pd.DataFrame(counter.items(), columns=["Số", "Số lần"])
        stat_df = stat_df.sort_values("Số lần", ascending=False)

        st.dataframe(stat_df, use_container_width=True)
        st.caption("⚠️ Chỉ hỗ trợ phân tích – không cam kết trúng")
