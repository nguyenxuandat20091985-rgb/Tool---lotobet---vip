import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================= DATA =================
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers"])

    df = pd.read_csv(DATA_FILE)
    df["numbers"] = df["numbers"].astype(str)
    return df[["time", "numbers"]]

def save_many(values):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for v in values:
        if v.isdigit() and len(v) == 5:
            rows.append({"time": now, "numbers": v})

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

    return len(rows)

# ================= CORE =================
def parse_numbers(v):
    return [int(x) for x in str(v) if x.isdigit()][:5]

def flatten(df):
    out = []
    for v in df["numbers"]:
        out.extend(parse_numbers(v))
    return out

def recent_weight(df, window=20):
    recent = df.tail(window)
    nums = flatten(recent)
    return Counter(nums)

def build_groups(top_nums):
    if len(top_nums) < 3:
        return []
    return list(combinations(top_nums[:5], 3))[:2]

def ai_smart_pick(freq_all, freq_recent):
    score = {}

    for n in range(10):
        f_all = freq_all.get(n, 0)
        f_recent = freq_recent.get(n, 0)

        # loại số quá nóng hoặc chết
        if f_all == 0 or f_recent == 0:
            continue

        # công thức điểm AI
        score[n] = (f_recent * 2) - (f_all * 0.5)

    # sắp xếp theo điểm
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)

    picks = [str(n) for n, _ in ranked[:2]]
    if len(picks) < 2:
        return "--"

    return "".join(picks)

# ================= UI =================
st.title("🔷 NUMCORE")
st.caption("AI phân tích cầu – Không đoán bừa – Ưu tiên sống")

tab1, tab2 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🎯 Phân tích & Dự đoán"
])

# ============ TAB 1 ============
with tab1:
    st.subheader("📥 Nhập nhiều kỳ")

    raw = st.text_area(
        "Mỗi dòng = 1 kỳ (5 số)",
        height=160
    )

    if st.button("💾 Lưu dữ liệu"):
        saved = save_many([x.strip() for x in raw.splitlines()])
        if saved:
            st.success(f"Đã lưu {saved} kỳ")
        else:
            st.error("Không có dữ liệu hợp lệ")

    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

# ============ TAB 2 ============
with tab2:
    df = load_data()

    if len(df) < 6:
        st.warning("Chưa đủ dữ liệu phân tích")
    else:
        all_nums = flatten(df)
        freq_all = Counter(all_nums)
        freq_recent = recent_weight(df)

        top = [n for n, _ in freq_all.most_common(6)]

        st.subheader("🎯 SỐ TRUNG TÂM")
        groups = build_groups(top)

        c1, c2 = st.columns(2)
        if len(groups) > 0:
            c1.metric("Trung tâm A", "".join(map(str, groups[0])))
        if len(groups) > 1:
            c2.metric("Trung tâm B", "".join(map(str, groups[1])))

        st.divider()

        st.subheader("🧠 SỐ CHIẾN LƯỢC (AI)")
        ai_num = ai_smart_pick(freq_all, freq_recent)

        if ai_num == "--":
            st.error("AI từ chối đánh – Cầu xấu")
        else:
            st.success(f"AI đề xuất: **{ai_num}**")

        st.divider()

        st.subheader("📊 THỐNG KÊ")
        st.write(f"• Số kỳ phân tích: **{len(df)}**")
        st.write("• AI ưu tiên số đang lên, né số bão hòa")

st.caption("NUMCORE v6.6 – AI nâng cấp – Không random – Không số chập")
