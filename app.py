import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="NUMCORE v6.6", layout="centered")
DATA_FILE = "data.csv"

# ================= DATA =================
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers"])
    df = pd.read_csv(DATA_FILE)
    df["numbers"] = df["numbers"].astype(str)
    return df

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
    return [int(x) for x in v if x.isdigit()]

def flatten(df):
    out = []
    for v in df["numbers"]:
        out.extend(parse_numbers(v))
    return out

def recent_freq(df, window=20):
    return Counter(flatten(df.tail(window)))

def ai_score(freq_all, freq_recent, total_rounds):
    scores = {}
    for n in range(10):
        fa = freq_all.get(n, 0)
        fr = freq_recent.get(n, 0)

        if fr == 0:
            continue  # số chết → loại

        score = (
            fr * 3 +                 # đang lên
            fa * 0.5 -               # không quá nóng
            abs(fa - total_rounds*0.5) * 0.05
        )

        scores[n] = score
    return scores

# ================= UI =================
st.title("🔷 NUMCORE AI v6.6")
st.caption("AI lọc cầu – Ưu tiên sống – Không all-in")

tab1, tab2 = st.tabs(["📥 Dữ liệu", "🎯 Phân tích"])

# ===== TAB 1 =====
with tab1:
    raw = st.text_area("Mỗi dòng = 1 kỳ (5 số)", height=150)

    if st.button("💾 Lưu dữ liệu"):
        saved = save_many([x.strip() for x in raw.splitlines()])
        if saved > 0:
            st.success(f"Đã lưu {saved} kỳ")
        else:
            st.error("Không có dữ liệu hợp lệ")

    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

# ===== TAB 2 =====
with tab2:
    df = load_data()

    if len(df) < 10:
        st.warning("Chưa đủ dữ liệu để AI hoạt động")
    else:
        all_nums = flatten(df)
        freq_all = Counter(all_nums)
        freq_recent = recent_freq(df)

        scores = ai_score(freq_all, freq_recent, len(df))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        picks = [str(n) for n, _ in ranked[:5]]

        st.subheader("🎯 SỐ TRUNG TÂM")
        top = [n for n, _ in freq_all.most_common(6)]
        groups = list(combinations(top, 3))[:2]

        c1, c2 = st.columns(2)
        if len(groups) > 0:
            c1.metric("Trung tâm A", "".join(map(str, groups[0])))
        if len(groups) > 1:
            c2.metric("Trung tâm B", "".join(map(str, groups[1])))

        st.divider()

        st.subheader("🧠 5 SỐ CHIẾN LƯỢC (AI)")
        if len(picks) < 2:
            st.error("🔴 Cầu xấu – AI khuyên nghỉ")
        else:
            st.success(" • ".join(picks))
            st.info("👉 Đánh nhỏ – xoay vòng – KHÔNG all-in")

        st.divider()
        st.write(f"📊 Kỳ đã phân tích: **{len(df)}**")

st.caption("NUMCORE v6.6 – AI lọc cầu – Ổn định – Không ảo")
