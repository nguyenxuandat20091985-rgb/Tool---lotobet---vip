import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="NUMCORE v6.7", layout="centered")
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
    rows = [{"time": now, "numbers": v} for v in values if v.isdigit() and len(v) == 5]
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
    return len(rows)

# ================= CORE =================
def parse(v):
    return [int(x) for x in v]

def score_numbers(df):
    all_nums = []
    last_seen = {}
    for idx, row in df.iterrows():
        nums = parse(row["numbers"])
        for n in nums:
            all_nums.append(n)
            last_seen[n] = idx

    freq = Counter(all_nums)
    total = len(df)

    scores = {}
    for n in range(10):
        f = freq.get(n, 0)
        if f == 0:
            continue
        cold = total - last_seen.get(n, total)
        score = (f * 1.2) + (cold * 0.8)
        if f > total * 0.25:
            score *= 0.6  # phạt số quá nóng
        scores[n] = round(score, 2)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def ai_safe_mode(top_scores):
    if len(top_scores) < 3:
        return "⛔ NGHỈ – DỮ LIỆU YẾU", "red"
    spread = top_scores[0][1] - top_scores[2][1]
    if spread > 5:
        return "✅ NÊN ĐÁNH (CẦU ĐẸP)", "green"
    if spread > 2:
        return "⚠️ ĐÁNH NHẸ", "orange"
    return "⛔ NGHỈ – CẦU XẤU", "red"

# ================= UI =================
st.title("🔷 NUMCORE v6.7 – SAFE MODE")
st.caption("AI lọc số – Ưu tiên sống – Không all-in")

tab1, tab2 = st.tabs(["📥 Dữ liệu", "🎯 AI Dự đoán"])

with tab1:
    raw = st.text_area("Mỗi dòng 1 kỳ (5 số)")
    if st.button("💾 Lưu"):
        s = save_many(raw.splitlines())
        st.success(f"Đã lưu {s} kỳ") if s else st.error("Không hợp lệ")
    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    df = load_data()
    if len(df) < 10:
        st.warning("Chưa đủ dữ liệu")
    else:
        ranked = score_numbers(df)
        top5 = ranked[:5]

        st.subheader("🧠 5 SỐ CHIẾN LƯỢC (AI)")
        st.write("👉 **Ưu tiên đánh 3 số đầu**")
        for i, (n, s) in enumerate(top5, 1):
            st.write(f"{i}. **{n}** — điểm AI: `{s}`")

        status, color = ai_safe_mode(top5)
        st.divider()
        st.subheader("🚦 TRẠNG THÁI AI")
        st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)

st.caption("NUMCORE v6.7 – SAFE MODE – Không ảo – Không gấp thếp")
