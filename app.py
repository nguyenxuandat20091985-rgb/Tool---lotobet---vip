import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime
import os
import math

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE v6.6",
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

def freq_all(df):
    return Counter(flatten(df))

def freq_recent(df, window=20):
    recent = df.tail(window)
    return Counter(flatten(recent))

def hot_numbers(freq_all, freq_recent):
    score = {}
    for n in range(10):
        fa = freq_all.get(n, 0)
        fr = freq_recent.get(n, 0)
        if fa == 0:
            continue
        score[n] = (fr * 2) + math.sqrt(fa)
    return [n for n, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]

def build_centers(nums):
    if len(nums) < 5:
        return []
    return list(combinations(nums[:5], 3))[:2]

# ================= AI ENGINE =================
def ai_score_pair(pair, freq_all, freq_recent, last_nums):
    a, b = pair
    score = 0

    score += freq_recent.get(a, 0) * 2
    score += freq_recent.get(b, 0) * 2

    score += freq_all.get(a, 0) * 0.5
    score += freq_all.get(b, 0) * 0.5

    if a in last_nums:
        score -= 1
    if b in last_nums:
        score -= 1

    if abs(a - b) <= 1:
        score -= 0.5

    return round(score, 2)

def ai_predict_pairs(df):
    fa = freq_all(df)
    fr = freq_recent(df)
    last_nums = parse_numbers(df.iloc[-1]["numbers"])

    candidates = list(combinations(range(10), 2))
    scored = []

    for p in candidates:
        s = ai_score_pair(p, fa, fr, last_nums)
        if s > 0:
            scored.append((p, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    top = []
    for p, s in scored[:10]:
        conf = min(95, int(50 + s * 3))
        top.append({
            "pair": f"{p[0]}{p[1]}",
            "confidence": conf
        })

    return top

# ================= UI =================
st.title("🔷 NUMCORE v6.6 – SAFE PRO")
st.caption("AI sàng lọc cầu | Giảm rủi ro | Không spam số")

tab1, tab2 = st.tabs([
    "📥 Nhập dữ liệu",
    "🎯 AI Phân tích"
])

# ============ TAB 1 ============
with tab1:
    st.subheader("📥 Nhập nhiều kỳ (5 số / dòng)")

    raw = st.text_area(
        "Ví dụ:\n17723\n55324\n95060",
        height=160
    )

    if st.button("💾 Lưu dữ liệu"):
        saved = save_many([x.strip() for x in raw.splitlines()])
        if saved:
            st.success(f"Đã lưu {saved} kỳ hợp lệ")
        else:
            st.error("Không có dữ liệu hợp lệ")

    df = load_data()
    if not df.empty:
        st.subheader("📄 10 kỳ gần nhất")
        st.dataframe(df.tail(10), use_container_width=True)

# ============ TAB 2 ============
with tab2:
    df = load_data()

    if len(df) < 15:
        st.warning("⚠️ Cần tối thiểu 15 kỳ để AI phân tích")
    else:
        fa = freq_all(df)
        fr = freq_recent(df)

        hot = hot_numbers(fa, fr)

        st.subheader("🎯 SỐ TRUNG TÂM")
        centers = build_centers(hot)

        c1, c2 = st.columns(2)
        if len(centers) > 0:
            c1.metric("Trung tâm A", "".join(map(str, centers[0])))
        if len(centers) > 1:
            c2.metric("Trung tâm B", "".join(map(str, centers[1])))

        st.divider()

        st.subheader("🧠 AI CHỐT 3 CẶP CHIẾN LƯỢC")

        predictions = ai_predict_pairs(df)
        strong = [p for p in predictions if p["confidence"] >= 70][:3]

        if not strong:
            st.error("🚦 AI KHUYÊN: KHÔNG ĐÁNH KỲ NÀY")
        else:
            avg_conf = sum(p["confidence"] for p in strong) / len(strong)

            if avg_conf >= 75:
                st.success("✅ CẦU ĐẸP – NÊN ĐÁNH")
            else:
                st.warning("⚠️ ĐÁNH NHẸ – GIỮ VỐN")

            cols = st.columns(len(strong))
            for i, p in enumerate(strong):
                cols[i].metric(
                    f"Cặp {i+1}",
                    p["pair"],
                    f"{p['confidence']}%"
                )

        st.divider()
        st.subheader("📊 Thống kê")
        st.write(f"• Tổng kỳ phân tích: **{len(df)}**")
        st.write("• AI ưu tiên cầu đang lên, né cầu gãy")

st.caption("NUMCORE v6.6 SAFE PRO – Ít số – Rõ ràng – Không ảo")
