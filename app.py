import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import os
import math

# ================= CONFIG =================
st.set_page_config(page_title="NUMCORE", layout="centered")
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
    return [int(x) for x in v if x.isdigit()][:5]

def flatten(df):
    out = []
    for v in df["numbers"]:
        out.extend(parse_numbers(v))
    return out

def freq_window(df, n):
    nums = []
    for v in df.tail(n)["numbers"]:
        nums.extend(parse_numbers(v))
    return Counter(nums)

def gap_score(df):
    pos = defaultdict(list)
    for i, v in enumerate(df["numbers"]):
        for n in parse_numbers(v):
            pos[n].append(i)
    score = {}
    for n in range(10):
        if n not in pos or len(pos[n]) < 2:
            score[n] = -2
        else:
            g = pos[n][-1] - pos[n][-2]
            score[n] = 2 if 2 <= g <= 6 else -1
    return score

# ================= AI ENSEMBLE CORE =================
def ai_super_engine(df):
    score = defaultdict(float)

    all_freq = Counter(flatten(df))
    mean = sum(all_freq.values()) / 10 if all_freq else 0

    # 1️⃣ Tần suất đa khung
    for w, wgt in [(5,2),(10,1.6),(20,1.2),(40,0.8)]:
        fw = freq_window(df, min(w,len(df)))
        for n in range(10):
            score[n] += fw.get(n,0) * wgt

    # 2️⃣ Xu hướng
    f10 = freq_window(df, min(10,len(df)))
    f20 = freq_window(df, min(20,len(df)))
    for n in range(10):
        score[n] += max(0, f10.get(n,0) - f20.get(n,0)*0.7)

    # 3️⃣ Né số bão hòa
    for n in range(10):
        if all_freq.get(n,0) > mean*1.7:
            score[n] -= 3

    # 4️⃣ Né số chết
    for n in range(10):
        if all_freq.get(n,0) == 0:
            score[n] -= 5

    # 5️⃣ Gap
    gap = gap_score(df)
    for n in range(10):
        score[n] += gap[n]

    # 6️⃣ Ổn định
    for n in range(10):
        score[n] += 0.5 if f10.get(n,0)>0 else -0.5

    # ================= KẾT QUẢ =================
    ranked = sorted(score.items(), key=lambda x:x[1], reverse=True)
    top5 = [n for n,s in ranked if s>0][:5]

    best_pair = "--"
    if len(top5) >= 2:
        pair_scores = {}
        for a,b in combinations(top5,2):
            pair_scores[f"{a}{b}"] = score[a] + score[b]
        best_pair = max(pair_scores, key=pair_scores.get)

    return top5, best_pair, score

# ================= UI =================
st.title("🔷 NUMCORE v6.6")
st.caption("AI Ensemble siêu lọc – Ưu tiên sống – Không random")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu","🎯 AI Phân tích"])

with tab1:
    raw = st.text_area("Mỗi dòng 1 kỳ (5 số)", height=160)
    if st.button("💾 Lưu dữ liệu"):
        saved = save_many([x.strip() for x in raw.splitlines()])
        st.success(f"Đã lưu {saved} kỳ") if saved else st.error("Không có dữ liệu hợp lệ")
    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    df = load_data()
    if len(df) < 8:
        st.warning("Chưa đủ dữ liệu")
    else:
        top5, pair, score = ai_super_engine(df)

        st.subheader("🔥 5 SỐ CHIẾN LƯỢC MẠNH NHẤT")
        st.write(" ".join(str(x) for x in top5))

        st.divider()
        st.subheader("🎯 2 TINH ĐÁNH CHÍNH")
        st.success(pair if pair!="--" else "AI từ chối đánh")

st.caption("NUMCORE v6.6 – AI Ensemble siêu lọc – Không số chập")
