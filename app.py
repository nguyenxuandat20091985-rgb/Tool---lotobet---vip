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
    return [int(x) for x in str(v) if x.isdigit()][:5]

def flatten(df):
    out = []
    for v in df["numbers"]:
        out.extend(parse_numbers(v))
    return out

def freq_window(df, window):
    nums = []
    for v in df.tail(window)["numbers"]:
        nums.extend(parse_numbers(v))
    return Counter(nums)

def gaps_by_number(df):
    """Khoảng cách giữa các lần xuất hiện gần nhất"""
    pos = defaultdict(list)
    for i, v in enumerate(df["numbers"]):
        for n in parse_numbers(v):
            pos[n].append(i)
    gaps = {}
    for n, lst in pos.items():
        if len(lst) >= 2:
            gaps[n] = lst[-1] - lst[-2]
        else:
            gaps[n] = 999
    return gaps

def position_balance(df):
    """Phân bố vị trí 5 tinh (không cố định)"""
    pos_cnt = Counter()
    for v in df["numbers"]:
        nums = parse_numbers(v)
        for idx, n in enumerate(nums):
            pos_cnt[(idx, n)] += 1
    # gom theo số
    score = defaultdict(int)
    for (idx, n), c in pos_cnt.items():
        score[n] += c
    return score

def build_groups(top_nums):
    if len(top_nums) < 3:
        return []
    return list(combinations(top_nums[:5], 3))[:2]

# ================= AI – ENSEMBLE 100 =================
def ai_ensemble_pick(df):
    scores = defaultdict(float)

    if len(df) < 8:
        return "--", {}

    # 1) Tần suất đa cửa sổ (5/10/20/40)
    for w, weight in [(5,1.5),(10,1.2),(20,1.0),(40,0.8)]:
        fw = freq_window(df, min(w, len(df)))
        for n in range(10):
            scores[n] += fw.get(n,0) * weight

    # 2) Xu hướng (so sánh 10 vs 20)
    f10 = freq_window(df, min(10, len(df)))
    f20 = freq_window(df, min(20, len(df)))
    for n in range(10):
        scores[n] += max(0, f10.get(n,0) - 0.6*f20.get(n,0))

    # 3) Độ mới (recency)
    recent = freq_window(df, min(8, len(df)))
    for n in range(10):
        scores[n] += recent.get(n,0) * 1.3

    # 4) Tránh bão hòa (phạt số quá nóng)
    fall = Counter(flatten(df))
    mean = sum(fall.values())/10 if fall else 0
    for n in range(10):
        if fall.get(n,0) > mean*1.6:
            scores[n] -= 2.0

    # 5) Cân bằng nóng–nguội
    for n in range(10):
        scores[n] += 0.5 if fall.get(n,0) >= mean*0.7 else 0.2

    # 6) Khoảng cách (gap)
    gaps = gaps_by_number(df)
    for n in range(10):
        g = gaps.get(n,999)
        if 2 <= g <= 6:
            scores[n] += 1.2
        elif g > 12:
            scores[n] -= 0.8

    # 7) Phân bố vị trí
    pos_score = position_balance(df)
    for n in range(10):
        scores[n] += math.log(1+pos_score.get(n,0)) * 0.3

    # 8) Ổn định (variance nhẹ)
    for n in range(10):
        scores[n] += 0.3 if recent.get(n,0) > 0 else -0.2

    # 9) Loại số chết
    for n in range(10):
        if fall.get(n,0) == 0:
            scores[n] -= 5.0

    # 10) Bỏ phiếu cuối + lọc cứng
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picks = []
    for n, s in ranked:
        if s <= 0:
            continue
        if n not in picks:
            picks.append(n)
        if len(picks) == 2:
            break

    if len(picks) < 2:
        return "--", scores

    return f"{picks[0]}{picks[1]}", scores

# ================= UI =================
st.title("🔷 NUMCORE")
st.caption("AI Ensemble 100 – Lọc cầu – Không random – Ưu tiên sống")

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu","🎯 Phân tích & Dự đoán"])

# TAB 1
with tab1:
    st.subheader("📥 Nhập nhiều kỳ (mỗi dòng 5 số)")
    raw = st.text_area("Dữ liệu", height=160)
    if st.button("💾 Lưu dữ liệu"):
        saved = save_many([x.strip() for x in raw.splitlines()])
        st.success(f"Đã lưu {saved} kỳ") if saved else st.error("Không có dữ liệu hợp lệ")
    df = load_data()
    if not df.empty:
        st.dataframe(df.tail(10), use_container_width=True)

# TAB 2
with tab2:
    df = load_data()
    if len(df) < 8:
        st.warning("Chưa đủ dữ liệu phân tích")
    else:
        all_nums = flatten(df)
        freq_all = Counter(all_nums)
        top = [n for n,_ in freq_all.most_common(6)]

        st.subheader("🎯 SỐ TRUNG TÂM")
        groups = build_groups(top)
        c1,c2 = st.columns(2)
        if len(groups)>0: c1.metric("Trung tâm A","".join(map(str,groups[0])))
        if len(groups)>1: c2.metric("Trung tâm B","".join(map(str,groups[1])))

        st.divider()

        st.subheader("🧠 SỐ CHIẾN LƯỢC (AI)")
        ai_num, score_map = ai_ensemble_pick(df)
        if ai_num=="--":
            st.error("AI từ chối đánh – Cầu xấu")
        else:
            st.success(f"AI đề xuất: **{ai_num}**")

        st.divider()
        st.subheader("📊 THỐNG KÊ")
        st.write(f"• Số kỳ phân tích: **{len(df)}**")
        st.write("• AI dùng ensemble 100 tiêu chí, lọc cứng trước khi đề xuất")

st.caption("NUMCORE v6.6 – AI Ensemble 100 – Ổn định – Không số chập")
