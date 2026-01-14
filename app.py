import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET PRO – V3.5",
    layout="centered",
    page_icon="🎯"
)

# ================= STORAGE =================
RAW_FILE = "raw_input.csv"
PAIR2_FILE = "pair_2.csv"
PAIR3_FILE = "pair_3.csv"

# ================= UTIL =================
def load_df(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def next_ky(df):
    return 1 if df.empty else int(df["ky"].astype(int).max()) + 1

# ================= SAVE DATA =================
def save_numbers(numbers):
    raw_df = load_df(RAW_FILE, ["time", "ky", "number5"])
    p2_df = load_df(PAIR2_FILE, ["time", "ky", "pair"])
    p3_df = load_df(PAIR3_FILE, ["time", "ky", "pair"])

    ky = next_ky(raw_df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for num in numbers:
        # bỏ trùng tuyệt đối
        if not raw_df[raw_df["number5"] == num].empty:
            continue

        raw_df.loc[len(raw_df)] = [now, ky, num]

        p2 = num[-2:]
        if p2_df.empty or p2_df.iloc[-1]["pair"] != p2:
            p2_df.loc[len(p2_df)] = [now, ky, p2]

        p3 = num[-3:]
        if p3_df.empty or p3_df.iloc[-1]["pair"] != p3:
            p3_df.loc[len(p3_df)] = [now, ky, p3]

        ky += 1
        added += 1

    raw_df.to_csv(RAW_FILE, index=False)
    p2_df.to_csv(PAIR2_FILE, index=False)
    p3_df.to_csv(PAIR3_FILE, index=False)

    return added

# ================= AI CORE =================
def cycle_note(seq, pair):
    pos = [i for i, p in enumerate(seq) if p == pair]
    if len(pos) < 3:
        return -10, "Thiếu dữ liệu"

    gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
    avg = sum(gaps[-3:]) / len(gaps[-3:])
    last_gap = len(seq) - 1 - pos[-1]

    if abs(last_gap - avg) <= 1:
        return 20, "🎯 Đúng nhịp"
    elif last_gap < avg:
        return -5, "⏳ Vừa ra"
    else:
        return -15, "⚠️ Quá hạn"

def analyze_v35(df, mode="2"):
    total = len(df)
    seq = df["pair"].tolist()
    cnt = Counter(seq)

    rows = []
    for pair, count in cnt.items():
        if mode == "2" and len(pair) != 2:
            continue
        if mode == "3" and len(pair) != 3:
            continue

        # ① TẦN SUẤT
        freq_score = (count / total) * 100

        # ② CHU KỲ
        c_score, c_note = cycle_note(seq, pair)

        # ③ TRÁNH CẦU VỪA RA
        recent_penalty = -20 if seq[-1] == pair else 0

        score = round(freq_score + c_score + recent_penalty, 2)

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": c_note
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    return df_out.sort_values("Điểm AI (%)", ascending=False)

# ================= UI =================
st.title("🎯 LOTOBET PRO – V3.5")

raw = st.text_area(
    "📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)",
    height=120,
    placeholder="Ví dụ:\n46570\n06787\n38527"
)

if st.button("💾 Lưu dữ liệu"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        added = save_numbers(nums)
        st.success(f"✅ Đã lưu {added} kỳ (lọc trùng tự động)")
    else:
        st.error("❌ Sai định dạng dữ liệu")

raw_df = load_df(RAW_FILE, ["time", "ky", "number5"])
st.info(f"📊 Tổng dữ liệu: {len(raw_df)} kỳ")

# ================= TOP 2 TINH =================
st.divider()
st.subheader("🔥 TOP 2 TINH (KHÔNG CỐ ĐỊNH)")

df2 = load_df(PAIR2_FILE, ["time", "ky", "pair"])
if len(df2) < 30:
    st.warning("⚠️ Chưa đủ dữ liệu 2 tinh")
else:
    a2 = analyze_v35(df2, "2")
    st.dataframe(a2.head(5), use_container_width=True, hide_index=True)

# ================= TOP 3 TINH =================
st.divider()
st.subheader("🔥 TOP 3 TINH (KHÔNG CỐ ĐỊNH)")

df3 = load_df(PAIR3_FILE, ["time", "ky", "pair"])
if len(df3) < 30:
    st.warning("⚠️ Chưa đủ dữ liệu 3 tinh")
else:
    a3 = analyze_v35(df3, "3")
    st.dataframe(a3.head(5), use_container_width=True, hide_index=True)

st.caption("⚠️ AI hỗ trợ xác suất – quản lý vốn & kỷ luật là bắt buộc")
