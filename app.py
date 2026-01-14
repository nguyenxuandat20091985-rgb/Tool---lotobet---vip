import streamlit as st
import pandas as pd
import re, os
from collections import Counter
from datetime import datetime
import time

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.7",
    page_icon="🎯",
    layout="centered"
)

RAW_FILE = "raw_5so.csv"
PAIR2_FILE = "pair_2tinh.csv"
PAIR3_FILE = "pair_3tinh.csv"
MEMORY_FILE = "ai_memory.csv"

MIN_DATA = 40

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
    if df.empty:
        return 1
    return int(df["ky"].astype(int).max()) + 1

# ================= SAVE DATA =================
def save_input(numbers):
    raw = load_df(RAW_FILE, ["time","ky","number5"])
    p2  = load_df(PAIR2_FILE, ["time","ky","pair"])
    p3  = load_df(PAIR3_FILE, ["time","ky","pair"])

    ky = next_ky(raw)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for num in numbers:
        if not raw[raw["number5"] == num].empty:
            continue

        raw.loc[len(raw)] = [now, ky, num]

        pair2 = num[-2:]
        pair3 = num[-3:]

        if p2.empty or p2.iloc[-1]["pair"] != pair2:
            p2.loc[len(p2)] = [now, ky, pair2]

        if p3.empty or p3.iloc[-1]["pair"] != pair3:
            p3.loc[len(p3)] = [now, ky, pair3]

        ky += 1
        added += 1

    raw.to_csv(RAW_FILE, index=False)
    p2.to_csv(PAIR2_FILE, index=False)
    p3.to_csv(PAIR3_FILE, index=False)
    return added

# ================= AI MEMORY =================
def load_memory():
    return load_df(MEMORY_FILE, ["pair","hit","miss","score"])

def update_memory(pair, hit):
    mem = load_memory()
    if pair not in mem["pair"].values:
        mem.loc[len(mem)] = [pair,0,0,50]

    idx = mem[mem["pair"] == pair].index[0]
    if hit:
        mem.loc[idx,"hit"] = int(mem.loc[idx,"hit"]) + 1
        mem.loc[idx,"score"] = min(100, int(mem.loc[idx,"score"]) + 5)
    else:
        mem.loc[idx,"miss"] = int(mem.loc[idx,"miss"]) + 1
        mem.loc[idx,"score"] = max(0, int(mem.loc[idx,"score"]) - 7)

    mem.to_csv(MEMORY_FILE, index=False)

# ================= AI CORE =================
def cycle_eval(seq, pair):
    pos = [i for i,v in enumerate(seq) if v == pair]
    if len(pos) < 3:
        return -10, "Thiếu dữ liệu"
    gaps = [pos[i]-pos[i-1] for i in range(1,len(pos))]
    avg = sum(gaps[-3:]) / len(gaps[-3:])
    last = len(seq)-1-pos[-1]

    if abs(last-avg) <= 1:
        return 20, "🎯 Đúng nhịp"
    if last < avg:
        return -10, "⏳ Vừa ra"
    return -15, "⚠️ Quá hạn"

def analyze(pair_file):
    df = load_df(pair_file, ["time","ky","pair"])
    if len(df) < MIN_DATA:
        return pd.DataFrame()

    seq = df["pair"].tolist()
    cnt10 = Counter(seq[-10:])
    cnt20 = Counter(seq[-20:])
    cnt_all = Counter(seq)

    mem = load_memory()
    mem_map = dict(zip(mem["pair"], mem["score"].astype(int)))

    rows = []
    for p in cnt_all:
        freq = ((cnt10[p]/10)*0.4 + (cnt20[p]/20)*0.3 + (cnt_all[p]/len(seq))*0.3)*100
        c_score, note = cycle_eval(seq, p)
        mem_score = mem_map.get(p, 50)

        total = round(freq + c_score + (mem_score-50)*0.6,2)

        if total <= 0:
            continue

        decision = "🔴 CẤM ĐÁNH"
        if total >= 70 and mem_score >= 55:
            decision = "🟢 NÊN ĐÁNH"
        elif total >= 50:
            decision = "🟡 THEO DÕI"

        rows.append({
            "Cặp": p,
            "Điểm AI (%)": total,
            "Cầu": note,
            "Memory": mem_score,
            "Khuyến nghị": decision
        })

    out = pd.DataFrame(rows)
    return out.sort_values("Điểm AI (%)", ascending=False)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.7")

raw = st.text_area(
    "📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)",
    height=120
)

if st.button("💾 LƯU & PHÂN TÍCH"):
    nums = re.findall(r"\d{5}", raw)
    if not nums:
        st.error("Sai định dạng dữ liệu")
    else:
        with st.spinner("⏳ ĐANG PHÂN TÍCH DỮ LIỆU..."):
            time.sleep(1)
            added = save_input(nums)
        st.success(f"Đã lưu {added} kỳ hợp lệ")

raw_df = load_df(RAW_FILE, ["time","ky","number5"])
st.info(f"📊 Tổng dữ liệu: {len(raw_df)} kỳ")

if len(raw_df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI hoạt động")
    st.stop()

# ================= 2 TỈNH =================
st.divider()
st.subheader("🔥 TOP 2 TỈNH")

analysis2 = analyze(PAIR2_FILE)
st.dataframe(analysis2.head(5), use_container_width=True, hide_index=True)

best2 = analysis2.iloc[0]
st.markdown(f"""
### 🧠 KẾT LUẬN 2 TỈNH
- 🎯 **Cặp:** `{best2['Cặp']}`
- 📊 **Điểm AI:** `{best2['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best2['Cầu']}
- 🧠 **Memory:** `{best2['Memory']}`
- 🚦 **{best2['Khuyến nghị']}**
""")

# ================= 3 TỈNH =================
st.divider()
st.subheader("🔥 TOP 3 TỈNH")

analysis3 = analyze(PAIR3_FILE)
st.dataframe(analysis3.head(5), use_container_width=True, hide_index=True)

best3 = analysis3.iloc[0]
st.markdown(f"""
### 🧠 KẾT LUẬN 3 TỈNH
- 🎯 **Cặp:** `{best3['Cặp']}`
- 📊 **Điểm AI:** `{best3['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best3['Cầu']}
- 🧠 **Memory:** `{best3['Memory']}`
- 🚦 **{best3['Khuyến nghị']}**
""")

# ================= MEMORY FEEDBACK =================
st.divider()
st.subheader("🧾 GHI NHẬN KẾT QUẢ AI")

c1, c2 = st.columns(2)
with c1:
    if st.button("✅ TRÚNG 2 TỈNH"):
        update_memory(best2["Cặp"], True)
        st.success("AI đã học (2 tỉnh)")
with c2:
    if st.button("❌ TRƯỢT 2 TỈNH"):
        update_memory(best2["Cặp"], False)
        st.warning("AI đã học (2 tỉnh)")

st.caption("⚠️ AI hỗ trợ xác suất – kỷ luật & quản lý vốn là bắt buộc")
