import streamlit as st
import pandas as pd
import re
from collections import Counter
import os

st.set_page_config(page_title="LOTOBET AI – 2 SỐ 5 TINH", layout="centered")

DATA_DIR = "data"
ALL_DATA = f"{DATA_DIR}/data_all.csv"
NEW_DATA = f"{DATA_DIR}/data_new.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- STYLE ----------
st.markdown("""
<style>
body {background:#0e1117;color:white;}
.card {background:#1e1e2f;padding:15px;border-radius:14px;margin-top:12px;}
.num {font-size:30px;color:#00e5ff;font-weight:bold;text-align:center;}
.warn {background:#4b0000;color:#ff4b4b;padding:10px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["pair"])

def save_pairs(pairs):
    df_new = pd.DataFrame(pairs, columns=["pair"])
    df_new.to_csv(NEW_DATA, mode="a", header=not os.path.exists(NEW_DATA), index=False)

    df_all = load_csv(ALL_DATA)
    df_all = pd.concat([df_all, df_new], ignore_index=True)
    df_all.to_csv(ALL_DATA, index=False)

def analyze_top3(df):
    nums = []
    for p in df["pair"]:
        nums.append(p // 10)
        nums.append(p % 10)

    counter = Counter(nums)
    hot = [n for n,_ in counter.most_common(6)]

    return [(hot[i], hot[i+1]) for i in range(0,6,2)]

def detect_bet(df):
    recent = df.tail(10)["pair"]
    c = Counter(recent)
    return [k for k,v in c.items() if v >= 3]

# ---------- UI ----------
st.title("🎯 LOTOBET AI – 2 SỐ 5 TINH")

with st.expander("📥 NẠP KẾT QUẢ 5 TINH"):
    raw = st.text_area("Dán kết quả (vd: 71765 00387 50554)", height=120)
    if st.button("🚀 NẠP DỮ LIỆU"):
        digits = re.findall(r"\d", raw)
        rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
        pairs = [int(r[-2]+r[-1]) for r in rows if len(r)==5]

        if pairs:
            save_pairs(pairs)
            st.success(f"Đã nạp {len(pairs)} kỳ")
        else:
            st.error("Không nhận diện được dữ liệu")

df_all = load_csv(ALL_DATA)
df_new = load_csv(NEW_DATA)

st.info(f"📊 Tổng dữ liệu: {len(df_all)} | 🆕 Mới: {len(df_new)}")

if st.button("🔮 PHÂN TÍCH KỲ TIẾP"):
    if len(df_all) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
    else:
        bet = detect_bet(df_all)
        if bet:
            st.markdown(f"<div class='warn'>🚨 CẦU BỆT: {', '.join(map(str, bet))}</div>", unsafe_allow_html=True)

        top3 = analyze_top3(df_all)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 3 CẶP 2 SỐ 5 TINH MẠNH NHẤT")
        for a,b in top3:
            st.markdown(f"<div class='num'>{a} - {b}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚠️ Công cụ thống kê – không đảm bảo trúng. Quản lý vốn.")
