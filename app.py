import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import Counter
import os

st.set_page_config("LOTOBET V3.5 – Không cố định 2–3 tinh", layout="centered")

DATA_FILE = "results.csv"
MIN_DATA = 30

# ================= DATA =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["result"])

def save_results(nums):
    df = load_data()
    new = pd.DataFrame({"result": nums})
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def result_to_set(x):
    return set(str(x).zfill(5))

# ================= CORE =================
def build_sets(df):
    return [result_to_set(x) for x in df["result"]]

def generate_candidates(size):
    digits = list("0123456789")
    return [set(c) for c in combinations(digits, size)]

def check_hit(open_set, bet_set):
    return bet_set.issubset(open_set)

def analyze(size, df, lookback=30):
    opens = build_sets(df)
    candidates = generate_candidates(size)
    stats = []

    for c in candidates:
        hits = 0
        last_hit = None

        for i in range(len(opens)-1, max(-1, len(opens)-lookback-1), -1):
            if check_hit(opens[i], c):
                hits += 1
                if last_hit is None:
                    last_hit = len(opens)-1 - i

        rate = round(hits / lookback * 100, 2)
        cycle = last_hit if last_hit is not None else 999

        if cycle <= 1:
            status = "⏳ Vừa ra"
        elif rate >= 25:
            status = "🔥 Đang chạy"
        elif cycle >= 15:
            status = "❄️ Lạnh"
        else:
            status = "⚠️ Theo dõi"

        stats.append({
            "Bộ số": ",".join(sorted(c)),
            "Tỷ lệ %": rate,
            "Chu kỳ": cycle,
            "Trạng thái": status
        })

    return sorted(stats, key=lambda x: (-x["Tỷ lệ %"], x["Chu kỳ"]))

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3.5")

raw = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)")
if st.button("💾 Lưu dữ liệu"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        save_results(nums)
        st.success(f"Đã lưu {len(nums)} kỳ")
    else:
        st.error("Sai định dạng")

df = load_data()
st.info(f"Tổng dữ liệu: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để phân tích")
    st.stop()

st.divider()

# ===== 2 TINH =====
st.subheader("🔢 TOP 2 TINH (KHÔNG CỐ ĐỊNH)")
res2 = analyze(2, df)
st.table(res2[:5])

best2 = res2[0]
st.markdown(f"""
**Đề xuất:** `{best2['Bộ số']}`  
**Tỷ lệ:** `{best2['Tỷ lệ %']}%`  
**Chu kỳ:** `{best2['Chu kỳ']}`  
**Trạng thái:** {best2['Trạng thái']}
""")

st.divider()

# ===== 3 TINH =====
st.subheader("🔢 TOP 3 TINH (KHÔNG CỐ ĐỊNH)")
res3 = analyze(3, df)
st.table(res3[:5])

best3 = res3[0]
st.markdown(f"""
**Đề xuất:** `{best3['Bộ số']}`  
**Tỷ lệ:** `{best3['Tỷ lệ %']}%`  
**Chu kỳ:** `{best3['Chu kỳ']}`  
**Trạng thái:** {best3['Trạng thái']}
""")

st.caption("⚠️ Tool hỗ trợ xác suất – đánh phải có kỷ luật")
