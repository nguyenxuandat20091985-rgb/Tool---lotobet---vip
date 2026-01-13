import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os
from itertools import combinations

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3 KU",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "data.csv"

# ================== DATA CORE ==================
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "result"])

    df = pd.read_csv(DATA_FILE)

    # FIX DATA CŨ (pair → result)
    if "result" not in df.columns and "pair" in df.columns:
        df["result"] = df["pair"].astype(str).str.zfill(5)
        df = df[["time", "result"]]
        df.to_csv(DATA_FILE, index=False)

    return df

def save_results(results):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{"time": now, "result": r} for r in results])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================== ANALYSIS ==================
def analyze_pairs(df):
    pairs = Counter()
    for r in df["result"]:
        r = str(r).zfill(5)
        pairs.update([int(r[-2:])])
    return pairs.most_common(10)

def analyze_non_fixed(df, k):
    results = df["result"].astype(str).str.zfill(5)
    stats = []

    for comb in combinations(range(10), k):
        hit = 0
        for r in results:
            if set(map(str, comb)).issubset(set(r)):
                hit += 1
        rate = hit / len(results) * 100
        stats.append({
            "Bộ số": "-".join(map(str, comb)),
            "Số lần trúng": hit,
            "Tỉ lệ %": round(rate, 2)
        })

    return sorted(stats, key=lambda x: x["Tỉ lệ %"], reverse=True)

# ================== UI ==================
st.title("🎯 LOTOBET AUTO PRO – V3 (CHUẨN KU – FIXED)")

with st.expander("📥 NHẬP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area(
        "Mỗi dòng 1 số 5 tinh (VD: 12864)",
        height=120
    )
    if st.button("💾 LƯU KẾT QUẢ"):
        nums = re.findall(r"\d{5}", raw)
        if nums:
            save_results(nums)
            st.success(f"Đã lưu {len(nums)} kỳ")
        else:
            st.error("Không nhận diện được số 5 tinh")

df = load_data()
st.info(f"📊 Tổng dữ liệu hiện có: {len(df)} kỳ")

if len(df) < 30:
    st.warning("Cần tối thiểu 30 kỳ để phân tích")
    st.stop()

# ================== TABS ==================
tab1, tab2, tab3 = st.tabs([
    "🔢 HÀNG SỐ 5 TINH",
    "🟢 2 SỐ 5 TINH",
    "🔥 3 SỐ 5 TINH"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("📈 HÀNG SỐ 5 TINH (2 SỐ CUỐI)")
    top_pairs = analyze_pairs(df)
    st.table(pd.DataFrame(top_pairs, columns=["Cặp số", "Số lần về"]))

    best = top_pairs[0]
    st.success(f"🎯 KHUYẾN NGHỊ: ĐÁNH CẶP **{best[0]}**")

# ================== TAB 2 ==================
with tab2:
    st.subheader("🟢 KHÔNG CỐ ĐỊNH – 2 SỐ 5 TINH")
    top2 = analyze_non_fixed(df, 2)[:5]
    st.table(pd.DataFrame(top2))

    best = top2[0]
    st.success(
        f"🎯 ĐÁNH 2 SỐ **{best['Bộ số']}** | "
        f"Tỉ lệ {best['Tỉ lệ %']}%"
    )

# ================== TAB 3 ==================
with tab3:
    st.subheader("🔥 KHÔNG CỐ ĐỊNH – 3 SỐ 5 TINH")
    top3 = analyze_non_fixed(df, 3)[:5]
    st.table(pd.DataFrame(top3))

    best = top3[0]
    st.success(
        f"🎯 ĐÁNH 3 SỐ **{best['Bộ số']}** | "
        f"Tỉ lệ {best['Tỉ lệ %']}%"
    )

st.markdown("---")
st.caption("🚀 LOTOBET AUTO PRO V3 | FIXED | Phân tích đúng luật KU")
