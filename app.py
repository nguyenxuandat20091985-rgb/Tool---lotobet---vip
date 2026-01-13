import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3 KU",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
AI_FILE = "ai_weight.csv"

# ================== UTIL ==================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_results(results):
    df = load_csv(DATA_FILE, ["time", "number"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{"time": now, "number": n} for n in results])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def load_ai():
    return load_csv(AI_FILE, ["key", "weight"])

def update_ai(key, win):
    ai = load_ai()
    if key not in ai["key"].values:
        ai.loc[len(ai)] = [key, 1.0]

    idx = ai[ai["key"] == key].index[0]
    ai.loc[idx, "weight"] += 0.25 if win else -0.15
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    ai.to_csv(AI_FILE, index=False)

# ================== CHECK WIN ==================
def check_non_fixed(selected, result):
    digits = set(map(int, str(result).zfill(5)))
    return set(selected).issubset(digits)

# ================== ANALYSIS CORE ==================
def analyze_numbers(df):
    numbers = df["number"].astype(str).str.zfill(5)

    digit_counter = Counter()
    pair_counter = Counter()

    for n in numbers:
        ds = list(map(int, n))
        digit_counter.update(ds)
        pair_counter.update([int(n[-2:])])

    return digit_counter, pair_counter

def analyze_non_fixed(df, k):
    freq = Counter()
    numbers = df["number"].astype(str).str.zfill(5)

    for n in numbers:
        for d in set(n):
            freq[d] += 1

    combos = []
    digits = list(range(10))
    from itertools import combinations

    for c in combinations(digits, k):
        hit = 0
        for n in numbers:
            if set(map(int, c)).issubset(set(map(int, n))):
                hit += 1
        rate = hit / len(numbers) * 100
        combos.append({
            "set": c,
            "hits": hit,
            "rate": round(rate, 2)
        })

    return sorted(combos, key=lambda x: x["rate"], reverse=True)

# ================== UI ==================
st.title("🎯 LOTOBET AUTO PRO – V3 (CHUẨN KU)")

with st.expander("📥 NHẬP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area("Dán kết quả (mỗi dòng 1 số 5 tinh)", height=120)
    if st.button("💾 LƯU DỮ LIỆU"):
        nums = re.findall(r"\d{5}", raw)
        if nums:
            save_results(nums)
            st.success(f"Đã lưu {len(nums)} kỳ")
        else:
            st.error("Không nhận diện được số 5 tinh")

df = load_csv(DATA_FILE, ["time", "number"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < 50:
    st.warning("Cần tối thiểu 50 kỳ để phân tích mạnh")
    st.stop()

# ================== TABS ==================
tab1, tab2, tab3 = st.tabs([
    "🔢 HÀNG SỐ 5 TINH",
    "🟢 2 SỐ 5 TINH",
    "🔥 3 SỐ 5 TINH"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("📈 Phân tích HÀNG SỐ 5 TINH (2 số cuối)")

    digit_cnt, pair_cnt = analyze_numbers(df)
    top_pairs = pair_cnt.most_common(10)

    st.table(pd.DataFrame(top_pairs, columns=["Cặp", "Số lần"]))

    best_pair = top_pairs[0][0]
    st.success(f"🎯 KẾT LUẬN: Ưu tiên đánh cặp **{best_pair}**")

# ================== TAB 2 ==================
with tab2:
    st.subheader("🟢 Không cố định – 2 SỐ 5 TINH")

    top2 = analyze_non_fixed(df, 2)[:5]
    df2 = pd.DataFrame(top2)
    df2["Bộ số"] = df2["set"].apply(lambda x: "-".join(map(str, x)))
    st.table(df2[["Bộ số", "hits", "rate"]])

    best2 = top2[0]
    st.success(
        f"🎯 KẾT LUẬN: Đánh **2 số {best2['set']}** | "
        f"Tỉ lệ trúng {best2['rate']}%"
    )

# ================== TAB 3 ==================
with tab3:
    st.subheader("🔥 Không cố định – 3 SỐ 5 TINH")

    top3 = analyze_non_fixed(df, 3)[:5]
    df3 = pd.DataFrame(top3)
    df3["Bộ số"] = df3["set"].apply(lambda x: "-".join(map(str, x)))
    st.table(df3[["Bộ số", "hits", "rate"]])

    best3 = top3[0]
    st.success(
        f"🎯 KẾT LUẬN: Đánh **3 số {best3['set']}** | "
        f"Tỉ lệ trúng {best3['rate']}%"
    )

# ================== FOOTER ==================
st.markdown("---")
st.caption("🚀 LOTOBET AUTO PRO V3 | Phân tích chuẩn KU | Không đoán mò")
