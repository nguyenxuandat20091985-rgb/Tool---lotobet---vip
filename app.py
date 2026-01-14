import streamlit as st
import pandas as pd
import os, re, json
from datetime import datetime
from itertools import combinations
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3",
    layout="wide",
    page_icon="🎯"
)

RESULT_FILE = "results.csv"
SESSION_FILE = "sessions.csv"
WIN_FILE = "wins.csv"
STATE_FILE = "state.json"

MIN_DATA = 30
BET_THRESHOLD = 5  # số kỳ bệt để canh đánh

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE, "r"))
    return {"current_set": [], "type": "", "ky": 0}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

# ================= DATA INPUT =================
def save_results(results):
    df = load_csv(RESULT_FILE, ["ky", "time", "result"])
    last_ky = df["ky"].max() if not df.empty else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_rows = []
    for r in results:
        if df.empty or r not in df["result"].values:
            last_ky += 1
            new_rows.append({"ky": last_ky, "time": now, "result": r})

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_csv(df, RESULT_FILE)
    return len(new_rows)

# ================= ANALYSIS =================
def streak_analysis(df):
    results = df["result"].astype(str).str.zfill(5)
    streak = {}
    for n in "0123456789":
        miss = 0
        for r in reversed(results):
            if n in r:
                break
            miss += 1
        streak[n] = miss
    return streak

def analyze_pairs(df):
    pairs = Counter()
    for r in df["result"]:
        r = str(r).zfill(5)
        pairs.update([r[-2:]])
    return pairs

def analyze_non_fixed(df, k):
    results = df["result"].astype(str).str.zfill(5)
    stats = []

    for combo in combinations("0123456789", k):
        combo = tuple(sorted(combo))
        hit = sum(1 for r in results if set(combo).issubset(set(r)))
        rate = round(hit / len(results) * 100, 2)

        stats.append({
            "Bộ số": "-".join(combo),
            "Số lần trúng": hit,
            "Tỉ lệ %": rate
        })

    return sorted(stats, key=lambda x: x["Tỉ lệ %"], reverse=True)

def confidence_score(rate, streaks, combo):
    nums = combo.split("-")
    biet = sum(1 for n in nums if streaks[n] >= BET_THRESHOLD)
    score = rate + biet * 5
    return min(round(score, 2), 99)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3 (THEO KỲ – THỰC CHIẾN)")

# ===== INPUT =====
with st.expander("📥 NHẬP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area("Mỗi dòng 1 số (VD: 12864)", height=100)
    if st.button("💾 LƯU KẾT QUẢ"):
        nums = re.findall(r"\d{5}", raw)
        if nums:
            added = save_results(nums)
            st.success(f"Đã thêm {added} kỳ mới")
        else:
            st.error("Dữ liệu không hợp lệ")

df = load_csv(RESULT_FILE, ["ky", "time", "result"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để phân tích")
    st.stop()

# ===== STREAK =====
st.subheader("🔥 THEO DÕI BỆT SỐ")
streaks = streak_analysis(df)
st.table(pd.DataFrame([
    {
        "Số": k,
        "Số kỳ chưa ra": v,
        "Trạng thái": "🔥 SẮP BẬT" if v >= BET_THRESHOLD else "🟢 BÌNH THƯỜNG"
    } for k, v in streaks.items()
]))

# ===== TABS =====
tab1, tab2, tab3 = st.tabs([
    "🔢 HÀNG SỐ (2 SỐ CUỐI)",
    "🟢 2 SỐ 5 TINH",
    "🔥 3 SỐ 5 TINH"
])

with tab1:
    pairs = analyze_pairs(df)
    top_pairs = pairs.most_common(10)
    st.table(pd.DataFrame(top_pairs, columns=["Hàng số", "Số lần về"]))

with tab2:
    top2 = analyze_non_fixed(df, 2)[:10]
    st.table(top2)

with tab3:
    top3 = analyze_non_fixed(df, 3)[:10]
    for x in top3:
        x["Confidence"] = confidence_score(x["Tỉ lệ %"], streaks, x["Bộ số"])
    st.table(top3)

# ===== SUGGEST =====
st.subheader("🚦 ĐỀ XUẤT KỲ TIẾP THEO (AN TOÀN)")
safe = [x for x in top3 if x.get("Confidence", 0) >= 70][:3]
st.table(safe)

state = load_state()

if st.button("📌 CHỐT BỘ SỐ KỲ TỚI"):
    if safe:
        state["current_set"] = [safe[0]["Bộ số"]]
        state["type"] = "3 số 5 tinh"
        state["ky"] = int(df["ky"].max()) + 1
        save_state(state)
        st.success("Đã chốt bộ số kỳ tiếp theo")

# ===== SESSION =====
st.subheader("📊 THEO DÕI KỲ ĐANG ĐÁNH")
state = load_state()
st.info(f"Kỳ: {state['ky']} | Bộ số: {state['current_set']} | Loại: {state['type']}")

if st.button("✅ TRÚNG"):
    win_df = load_csv(WIN_FILE, ["time", "ky", "combo", "type"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in state["current_set"]:
        win_df.loc[len(win_df)] = [now, state["ky"], c, state["type"]]
    save_csv(win_df, WIN_FILE)
    st.success("Đã ghi nhận kỳ TRÚNG")

if st.button("❌ THUA – RESET"):
    save_state({"current_set": [], "type": "", "ky": state["ky"]})
    st.warning("Đã reset bộ số – phân tích lại")

# ===== HISTORY =====
st.subheader("🏆 LỊCH SỬ THẮNG")
win_df = load_csv(WIN_FILE, ["time", "ky", "combo", "type"])
if not win_df.empty:
    st.table(win_df.tail(10))

st.caption("🚀 LOTOBET AUTO PRO V3 | Đánh theo kỳ – Có kỷ luật – Không đoán mò")
