import streamlit as st
import pandas as pd
import re, os, json
from datetime import datetime
from itertools import combinations
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO V3.5",
    layout="wide",
    page_icon="🎯"
)

RESULT_FILE = "results.csv"
SESSION_FILE = "sessions.csv"
WIN_FILE = "wins.csv"
STATE_FILE = "state.json"

# ================= CORE DATA =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"current_set": [], "type": ""}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)

# ================= SAVE RESULT =================
def save_result(numbers):
    df = load_csv(RESULT_FILE, ["time", "result"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{"time": now, "result": n} for n in numbers])
    save_csv(pd.concat([df, new]), RESULT_FILE)

# ================= STREAK ANALYSIS =================
def analyze_streaks(df):
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

# ================= NON FIXED ANALYSIS =================
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

# ================= CHECK WIN =================
def check_win(combo, result):
    return set(combo.split("-")).issubset(set(result))

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3.5 (THỰC CHIẾN KU)")

# ================= INPUT =================
with st.expander("📥 NHẬP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area("Mỗi dòng 1 số (VD: 12864)", height=100)
    if st.button("💾 LƯU KẾT QUẢ"):
        nums = re.findall(r"\d{5}", raw)
        if nums:
            save_result(nums)
            st.success(f"Đã lưu {len(nums)} kỳ")
        else:
            st.error("Không hợp lệ")

df = load_csv(RESULT_FILE, ["time", "result"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < 30:
    st.warning("Cần ít nhất 30 kỳ để phân tích")
    st.stop()

# ================= STREAK =================
streaks = analyze_streaks(df)

st.subheader("🔥 THEO DÕI BỆT SỐ")
st.table(pd.DataFrame([
    {"Số": k, "Số kỳ chưa ra": v,
     "Trạng thái": "🔥 SẮP BẬT" if v >= 5 else "🟢 BÌNH THƯỜNG"}
    for k, v in streaks.items()
]))

# ================= ANALYSIS =================
tab1, tab2 = st.tabs(["🟢 2 SỐ 5 TINH", "🔥 3 SỐ 5 TINH"])

with tab1:
    top2 = analyze_non_fixed(df, 2)[:10]
    st.table(top2)

with tab2:
    top3 = analyze_non_fixed(df, 3)[:10]
    st.table(top3)

# ================= SUGGEST NEXT =================
def suggest(stats):
    sug = []
    for s in stats:
        nums = s["Bộ số"].split("-")
        hot = sum(1 for n in nums if streaks[n] >= 5)
        if hot >= 1:
            sug.append(s)
    return sug[:3]

st.subheader("🚦 ĐỀ XUẤT KỲ TIẾP THEO")

suggest_3 = suggest(top3)
st.table(suggest_3)

state = load_state()

if st.button("📌 CHỌN BỘ SỐ ĐÁNH KỲ TỚI"):
    if suggest_3:
        state["current_set"] = [suggest_3[0]["Bộ số"]]
        state["type"] = "3 số 5 tinh"
        save_state(state)
        st.success("Đã chọn bộ số đánh")

# ================= SESSION TRACK =================
st.subheader("📊 THEO DÕI KỲ")

state = load_state()
st.info(f"🎯 Bộ đang đánh: {state['current_set']} | Loại: {state['type']}")

if st.button("✅ XÁC NHẬN TRÚNG"):
    win_df = load_csv(WIN_FILE, ["time", "combo", "type"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in state["current_set"]:
        win_df.loc[len(win_df)] = [now, c, state["type"]]
    save_csv(win_df, WIN_FILE)
    st.success("Đã ghi nhận TRÚNG")

if st.button("❌ THUA – RESET BỘ SỐ"):
    save_state({"current_set": [], "type": ""})
    st.warning("Đã reset bộ số")

# ================= HISTORY =================
st.subheader("🏆 LỊCH SỬ TRÚNG")
win_df = load_csv(WIN_FILE, ["time", "combo", "type"])
if not win_df.empty:
    st.table(win_df.tail(10))

st.caption("🚀 LOTOBET AUTO PRO V3.5 | Chơi theo kỳ – Không đoán mò")
