import streamlit as st
import re
import os
import csv
import itertools
from collections import Counter

# ================== CONFIG ==================
st.set_page_config(page_title="LOTOBET AI – 2 SỐ 5 TINH", layout="centered")

# ================== CSS ==================
st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    font-weight: bold;
    background: linear-gradient(45deg, #ff4b4b, #ff7a7a);
    border: none;
}
.card {
    background: rgba(30,30,50,0.85);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    text-align: center;
    border: 1px solid #3a3a5a;
}
.big {
    font-size: 2.1em;
    font-weight: bold;
    color: #00ffcc;
}
.alert {
    background: #4b0000;
    color: #ffb3b3;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #ff4b4b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================== DATA MEMORY ==================
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def load_history():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [list(map(int, row)) for row in csv.reader(f) if len(row) == 5]

def save_history(rows):
    with open(DATA_FILE, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

# ================== SESSION ==================
if "history" not in st.session_state:
    st.session_state.history = load_history()

# ================== HEADER ==================
st.title("🎯 LOTOBET AI – 2 SỐ 5 TINH")
st.caption("Phân tích cặp mạnh nhất – đánh 3 cặp / kỳ")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ HỆ THỐNG")
    st.write(f"📊 Tổng kỳ đã lưu: **{len(st.session_state.history)}**")
    if st.button("🗑️ RESET TOÀN BỘ"):
        st.session_state.history = []
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.rerun()

# ================== INPUT ==================
with st.expander("📥 NẠP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area("Dán kết quả (vd: 12121 90834 ...)", height=120)
    if st.button("🚀 NẠP DỮ LIỆU"):
        digits = re.findall(r"\d", raw)
        rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
        rows = [list(map(int, r)) for r in rows if len(r) == 5]
        if rows:
            save_history(rows)
            st.session_state.history.extend(rows)
            st.success(f"Đã nạp {len(rows)} kỳ (lưu vĩnh viễn)")
            st.rerun()
        else:
            st.warning("Không có dữ liệu hợp lệ")

# ================== CORE LOGIC ==================
def analyze_2_tinh(history):
    if len(history) < 15:
        return None

    # --- ĐẾM CẶP 2 SỐ ---
    pair_counter = Counter()
    for row in history:
        for a, b in itertools.combinations(set(row), 2):
            pair_counter[tuple(sorted((a, b)))] += 1

    # --- CẦU BỆT ---
    streak = Counter()
    last = history[-6:]
    for i in range(1, len(last)):
        for n in set(last[i]) & set(last[i-1]):
            streak[n] += 1
    bet_nums = {n for n, c in streak.items() if c >= 2}

    # --- CHẤM ĐIỂM CẶP ---
    scored = []
    for (a, b), cnt in pair_counter.items():
        score = cnt
        if a in bet_nums or b in bet_nums:
            score += 3   # ưu tiên có số bệt
        scored.append(((a, b), score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # --- CHỌN 3 CẶP ÍT TRÙNG ---
    selected = []
    used_nums = set()

    for pair, _ in scored:
        if len(selected) == 3:
            break
        if pair[0] not in used_nums or pair[1] not in used_nums:
            selected.append(pair)
            used_nums.update(pair)

    return selected, bet_nums

# ================== OUTPUT ==================
if st.session_state.history:
    if st.button("🔮 PHÂN TÍCH KỲ TIẾP"):
        result = analyze_2_tinh(st.session_state.history)

        if not result:
            st.warning("⚠️ Cần ít nhất 15 kỳ để phân tích chính xác")
        else:
            pairs, bet_nums = result

            if bet_nums:
                st.markdown(
                    f"<div class='alert'>🚨 CẦU BỆT: {', '.join(map(str, bet_nums))}</div>",
                    unsafe_allow_html=True
                )

            st.subheader("🎯 3 CẶP 2 SỐ 5 TINH MẠNH NHẤT")

            for p in pairs:
                st.markdown(
                    f"<div class='card'><div class='big'>{p[0]} - {p[1]}</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.info("💡 Gợi ý: Đánh tối đa 3 cặp / kỳ. Thua 2 kỳ liên tiếp nên dừng.")

st.caption("© LOTOBET AI – Công cụ thống kê, không cam kết trúng")
