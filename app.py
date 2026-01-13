import streamlit as st
import re
import os
import csv
import itertools
from collections import Counter

# ================== CẤU HÌNH ==================
st.set_page_config(
    page_title="LOTOBET AI – 2 SỐ 5 TINH",
    layout="centered"
)

# ================== CSS ==================
st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-weight: bold;
    background: linear-gradient(45deg, #ff4b4b, #ff7a7a);
    border: none;
}
.card {
    background: rgba(30,30,50,0.85);
    padding: 15px;
    border-radius: 14px;
    margin-bottom: 12px;
    text-align: center;
    border: 1px solid #3a3a5a;
}
.big {
    font-size: 2.3em;
    font-weight: bold;
    color: #00ffcc;
}
.alert {
    background: #4b0000;
    color: #ff9999;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #ff4b4b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================== BỘ NHỚ VĨNH VIỄN ==================
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def load_history():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
        return [list(map(int, row)) for row in csv.reader(f) if len(row) == 5]

def save_history(rows):
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# ================== SESSION ==================
if "history" not in st.session_state:
    st.session_state.history = load_history()

# ================== HEADER ==================
st.title("🎯 LOTOBET AI – 2 SỐ 5 TINH")
st.caption("Tool thống kê – đúng luật – lưu dữ liệu vĩnh viễn")

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ HỆ THỐNG")
    st.write(f"📊 Tổng kỳ đã lưu: **{len(st.session_state.history)}**")
    if st.button("🗑️ RESET TOÀN BỘ"):
        st.session_state.history = []
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.rerun()

# ================== NẠP DỮ LIỆU ==================
with st.expander("📥 NẠP KẾT QUẢ 5 TINH", expanded=True):
    raw = st.text_area(
        "Dán kết quả (vd: 12121 90834 11234 ...)",
        height=120
    )

    if st.button("🚀 NẠP DỮ LIỆU"):
        digits = re.findall(r"\d", raw)
        rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
        rows = [list(map(int, r)) for r in rows if len(r) == 5]

        if rows:
            save_history(rows)
            st.session_state.history.extend(rows)
            st.success(f"✅ Đã nạp {len(rows)} kỳ (lưu vĩnh viễn)")
            st.rerun()
        else:
            st.warning("❌ Không phát hiện dữ liệu hợp lệ")

# ================== THUẬT TOÁN ==================
def analyze(history):
    if len(history) < 10:
        return None

    # ---- 2 SỐ 5 TINH (ĐÚNG LUẬT) ----
    pair_counter = Counter()
    for row in history:
        for a, b in itertools.combinations(set(row), 2):
            pair_counter[tuple(sorted((a, b)))] += 1

    top_pairs = pair_counter.most_common(3)

    # ---- CẦU BỆT LIÊN TIẾP ----
    streak = Counter()
    last = history[-6:]
    for i in range(1, len(last)):
        for n in set(last[i]) & set(last[i-1]):
            streak[n] += 1
    bet_nums = [n for n, c in streak.items() if c >= 2]

    # ---- TAM THỦ ----
    all_nums = [n for row in history for n in row]
    top_3 = [n for n, _ in Counter(all_nums).most_common(3)]

    return top_pairs, bet_nums, top_3

# ================== PHÂN TÍCH ==================
if st.session_state.history:
    if st.button("🔮 PHÂN TÍCH KỲ TIẾP"):
        result = analyze(st.session_state.history)

        if not result:
            st.warning("⚠️ Cần ít nhất 10 kỳ để phân tích chính xác")
        else:
            pairs, bet_nums, top_3 = result

            if bet_nums:
                st.markdown(
                    f"<div class='alert'>🚨 CẦU BỆT: {', '.join(map(str, bet_nums))}</div>",
                    unsafe_allow_html=True
                )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 2 SỐ 5 TINH")
                if pairs:
                    a, b = pairs[0][0]
                    st.markdown(
                        f"<div class='card'><div class='big'>{a} - {b}</div><small>Cặp xuất hiện nhiều nhất</small></div>",
                        unsafe_allow_html=True
                    )

            with col2:
                st.subheader("🌟 TAM THỦ 3 TINH")
                st.markdown(
                    f"<div class='card'><div class='big'>{''.join(map(str, top_3))}</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.info("💡 Gợi ý: Đánh 1–2 cặp / kỳ. Thua 2 kỳ nên dừng.")

st.caption("© LOTOBET AI – Công cụ thống kê, không cam kết trúng")
