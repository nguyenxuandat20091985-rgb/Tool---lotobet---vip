import streamlit as st
import re
from collections import Counter, defaultdict
from datetime import datetime

st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TINH v6.6",
    layout="centered"
)

# ================== SESSION ==================
if "history" not in st.session_state:
    st.session_state.history = []   # list of list 5 digits
if "pair_stat" not in st.session_state:
    st.session_state.pair_stat = defaultdict(lambda: {"hit": 0, "miss": 0})

# ================== FUNCTIONS ==================
def extract_5_digits(text):
    nums = re.findall(r"\d", text)
    results = []
    for i in range(0, len(nums), 5):
        if len(nums[i:i+5]) == 5:
            results.append(nums[i:i+5])
    return results

def normalize_confidence(raw_score, miss_count):
    penalty = max(0, miss_count - 3) * 5
    score = raw_score - penalty
    score = max(45, min(score, 88))
    return round(score, 1)

def classify_pair(conf, miss):
    if conf >= 75 and 2 <= miss <= 5:
        return "HOT"
    elif conf >= 60:
        return "WATCH"
    else:
        return "SKIP"

def analyze_pairs(history):
    digit_freq = Counter()
    pair_freq = Counter()

    for row in history:
        for d in row:
            digit_freq[d] += 1
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pair = "".join(sorted([row[i], row[j]]))
                pair_freq[pair] += 1

    results = []
    for pair, freq in pair_freq.most_common(20):
        hit = st.session_state.pair_stat[pair]["hit"]
        miss = st.session_state.pair_stat[pair]["miss"]

        raw_score = freq * 3 + hit * 5 - miss * 2
        conf = normalize_confidence(raw_score, miss)
        ptype = classify_pair(conf, miss)

        results.append({
            "pair": pair,
            "confidence": conf,
            "miss": miss,
            "type": ptype
        })

    return results[:6]

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🤖 Dự đoán AI",
    "🎯 CẦN ĐÁNH",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Dán dữ liệu kết quả (mỗi kỳ 5 số)")
    raw = st.text_area("Ví dụ: 15406 92831 40672 ...", height=150)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU DỮ LIỆU"):
            rows = extract_5_digits(raw)
            if rows:
                st.session_state.history.extend(rows)
                st.success(f"Đã lưu {len(rows)} kỳ")
            else:
                st.warning("Không nhận diện được dữ liệu")

    with col2:
        if st.button("🗑️ XÓA SẠCH"):
            st.session_state.history = []
            st.session_state.pair_stat = defaultdict(lambda: {"hit": 0, "miss": 0})
            st.warning("Đã xóa toàn bộ dữ liệu")

    st.info(f"Tổng số kỳ đang có: {len(st.session_state.history)}")

# ================== TAB 2 ==================
with tab2:
    if len(st.session_state.history) < 5:
        st.warning("Cần ít nhất 5 kỳ để phân tích")
    else:
        results = analyze_pairs(st.session_state.history)
        for r in results:
            st.markdown(f"""
            <div style="
                background:#111;
                border-radius:16px;
                padding:20px;
                margin-bottom:15px;
                text-align:center;
                border:2px solid #20c997;">
                <div style="font-size:44px;color:#20c997;font-weight:800;">
                    {r['pair']}
                </div>
                <div style="font-size:18px;color:#f1c40f;">
                    Tin cậy: {r['confidence']}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================== TAB 3 ==================
with tab3:
    if len(st.session_state.history) < 5:
        st.warning("Chưa đủ dữ liệu")
    else:
        results = analyze_pairs(st.session_state.history)
        hot_pairs = [p for p in results if p["type"] == "HOT"]

        if not hot_pairs:
            st.warning("⏳ Chưa có cầu đủ điều kiện đánh mạnh")
        else:
            for p in hot_pairs:
                st.markdown(f"""
                <div style="
                    background:#0f5132;
                    border-radius:18px;
                    padding:22px;
                    margin-bottom:18px;
                    text-align:center;
                    border:2px solid #20c997;">
                    <div style="font-size:50px;color:#ff4d4d;font-weight:900;">
                        {p['pair']}
                    </div>
                    <div style="font-size:22px;color:#ffd43b;">
                        Tỷ lệ thắng: {p['confidence']}%
                    </div>
                    <div style="color:#ffffffcc;">
                        Treo {p['miss']} kỳ – Ưu tiên kỳ tới ⚠️
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ================== TAB 4 ==================
with tab4:
    st.subheader("Theo dõi cầu")
    if not st.session_state.pair_stat:
        st.info("Chưa có thống kê")
    else:
        for pair, stat in st.session_state.pair_stat.items():
            st.write(
                f"{pair} | Trúng: {stat['hit']} | Trượt: {stat['miss']}"
            )

st.caption("⚠️ Công cụ hỗ trợ phân tích – không đảm bảo 100% thắng")
