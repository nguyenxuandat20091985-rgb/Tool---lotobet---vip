import streamlit as st
import re
from collections import Counter, defaultdict

st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TINH v6.6",
    layout="centered"
)

# ================== SESSION ==================
if "history" not in st.session_state:
    st.session_state.history = []  # mỗi phần tử là list 5 số
if "pair_stat" not in st.session_state:
    st.session_state.pair_stat = defaultdict(lambda: {
        "hit": 0,
        "miss": 0,
        "last_hit": "-"
    })

# ================== FUNCTIONS ==================
def extract_5_digits(text):
    nums = re.findall(r"\d", text)
    results = []
    for i in range(0, len(nums), 5):
        if len(nums[i:i+5]) == 5:
            results.append(nums[i:i+5])
    return results

def normalize_confidence(raw, miss):
    penalty = max(0, miss - 3) * 6
    score = raw - penalty
    return round(max(50, min(score, 88)), 1)

def analyze_pairs(history, top_n=10):
    digit_freq = Counter()
    pair_freq = Counter()

    for row in history:
        for d in row:
            digit_freq[d] += 1
        for i in range(5):
            for j in range(i + 1, 5):
                pair = "".join(sorted([row[i], row[j]]))
                pair_freq[pair] += 1

    results = []
    for pair, freq in pair_freq.most_common(top_n * 2):
        stat = st.session_state.pair_stat[pair]
        raw = freq * 4 + stat["hit"] * 6 - stat["miss"] * 2
        conf = normalize_confidence(raw, stat["miss"])

        results.append({
            "pair": pair,
            "confidence": conf,
            "hit": stat["hit"],
            "miss": stat["miss"],
            "last_hit": stat["last_hit"]
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_n]

def update_stats(prev_row, new_index):
    if not prev_row:
        return
    prev_pairs = set()
    for i in range(5):
        for j in range(i + 1, 5):
            prev_pairs.add("".join(sorted([prev_row[i], prev_row[j]])))

    for pair in st.session_state.pair_stat:
        if pair in prev_pairs:
            st.session_state.pair_stat[pair]["hit"] += 1
            st.session_state.pair_stat[pair]["last_hit"] = f"Kỳ {new_index}"
        else:
            st.session_state.pair_stat[pair]["miss"] += 1

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Quản lý dữ liệu",
    "🤖 10 SỐ AI",
    "🎯 ƯU TIÊN CAO",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    raw = st.text_area("Dán kết quả (mỗi kỳ 5 số)", height=150)

    if st.button("💾 LƯU DỮ LIỆU"):
        rows = extract_5_digits(raw)
        if rows:
            for row in rows:
                prev = st.session_state.history[-1] if st.session_state.history else None
                st.session_state.history.append(row)
                update_stats(prev, len(st.session_state.history))
            st.success(f"Đã lưu {len(rows)} kỳ")
        else:
            st.warning("Không nhận diện được dữ liệu")

    if st.button("🗑️ XÓA SẠCH"):
        st.session_state.history.clear()
        st.session_state.pair_stat.clear()
        st.warning("Đã xóa toàn bộ dữ liệu")

    st.info(f"Tổng số kỳ: {len(st.session_state.history)}")

# ================== TAB 2 ==================
with tab2:
    if len(st.session_state.history) < 5:
        st.warning("Cần tối thiểu 5 kỳ")
    else:
        results = analyze_pairs(st.session_state.history, 10)
        for r in results:
            st.markdown(f"""
            <div style="
                background:#111;
                border-radius:14px;
                padding:14px;
                margin-bottom:12px;
                text-align:center;
                border:1px solid #20c997;">
                <div style="font-size:34px;color:#20c997;font-weight:700;">
                    {r['pair']}
                </div>
                <div style="color:#f1c40f;">
                    {r['confidence']}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================== TAB 3 ==================
with tab3:
    if len(st.session_state.history) < 5:
        st.warning("Chưa đủ dữ liệu")
    else:
        results = analyze_pairs(st.session_state.history, 10)
        best = results[0]

        st.markdown(f"""
        <div style="
            background:#0f5132;
            border-radius:20px;
            padding:26px;
            text-align:center;
            border:3px solid #20c997;">
            <div style="font-size:56px;color:#ff4d4d;font-weight:900;">
                {best['pair']}
            </div>
            <div style="font-size:24px;color:#ffd43b;">
                Tỷ lệ thắng: {best['confidence']}%
            </div>
            <div style="color:#ffffffcc;">
                Trúng: {best['hit']} | Trượt: {best['miss']} | {best['last_hit']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================== TAB 4 ==================
with tab4:
    if not st.session_state.pair_stat:
        st.info("Chưa có thống kê")
    else:
        for pair, stat in st.session_state.pair_stat.items():
            st.write(
                f"{pair} | Trúng: {stat['hit']} | Trượt: {stat['miss']} | {stat['last_hit']}"
            )

st.caption("⚠️ Công cụ hỗ trợ phân tích – không cam kết 100% thắng")
