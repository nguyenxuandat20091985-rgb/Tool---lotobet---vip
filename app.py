import streamlit as st
import re
import itertools
import random
from collections import Counter

# ================== CẤU HÌNH ==================
st.set_page_config(page_title="LOTOBET 2 SỐ 5 TINH v6.6", layout="centered")

# Thu nhỏ TAB cho Android
st.markdown("""
<style>
div[data-baseweb="tab-list"] button {
    font-size: 13px;
    padding: 6px 10px;
}
</style>
""", unsafe_allow_html=True)

# ================== SESSION ==================
if "data_5so" not in st.session_state:
    st.session_state.data_5so = []

if "predict_log" not in st.session_state:
    st.session_state.predict_log = []  # lưu lịch sử dự đoán

if "pair_state" not in st.session_state:
    st.session_state.pair_state = {}  # theo dõi cầu treo

# ================== CORE ==================
def extract_5_digits(text):
    digits = re.findall(r"\d", text)
    return [digits[i:i+5] for i in range(0, len(digits)-4, 5)]

def is_win(pair, kq5):
    return pair[0] in kq5 and pair[1] in kq5

def calc_freq(data):
    return Counter(itertools.chain.from_iterable(data))

def score_pair(pair, freq_all, freq_recent):
    base = freq_all[pair[0]] + freq_all[pair[1]]
    recent = freq_recent[pair[0]] + freq_recent[pair[1]]
    noise = random.uniform(0.9, 1.1)
    return (base*0.6 + recent*0.4) * noise

def predict_pairs(data):
    freq_all = calc_freq(data)
    recent = data[-30:] if len(data) > 30 else data
    freq_recent = calc_freq(recent)

    digits = list(freq_all.keys())
    pairs = list(itertools.combinations(digits, 2))

    scored = [(p, score_pair(p, freq_all, freq_recent)) for p in pairs]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:6]
    max_score = top[0][1]

    results = []
    for p, s in top:
        percent = int((s / max_score) * 100)
        results.append({"pair": f"{p[0]}{p[1]}", "score": s, "percent": percent})
    return results

def update_pair_state(predicted, kq5):
    for item in predicted:
        pair = item["pair"]
        if pair not in st.session_state.pair_state:
            st.session_state.pair_state[pair] = {"treo": 0, "win": 0, "lose": 0}

        if is_win(pair, kq5):
            st.session_state.pair_state[pair]["win"] += 1
            st.session_state.pair_state[pair]["treo"] = 0
        else:
            st.session_state.pair_state[pair]["lose"] += 1
            st.session_state.pair_state[pair]["treo"] += 1

def calc_warning_score(pair, base_percent):
    treo = st.session_state.pair_state[pair]["treo"]
    score = base_percent * 0.5 + treo * 8
    if treo > 7:
        score -= 10
    return min(int(score), 95)

# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.6")
st.caption("Tool não mạnh – nuôi cầu – chuẩn sảnh A")

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Dữ liệu",
    "🤖 Dự đoán",
    "📌 Chưa về",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    raw = st.text_area("Dán kết quả mở thưởng", height=140)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU KỲ"):
            chunks = extract_5_digits(raw)
            if chunks:
                st.session_state.data_5so.extend(chunks)
                st.success(f"Đã lưu {len(chunks)} kỳ")
            else:
                st.warning("Không đủ 5 số")

    with col2:
        if st.button("🗑️ XÓA SẠCH"):
            st.session_state.data_5so = []
            st.session_state.predict_log = []
            st.session_state.pair_state = {}
            st.success("Đã reset toàn bộ")

    st.info(f"Tổng số kỳ: {len(st.session_state.data_5so)}")

# ================== TAB 2 ==================
with tab2:
    if len(st.session_state.data_5so) < 10:
        st.warning("Cần ít nhất 10 kỳ")
    else:
        results = predict_pairs(st.session_state.data_5so)

        # lưu log
        st.session_state.predict_log.append({
            "ky": len(st.session_state.data_5so),
            "pairs": [r["pair"] for r in results],
            "detail": results
        })

        grid = st.columns(3)
        for i, r in enumerate(results):
            with grid[i % 3]:
                st.markdown(f"""
                <div style="border:2px solid #00ffcc;border-radius:12px;
                padding:14px;text-align:center;margin-bottom:10px;background:#0e1117">
                <div style="font-size:36px;font-weight:bold;color:#00ffcc">
                {r['pair']}
                </div>
                <div style="color:#ccc">Tin cậy: {r['percent']}%</div>
                </div>
                """, unsafe_allow_html=True)

# ================== TAB 3 ==================
with tab3:
    st.subheader("⚠️ Cặp số chưa về – nên lưu ý")

    if len(st.session_state.data_5so) < 2 or not st.session_state.predict_log:
        st.info("Chưa đủ dữ liệu")
    else:
        # check kỳ mới nhất
        kq5 = st.session_state.data_5so[-1]
        last_pred = st.session_state.predict_log[-1]["detail"]
        update_pair_state(last_pred, kq5)

        for item in last_pred:
            pair = item["pair"]
            treo = st.session_state.pair_state[pair]["treo"]
            if treo >= 3:
                warn = calc_warning_score(pair, item["percent"])
                label = "🚨 ƯU TIÊN" if warn >= 80 else "⚠️ CẦU NÓNG"
                st.markdown(f"**{pair}** | Treo {treo} kỳ | Khả năng về: **{warn}%** {label}")

# ================== TAB 4 ==================
with tab4:
    if not st.session_state.pair_state:
        st.info("Chưa có thống kê")
    else:
        win = sum(v["win"] for v in st.session_state.pair_state.values())
        lose = sum(v["lose"] for v in st.session_state.pair_state.values())
        total = win + lose
        rate = int((win / total) * 100) if total > 0 else 0

        st.metric("Tổng lượt", total)
        st.metric("Thắng", win)
        st.metric("Win rate", f"{rate}%")
