import streamlit as st
import re
from collections import Counter
import itertools
import random
import math

# ================== CẤU HÌNH ==================
st.set_page_config(
    page_title="LOTOBET 2 SỐ 5 TINH v6.0",
    layout="centered"
)

# ================== SESSION ==================
if "data_5so" not in st.session_state:
    st.session_state.data_5so = []

# ================== HÀM CORE ==================
def extract_5_digits(raw_text):
    """
    Lọc toàn bộ chữ số bằng Regex
    Gom thành từng kỳ 5 số
    """
    digits = re.findall(r"\d", raw_text)
    chunks = []
    for i in range(0, len(digits) - 4, 5):
        chunk = digits[i:i+5]
        if len(chunk) == 5:
            chunks.append(chunk)
    return chunks


def calc_frequency(data):
    flat = list(itertools.chain.from_iterable(data))
    return Counter(flat)


def calc_recent_bias(data, n=30):
    recent = data[-n:] if len(data) >= n else data
    flat = list(itertools.chain.from_iterable(recent))
    return Counter(flat)


def score_pair(pair, freq_all, freq_recent):
    """
    Tính trọng số cặp 2D dựa trên:
    - Tần suất tổng
    - Nhịp gần (bệt)
    - Ngẫu nhiên nhẹ để phân hóa %
    """
    a, b = pair
    base = freq_all[a] + freq_all[b]
    recent = freq_recent[a] + freq_recent[b]
    noise = random.uniform(0.9, 1.1)
    score = (base * 0.6 + recent * 0.4) * noise
    return score


def predict_pairs(data):
    freq_all = calc_frequency(data)
    freq_recent = calc_recent_bias(data)

    digits = list(freq_all.keys())
    all_pairs = list(itertools.combinations(digits, 2))

    scored = []
    for p in all_pairs:
        s = score_pair(p, freq_all, freq_recent)
        scored.append((p, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    top6 = scored[:6]

    max_score = top6[0][1]
    results = []
    for p, s in top6:
        percent = int((s / max_score) * 100)
        results.append({
            "pair": f"{p[0]}{p[1]}",
            "percent": percent
        })
    return results


# ================== UI ==================
st.title("🎯 LOTOBET 2 SỐ 5 TINH v6.0")
st.caption("Phân tích đủ 5 số – Không bỏ nhịp – Chuẩn sảnh A")

tab1, tab2, tab3 = st.tabs([
    "📂 Quản lý dữ liệu",
    "🤖 Dự đoán AI",
    "📊 Thống kê"
])

# ================== TAB 1 ==================
with tab1:
    st.subheader("📥 Nhập dữ liệu mở thưởng")
    raw = st.text_area(
        "Dán kết quả (OCR / Web / File)",
        height=150,
        placeholder="Ví dụ: 15406 98231 44019 ..."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 LƯU DỮ LIỆU"):
            chunks = extract_5_digits(raw)
            if chunks:
                st.session_state.data_5so.extend(chunks)
                st.success(f"Đã lưu {len(chunks)} kỳ (5 số/kỳ)")
            else:
                st.warning("Không phát hiện đủ cụm 5 số")

    with col2:
        if st.button("🗑️ XÓA SẠCH RAM"):
            st.session_state.data_5so = []
            st.success("Đã xóa toàn bộ dữ liệu")

    st.info(f"Tổng số kỳ đã lưu: {len(st.session_state.data_5so)}")

# ================== TAB 2 ==================
with tab2:
    st.subheader("🔥 6 CẶP 2 SỐ DỰ ĐOÁN CAO NHẤT")

    if len(st.session_state.data_5so) < 10:
        st.warning("Cần tối thiểu 10 kỳ để AI bắt nhịp")
    else:
        results = predict_pairs(st.session_state.data_5so)

        grid = st.columns(3)
        for i, res in enumerate(results):
            with grid[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border:2px solid #00ffcc;
                        border-radius:12px;
                        padding:14px;
                        text-align:center;
                        margin-bottom:10px;
                        background-color:#0e1117;
                    ">
                        <div style="font-size:38px;font-weight:bold;color:#00ffcc;">
                            {res['pair']}
                        </div>
                        <div style="font-size:16px;color:#cccccc;">
                            Tin cậy: {res['percent']}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ================== TAB 3 ==================
with tab3:
    st.subheader("📊 Thống kê tần suất 5 số")

    if st.session_state.data_5so:
        freq = calc_frequency(st.session_state.data_5so)
        for d in sorted(freq.keys()):
            st.write(f"Số {d}: {freq[d]} lần")
    else:
        st.info("Chưa có dữ liệu")
