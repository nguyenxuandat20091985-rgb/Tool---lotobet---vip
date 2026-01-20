import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="NUMCORE AI",
    page_icon="🎯",
    layout="centered"
)

DATA_FILE = "data.csv"

# ================== STYLE ==================
st.markdown("""
<style>
.big-title {
    font-size:32px;
    font-weight:700;
}
.sub {
    color:#666;
}
.card {
    padding:20px;
    border-radius:12px;
    background:#f8f9fa;
    margin-bottom:15px;
}
.ai {
    background:#e8f5e9;
}
</style>
""", unsafe_allow_html=True)

# ================== DATA ==================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_numbers(list_numbers):
    df = load_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame([{"time": now, "numbers": n} for n in list_numbers])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================== AI CORE ==================
def ai_center_numbers(df):
    all_digits = []
    recent_digits = []

    # lấy toàn bộ số
    for n in df["numbers"]:
        all_digits.extend(list(n))

    # 15 kỳ gần nhất
    for n in df.tail(15)["numbers"]:
        recent_digits.extend(list(n))

    freq_all = Counter(all_digits)
    freq_recent = Counter(recent_digits)

    score = {}
    for d in "0123456789":
        score[d] = (
            freq_all.get(d, 0) * 0.3 +
            freq_recent.get(d, 0) * 0.4
        )

    # loại số vừa về liên tiếp
    last = df.tail(2)["numbers"].tolist()
    bad = set(last[0]) & set(last[1]) if len(last) == 2 else set()

    for b in bad:
        score[b] *= 0.3

    # chọn 5 số mạnh nhất
    top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:5]
    return [x[0] for x in top]

# ================== UI ==================
st.markdown('<div class="big-title">🎯 NUMCORE AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Phân tích chuỗi số – Ưu tiên hiệu quả – Không nhiễu</div>', unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["📥 Quản lý dữ liệu", "🧠 Phân tích & Dự đoán"])

# ========== TAB 1 ==========
with tab1:
    st.markdown("### Nhập nhiều kỳ (mỗi dòng 1 kết quả – 5 số)")
    text = st.text_area(
        "Ví dụ:\n17723\n95060\n97508",
        height=150
    )

    if st.button("💾 Lưu dữ liệu"):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line.isdigit() and len(line) == 5:
                rows.append(line)
        if rows:
            save_numbers(rows)
            st.success(f"Đã lưu {len(rows)} kỳ")
        else:
            st.error("Không có dữ liệu hợp lệ")

    df = load_data()
    if not df.empty:
        st.markdown("### Dữ liệu đã lưu")
        st.dataframe(df.tail(20), use_container_width=True)

# ========== TAB 2 ==========
with tab2:
    df = load_data()
    if len(df) < 10:
        st.warning("Cần tối thiểu 10 kỳ để AI phân tích")
    else:
        ai_nums = ai_center_numbers(df)

        st.markdown('<div class="card ai">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI TRUNG TÂM")
        st.markdown(f"**Số AI chọn:** {' – '.join(ai_nums)}")
        st.markdown('</div>', unsafe_allow_html=True)

        a = ai_nums[0] + ai_nums[1] + ai_nums[2]
        b = ai_nums[1] + ai_nums[3] + ai_nums[4]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Tổ hợp đề xuất")
        st.markdown(f"- **Nhánh A:** {a}")
        st.markdown(f"- **Nhánh B:** {b}")
        st.markdown('</div>', unsafe_allow_html=True)

        last_digits = Counter("".join(df.tail(20)["numbers"]))
        hot = last_digits.most_common(3)

        st.markdown("### 📊 Số đang nóng")
        for d, c in hot:
            st.write(f"Số {d}: {c} lần / 20 kỳ")
