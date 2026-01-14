import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.4",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
RESULT_LOG = "result_log.csv"
MIN_DATA = 40

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

# ================= SAVE DATA (ANTI DUPLICATE) =================
def save_pairs_unique(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    existing = df["pair"].tolist()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_count = 0

    for p in pairs:
        p = str(p).zfill(2)

        # ❗ CHỐNG TRÙNG LIÊN TIẾP
        if len(existing) > 0 and p == existing[-1]:
            continue

        df.loc[len(df)] = [now, p]
        existing.append(p)
        new_count += 1

    save_csv(df, DATA_FILE)
    return new_count

# ================= RESULT TRACK =================
def log_result(pair, hit):
    df = load_csv(RESULT_LOG, ["time", "pair", "result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    save_csv(df, RESULT_LOG)

def win_rate(pair, lookback=30):
    df = load_csv(RESULT_LOG, ["time", "pair", "result"])
    df = df[df["pair"] == pair].tail(lookback)
    if len(df) == 0:
        return 0
    return round((df["result"] == "TRÚNG").mean() * 100, 2)

# ================= CYCLE / REPEAT =================
def cycle_score(df, pair):
    seq = df["pair"].tolist()
    pos = [i for i, p in enumerate(seq) if p == pair]

    if len(pos) < 3:
        return -5, "Thiếu dữ liệu"

    gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
    avg_gap = sum(gaps[-3:]) / len(gaps[-3:])
    last_gap = len(seq) - 1 - pos[-1]

    if abs(last_gap - avg_gap) <= 1:
        return 20, "🎯 Đúng nhịp"
    elif last_gap < avg_gap:
        return -10, "⏳ Vừa ra"
    else:
        return -15, "⚠️ Quá hạn"

# ================= CORE AI =================
def analyze_v34(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    rows = []

    for pair in cnt_all:
        # ===== TẦNG 1: TẦN SUẤT =====
        freq_score = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / total) * 0.2
        ) * 100

        # ===== TẦNG 2: CẦU LẶP =====
        c_score, c_note = cycle_score(df, pair)

        # ===== TẦNG 3: LOẠI CẦU BỆT =====
        if cnt10[pair] >= 4:
            biet = -20
        else:
            biet = 0

        score = round(freq_score + c_score + biet, 2)
        rate = win_rate(pair)

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": c_note,
            "Tỷ lệ trúng (%)": rate
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("Điểm AI (%)", ascending=False)

    # ❗ LOẠI CẦU ĐIỂM THẤP
    df_out = df_out[df_out["Điểm AI (%)"] > 0]

    return df_out

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.4")

raw = st.text_area(
    "📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)",
    height=120
)

if st.button("💾 LƯU KỲ"):
    nums = re.findall(r"\d{5}", raw)
    pairs = [n[-2:] for n in nums]

    if pairs:
        added = save_pairs_unique(pairs)
        st.success(f"Đã lưu {added} kỳ (tự động bỏ trùng)")
    else:
        st.error("Sai định dạng")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI phân tích")
    st.stop()

st.divider()

analysis = analyze_v34(df)

st.subheader("🔥 TOP 5 CẶP ĐỀ XUẤT")
st.dataframe(
    analysis.head(5),
    use_container_width=True,
    hide_index=True
)

best = analysis.iloc[0]

st.subheader("🧠 KẾT LUẬN AI")
st.markdown(f"""
### 🎯 Cặp đề xuất: **{best['Cặp']}**
- 📊 **Điểm AI:** `{best['Điểm AI (%)']}%`
- 🔁 **Cầu:** {best['Cầu']}
- ✅ **Tỷ lệ trúng (30 kỳ):** `{best['Tỷ lệ trúng (%)']}%`
""")

if best["Điểm AI (%)"] >= 65 and best["Tỷ lệ trúng (%)"] >= 25:
    st.success("✅ ĐỦ ĐIỀU KIỆN VÀO TIỀN")
else:
    st.warning("⚠️ NÊN THEO DÕI – CHƯA AN TOÀN")

st.divider()

st.subheader("🧾 GHI NHẬN KẾT QUẢ KỲ NÀY")
c1, c2 = st.columns(2)

with c1:
    if st.button("✅ TRÚNG"):
        log_result(best["Cặp"], True)
        st.success("Đã ghi TRÚNG")

with c2:
    if st.button("❌ TRƯỢT"):
        log_result(best["Cặp"], False)
        st.warning("Đã ghi TRƯỢT")

st.caption("⚠️ AI hỗ trợ xác suất – quản lý vốn & kỷ luật là bắt buộc")
