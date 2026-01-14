import streamlit as st
import pandas as pd
import re, os
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V3.7",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data_pair2.csv"
RESULT_FILE = "result_track.csv"
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

# ================= SAVE DATA (ANTI DUP) =================
def save_pairs_unique(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0

    for p in pairs:
        p = str(p).zfill(2)
        if not df.empty and df.iloc[-1]["pair"] == p:
            continue
        df.loc[len(df)] = [now, p]
        added += 1

    save_csv(df, DATA_FILE)
    return added

# ================= RESULT MEMORY =================
def log_result(pair, hit):
    df = load_csv(RESULT_FILE, ["time", "pair", "result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    save_csv(df, RESULT_FILE)

def recent_losses(pair, n=3):
    df = load_csv(RESULT_FILE, ["time", "pair", "result"])
    df = df[df["pair"] == pair].tail(n)
    return (df["result"] == "TRƯỢT").sum()

def win_rate(pair, lookback=30):
    df = load_csv(RESULT_FILE, ["time", "pair", "result"])
    df = df[df["pair"] == pair].tail(lookback)
    if df.empty:
        return 0
    return round((df["result"] == "TRÚNG").mean() * 100, 2)

# ================= CYCLE =================
def cycle_note(df, pair):
    seq = df["pair"].tolist()
    pos = [i for i,p in enumerate(seq) if p == pair]
    if len(pos) < 3:
        return "⏳ Mới"
    gaps = [pos[i]-pos[i-1] for i in range(1,len(pos))]
    avg = sum(gaps[-3:]) / len(gaps[-3:])
    last_gap = len(seq) - 1 - pos[-1]

    if abs(last_gap - avg) <= 1:
        return "🔁 Cầu lặp"
    if last_gap < avg:
        return "🔥 Nóng"
    return "⚠️ Gãy"

# ================= CORE AI =================
def analyze_ai(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    rows = []

    for pair in cnt_all:
        freq = (
            (cnt10[pair]/10)*0.5 +
            (cnt20[pair]/20)*0.3 +
            (cnt_all[pair]/total)*0.2
        ) * 100

        cycle = cycle_note(df, pair)
        rate = win_rate(pair)

        score = freq

        # 🔥 HOT / ❄️ BỆT
        if cnt10[pair] >= 4:
            score -= 25
        elif cnt10[pair] == 3:
            score += 10

        # 🚫 ANTI GỠ
        loss_streak = recent_losses(pair)
        if loss_streak >= 2:
            score -= 30

        score = round(score, 2)

        if score <= 0:
            continue

        rows.append({
            "Cặp": pair,
            "Điểm AI (%)": score,
            "Cầu": cycle,
            "Tỷ lệ trúng (%)": rate
        })

    if not rows:
        return pd.DataFrame(columns=["Cặp","Điểm AI (%)","Cầu","Tỷ lệ trúng (%)"])

    df_out = pd.DataFrame(rows)
    return df_out.sort_values("Điểm AI (%)", ascending=False)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V3.7")

raw = st.text_area("📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)", height=120)

if st.button("💾 LƯU KỲ"):
    nums = re.findall(r"\d{5}", raw)
    pairs = [n[-2:] for n in nums]
    if pairs:
        added = save_pairs_unique(pairs)
        st.success(f"Đã lưu {added} kỳ (tự bỏ trùng)")
    else:
        st.error("Sai định dạng")

df = load_csv(DATA_FILE, ["time","pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI phân tích")
    st.stop()

st.divider()

analysis = analyze_ai(df)

if analysis.empty:
    st.error("🚫 NGÀY XẤU – AI KHÓA ĐÁNH (BẢO VỆ VỐN)")
    st.stop()

# ================= TOP =================
st.subheader("🔥 TOP CẶP ĐỀ XUẤT")
st.dataframe(analysis.head(5), use_container_width=True, hide_index=True)

best = analysis.iloc[0]

# ================= DECISION =================
st.subheader("🧠 KẾT LUẬN AI")

score = best["Điểm AI (%)"]
rate = best["Tỷ lệ trúng (%)"]

if score >= 75 and rate >= 30:
    level = "🟢 ĐÁNH CHÍNH"
    pick = 1
elif score >= 60:
    level = "🟡 ĐÁNH NHẸ"
    pick = 2
else:
    level = "🔴 BỎ – KHÔNG VÀO"
    pick = 0

st.markdown(f"""
### 🎯 Cặp đề xuất: **{best['Cặp']}**
- 📊 Điểm AI: `{score}%`
- 🔁 Cầu: {best['Cầu']}
- ✅ Tỷ lệ trúng: `{rate}%`
- 🚦 Mức đánh: **{level}**
""")

# ================= BOARD =================
st.subheader("📋 BẢNG SỐ ĐỀ (CO GIÃN)")

if pick == 0:
    st.warning("🚫 Hôm nay KHÔNG CÓ SỐ AN TOÀN")
else:
    st.success(f"🎯 NÊN ĐÁNH {pick} CON:")
    st.write(list(analysis.head(pick)["Cặp"]))

# ================= RESULT INPUT =================
st.divider()
st.subheader("🧾 GHI NHẬN KẾT QUẢ")

c1, c2 = st.columns(2)
with c1:
    if st.button("✅ TRÚNG"):
        log_result(best["Cặp"], True)
        st.success("Đã ghi TRÚNG")
with c2:
    if st.button("❌ TRƯỢT"):
        log_result(best["Cặp"], False)
        st.warning("Đã ghi TRƯỢT")

st.caption("⚠️ AI hỗ trợ xác suất – kỷ luật & quản lý vốn quyết định lợi nhuận")
