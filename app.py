import streamlit as st
import pandas as pd
import re
import os
from collections import Counter
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3.6",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data_v36.csv"
HIS_FILE = "history_v36.csv"
MIN_DATA = 40
FAST_WINDOW = 300   # chỉ phân tích 300 kỳ gần nhất

# ================= STORAGE =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_data(new_pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    exist = set(df["pair"].astype(str) + df["time"].astype(str))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for p in new_pairs:
        key = str(p) + now
        if key not in exist:
            rows.append({"time": now, "pair": int(p)})

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
    return len(rows)

# ================= ANALYSIS =================
def analyze_core(df):
    df_fast = df.tail(FAST_WINDOW)
    pairs = df_fast["pair"].astype(str).str.zfill(2)

    cnt_all = Counter(pairs)
    cnt_20 = Counter(pairs.tail(20))
    cnt_50 = Counter(pairs.tail(50))

    results = []
    for p in cnt_all:
        score = (
            cnt_20.get(p,0)*0.5 +
            cnt_50.get(p,0)*0.3 +
            cnt_all.get(p,0)*0.2
        )
        percent = round(score / 20 * 100, 2)

        # cầu lặp
        last_positions = [i for i,x in enumerate(pairs) if x == p]
        cycle = "—"
        if len(last_positions) >= 2:
            gap = last_positions[-1] - last_positions[-2]
            if gap <= 2:
                cycle = "🔥 Lặp nhanh"
            elif gap <= 5:
                cycle = "⏳ Đang nuôi"
            else:
                cycle = "❄️ Lạnh"

        results.append({
            "pair": p,
            "score": percent,
            "cycle": cycle
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ================= TRACK =================
def record_result(pair, hit):
    df = load_csv(HIS_FILE, ["time","pair","result"])
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair,
        "TRÚNG" if hit else "TRƯỢT"
    ]
    df.to_csv(HIS_FILE, index=False)

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3.6")

raw = st.text_area("📥 Dán kết quả (mỗi dòng 1 số – tối thiểu 2 số cuối)", height=120)

if st.button("💾 LƯU DỮ LIỆU"):
    digits = re.findall(r"\d{2,}", raw)
    pairs = [d[-2:] for d in digits]
    if pairs:
        saved = save_data(pairs)
        st.success(f"✅ Đã lưu {saved} kỳ (đã tự lọc trùng)")
    else:
        st.error("❌ Không nhận diện được dữ liệu")

df = load_csv(DATA_FILE, ["time","pair"])
st.info(f"📊 Tổng dữ liệu hợp lệ: {len(df)} kỳ")

# ================= PREDICT =================
if len(df) >= MIN_DATA:
    analysis = analyze_core(df)

    st.subheader("🔥 TOP 5 CẶP ĐỀ XUẤT")
    st.table(pd.DataFrame(analysis[:5]))

    best = analysis[0]

    st.subheader("🧠 KẾT LUẬN AI")
    st.markdown(f"""
    **Cặp đề xuất:** `{best['pair']}`  
    **Xác suất AI:** `{best['score']}%`  
    **Trạng thái cầu:** {best['cycle']}
    """)

    if best["score"] >= 25:
        st.success("✅ NÊN ĐÁNH (1–2 tay)")
    else:
        st.warning("⚠️ NÊN THEO DÕI")

    col1, col2 = st.columns(2)
    if col1.button("✅ TRÚNG"):
        record_result(best["pair"], True)
        st.success("Đã ghi nhận TRÚNG")
    if col2.button("❌ TRƯỢT"):
        record_result(best["pair"], False)
        st.warning("Đã ghi nhận TRƯỢT")

# ================= HISTORY =================
st.subheader("🧾 LỊCH SỬ THEO DÕI")
his = load_csv(HIS_FILE, ["time","pair","result"])
if not his.empty:
    st.table(his.tail(10))
