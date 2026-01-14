import streamlit as st
import pandas as pd
import os, re
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V7 (STABLE)",
    layout="wide",
    page_icon="🎯"
)

RESULT_FILE = "results.csv"
MIN_DATA = 10   # test trước, sau nâng lên 30

# ================= DATA CORE =================
def init_results():
    if not os.path.exists(RESULT_FILE):
        df = pd.DataFrame(columns=["ky", "time", "result"])
        df.to_csv(RESULT_FILE, index=False)

def load_results():
    init_results()
    df = pd.read_csv(RESULT_FILE)

    # ÉP KIỂU – CHỐNG KEYERROR
    if "ky" not in df.columns:
        df["ky"] = range(1, len(df) + 1)
    if "time" not in df.columns:
        df["time"] = ""
    if "result" not in df.columns:
        df["result"] = ""

    df["ky"] = pd.to_numeric(df["ky"], errors="coerce").fillna(0).astype(int)
    df["result"] = df["result"].astype(str)

    return df

def save_results(nums):
    df = load_results()
    last_ky = int(df["ky"].max()) if not df.empty else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for n in nums:
        last_ky += 1
        rows.append({
            "ky": last_ky,
            "time": now,
            "result": n
        })

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(RESULT_FILE, index=False)

    return len(rows)

# ================= AI CORE =================
def analyze_ai(df):
    last = df.tail(20)["result"].str.zfill(5)

    # Phân tích cặp 2 số
    pairs = Counter([x[-2:] for x in last])
    hot = [k for k, v in pairs.items() if v >= 2]

    # Phân tích chữ số
    digits = Counter("".join(last))
    good_digits = [d for d, _ in digits.most_common(5)]

    scored = []
    for p in hot:
        score = 50
        for d in p:
            if d in good_digits:
                score += 10
        scored.append({"pair": p, "score": score})

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    best = scored[:2]

    conf = max([x["score"] for x in best], default=0)

    return {
        "best": best,
        "confidence": min(conf, 95),
        "decision": "✅ ĐÁNH" if conf >= 70 else "⛔ DỪNG"
    }

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V7 (BẢN ỔN ĐỊNH)")

raw = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)")

if st.button("💾 LƯU KẾT QUẢ"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        n = save_results(nums)
        st.success(f"✅ Đã lưu {n} kỳ")
        st.rerun()
    else:
        st.error("❌ Sai định dạng – cần 5 chữ số")

df = load_results()

st.subheader("📊 DỮ LIỆU ĐÃ LƯU")
st.dataframe(df.tail(20), use_container_width=True)

if len(df) < MIN_DATA:
    st.warning("⚠️ Chưa đủ dữ liệu để AI phân tích")
    st.stop()

st.divider()

ai = analyze_ai(df)

st.subheader("🧠 PHÂN TÍCH AI")
for x in ai["best"]:
    st.write(f"• Cặp {x['pair']} | Điểm {x['score']}")

st.metric("📊 Độ tin cậy", f"{ai['confidence']}%")
st.markdown(f"### 📌 QUYẾT ĐỊNH: **{ai['decision']}**")

st.caption("⚠️ AI hỗ trợ xác suất – bắt buộc quản lý vốn & kỷ luật")
