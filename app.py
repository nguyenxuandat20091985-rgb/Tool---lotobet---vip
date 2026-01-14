import streamlit as st
import pandas as pd
import os, re
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config("LOTOBET TOOL – MULTI LAYER", layout="wide")
DATA_FILE = "results.csv"
MIN_DATA = 10

# ================= DATA LAYER =================
def init_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["ky","time","result"]).to_csv(DATA_FILE, index=False)

def load_data():
    init_data()
    df = pd.read_csv(DATA_FILE)
    df["ky"] = pd.to_numeric(df["ky"], errors="coerce").fillna(0).astype(int)
    df["result"] = df["result"].astype(str).str.zfill(5)
    return df

def save_data(nums):
    df = load_data()
    ky = df["ky"].max() if not df.empty else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for n in nums:
        ky += 1
        rows.append({"ky": ky, "time": now, "result": n})

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return len(rows)

# ================= ANALYSIS LAYER =================
def pair_stats(df, n=20):
    last = df.tail(n)["result"]
    return Counter([x[-2:] for x in last])

def digit_stats(df):
    return Counter("".join(df["result"]))

def pair_age(df):
    age = {}
    for r in reversed(df["result"]):
        p = r[-2:]
        age[p] = age.get(p, 0) + 1
    return age

# ================= FILTER LAYER =================
def filter_layers(df, pairs):
    age = pair_age(df)
    out = []

    for p in pairs:
        # TẦNG 1: Tuổi cầu
        if not (2 <= age.get(p, 0) <= 7):
            continue

        # TẦNG 2: Không ra quá gần
        last3 = df.tail(3)["result"]
        if p in [x[-2:] for x in last3]:
            continue

        out.append(p)

    return out

# ================= DECISION LAYER =================
def decide(df):
    stats = pair_stats(df)
    hot = [k for k,v in stats.items() if v >= 2]

    if not hot:
        return {"bet":[], "conf":0, "decision":"⛔ KHÔNG CẦU"}

    filtered = filter_layers(df, hot)
    if not filtered:
        return {"bet":[], "conf":0, "decision":"⛔ BỊ LỌC"}

    digits = digit_stats(df)
    good = [d for d,_ in digits.most_common(5)]

    scored = []
    for p in filtered:
        score = 50
        for d in p:
            if d in good:
                score += 10
        scored.append({"pair":p, "score":score})

    scored = sorted(scored, key=lambda x:x["score"], reverse=True)
    best = scored[:2]
    conf = max([x["score"] for x in best])

    return {
        "bet": best,
        "conf": min(conf,95),
        "decision": "✅ ĐÁNH" if conf >= 70 else "⛔ DỪNG"
    }

# ================= UI =================
st.title("🎯 TOOL SOI CẦU ĐA TẦNG – BẢN MỚI")

raw = st.text_area("Nhập kết quả (mỗi dòng 1 số 5 chữ số)")
if st.button("LƯU"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        st.success(f"Đã lưu {save_data(nums)} kỳ")
        st.rerun()
    else:
        st.error("Sai định dạng")

df = load_data()
st.subheader("📊 DỮ LIỆU")
st.dataframe(df.tail(20), use_container_width=True)

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu")
    st.stop()

st.divider()
ai = decide(df)

st.subheader("🧠 KẾT LUẬN")
for x in ai["bet"]:
    st.write(f"• {x['pair']} | Điểm {x['score']}")

st.metric("Độ tin cậy", f"{ai['conf']}%")
st.markdown(f"### {ai['decision']}")
