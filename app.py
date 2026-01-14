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
WIN_FILE = "ai_win_memory.csv"
LOSS_FILE = "ai_loss_memory.csv"

MIN_DATA = 30

BASE_STAKE = 1

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

# ================= RESULT INPUT =================
def save_results(nums):
    df = load_csv(RESULT_FILE, ["ky", "time", "result"])
    last_ky = int(df["ky"].max()) if not df.empty else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for n in nums:
        last_ky += 1
        rows.append({"ky": last_ky, "time": now, "result": n})

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_csv(df, RESULT_FILE)

    return len(rows)

# ================= MEMORY =================
def load_memory(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["time", "pair"])

def remember(path, pair):
    df = load_memory(path)
    df.loc[len(df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pair]
    df.to_csv(path, index=False)

def recent_memory(path, n=5):
    return load_memory(path).tail(n)["pair"].tolist()

# ================= ANALYSIS CORE =================
def analyze_pairs(df, n=20):
    last = df.tail(n)["result"].astype(str).str.zfill(5)
    return Counter([x[-2:] for x in last])

def cau_age(df):
    results = df["result"].astype(str).str.zfill(5)
    age = {}
    for p in set([r[-2:] for r in results]):
        miss = 0
        for r in reversed(results):
            if r[-2:] == p:
                break
            miss += 1
        age[p] = miss
    return age

# ================= FILTER LAYERS =================
def layer_duplicate(df, pairs):
    recent = set(df.tail(3)["result"].astype(str).str.zfill(5).str[-2:])
    return [p for p in pairs if p not in recent]

def layer_freq(freq, pairs):
    return [p for p in pairs if 2 <= freq.get(p, 0) <= 4]

def layer_age(age, pairs):
    return [p for p in pairs if 2 <= age.get(p, 99) <= 8]

def layer_cold(df, pairs):
    results = df["result"].astype(str).str.zfill(5)
    ok = []
    for p in pairs:
        miss = 0
        for r in reversed(results):
            if r[-2:] == p:
                break
            miss += 1
        if miss <= 12:
            ok.append(p)
    return ok

def detect_broken(df):
    last = df.tail(6)["result"].astype(str).str.zfill(5)
    return len(set([x[-2:] for x in last])) >= 5

def rhythm_ok(df):
    last = df.tail(6)["result"].astype(str).str.zfill(5)
    return len(set([x[-2:] for x in last])) <= 4

# ================= MONEY =================
def stake_by_conf(conf):
    if conf >= 80:
        return BASE_STAKE * 3
    if conf >= 70:
        return BASE_STAKE * 2
    if conf >= 65:
        return BASE_STAKE
    return 0

# ================= AI V7 CORE =================
def ai_v7(df):
    freq = analyze_pairs(df, 20)
    hot = [k for k,v in freq.items() if v >= 3]
    warm = [k for k,v in freq.items() if v == 2]
    candidates = list(dict.fromkeys(hot + warm))

    if not candidates:
        return {"bet":[], "conf":0, "stake":0, "decision":"⛔ KHÔNG CẦU"}

    if not rhythm_ok(df):
        return {"bet":[], "conf":0, "stake":0, "decision":"⛔ NHỊP XẤU"}

    age = cau_age(df)

    digits = Counter("".join(df.tail(15)["result"].astype(str).str.zfill(5)))
    good_digits = [d for d,_ in digits.most_common(5)]
    bad_digits = [d for d,_ in digits.most_common()[-3:]]

    candidates = layer_duplicate(df, candidates)
    candidates = layer_freq(freq, candidates)
    candidates = layer_age(age, candidates)
    candidates = layer_cold(df, candidates)

    if not candidates:
        return {"bet":[], "conf":0, "stake":0, "decision":"⛔ BỊ LỌC"}

    broken = detect_broken(df)

    scored = []
    for p in candidates:
        score = 50

        for d in p:
            if d in good_digits: score += 10
            if d in bad_digits: score -= 15

        if p in recent_memory(LOSS_FILE): score -= 25
        if p in recent_memory(WIN_FILE): score += 15

        a = age.get(p, 10)
        if a <= 3: score += 10
        if a >= 9: score -= 20

        scored.append({"pair":p, "score":score})

    scored.sort(key=lambda x:x["score"], reverse=True)
    best = scored[:2]

    conf = max([x["score"] for x in best], default=0)
    if broken:
        conf -= 25

    conf = max(0, min(conf, 95))
    stake = stake_by_conf(conf)

    return {
        "bet": best,
        "conf": conf,
        "stake": stake,
        "decision": "✅ ĐÁNH" if conf >= 65 and stake > 0 and not broken else "⛔ DỪNG"
    }

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V7 (STABLE)")

raw = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)")

if st.button("💾 LƯU"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        st.success(f"Đã lưu {save_results(nums)} kỳ")
    else:
        st.error("Sai định dạng")

df = load_csv(RESULT_FILE, ["ky","time","result"])

if len(df) < MIN_DATA:
    st.warning("⚠️ Chưa đủ dữ liệu để AI hoạt động")
    st.stop()

st.divider()

ai = ai_v7(df)

st.subheader("🧠 KẾT LUẬN AI V7")

for x in ai["bet"]:
    st.write(f"• **{x['pair']}** | Điểm: {x['score']}")

st.metric("📊 Độ tin cậy", f"{ai['conf']}%")
st.metric("💰 Mức cược", ai["stake"])
st.markdown(f"## 📌 QUYẾT ĐỊNH: **{ai['decision']}**")

st.caption("⚠️ AI hỗ trợ xác suất – kỷ luật & quản trị vốn quyết định lợi nhuận")
