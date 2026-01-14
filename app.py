import streamlit as st
import pandas as pd
import os, re
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3",
    layout="wide",
    page_icon="🎯"
)

RESULT_FILE = "results.csv"
ROUND_FILE = "round_state.csv"
LOSS_FILE = "ai_loss_memory.csv"

MIN_DATA = 30
BET_THRESHOLD = 5

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

# ================= ROUND STATE =================
def load_round_state():
    if os.path.exists(ROUND_FILE):
        df = pd.read_csv(ROUND_FILE)
        if not df.empty:
            return df
    return pd.DataFrame([{
        "round_id": 0,
        "last_number": "",
        "last_result": "NONE"
    }])

def save_round_state(round_id, last_number, last_result):
    pd.DataFrame([{
        "round_id": round_id,
        "last_number": last_number,
        "last_result": last_result
    }]).to_csv(ROUND_FILE, index=False)

def ai_detect_new_round(new_ky):
    state = load_round_state().iloc[0]
    if int(new_ky) <= int(state["round_id"]):
        return False
    save_round_state(new_ky, "", "WAIT")
    return True

# ================= DATA INPUT =================
def save_results(results):
    df = load_csv(RESULT_FILE, ["ky", "time", "result"])
    last_ky = int(df["ky"].max()) if not df.empty else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for r in results:
        last_ky += 1
        rows.append({"ky": last_ky, "time": now, "result": r})

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_csv(df, RESULT_FILE)
    return len(rows)

# ================= CORE ANALYSIS =================
def streak_analysis(df):
    results = df["result"].astype(str).str.zfill(5)
    streak = {}
    for n in "0123456789":
        miss = 0
        for r in reversed(results):
            if n in r:
                break
            miss += 1
        streak[n] = miss
    return streak

def ai_analyze_trend(df):
    last20 = df.tail(20)["result"].astype(str).str.zfill(5)
    pairs = Counter([x[-2:] for x in last20])
    bet = [k for k, v in pairs.items() if v >= 3]
    hot = [k for k, v in pairs.items() if v == 2]
    return bet, hot

def ai_predict_digits(df):
    nums = df["result"].astype(str).str.zfill(5)
    cnt = Counter("".join(nums))
    return [d for d, _ in cnt.most_common(5)]

def ai_blacklist_digits(df):
    last10 = df.tail(10)["result"].astype(str).str.zfill(5)
    cnt = Counter("".join(last10))
    return [d for d, v in cnt.items() if v <= 1]

# ================= LOSS MEMORY =================
def ai_load_loss():
    if os.path.exists(LOSS_FILE):
        return pd.read_csv(LOSS_FILE)
    return pd.DataFrame(columns=["time", "pair"])

def ai_mark_loss(pair):
    df = ai_load_loss()
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pair
    ]
    df.to_csv(LOSS_FILE, index=False)

def ai_recent_losses():
    return ai_load_loss().tail(5)["pair"].tolist()

# ================= AI V6 =================
AI6_LOOKBACK = 25
AI6_MIN_CONF = 75
AI6_MAX_BET = 2

def ai6_cau_age(df):
    last = df.tail(AI6_LOOKBACK)["result"].astype(str).str.zfill(5)
    pairs = [x[-2:] for x in last]
    age = {}
    for p in pairs[::-1]:
        age[p] = age.get(p, 0) + 1
    return age

def ai6_is_broken(df):
    last = df.tail(6)["result"].astype(str).str.zfill(5)
    pairs = [x[-2:] for x in last]
    return len(set(pairs[-3:])) == 3

def ai6_predict(df):
    bet, hot = ai_analyze_trend(df)
    good = ai_predict_digits(df)
    bad = ai_blacklist_digits(df)
    losses = ai_recent_losses()
    age = ai6_cau_age(df)

    candidates = list(dict.fromkeys(bet + hot))
    scored = []

    for p in candidates:
        score = 50
        for d in p:
            if d in good:
                score += 10
            if d in bad:
                score -= 20
        if p in losses:
            score -= 35

        a = age.get(p, 1)
        if a <= 3:
            score += 15
        elif a >= 8:
            score -= 25

        scored.append({"pair": p, "score": score})

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    broken = ai6_is_broken(df)

    best = scored[:AI6_MAX_BET]
    conf = max([x["score"] for x in best], default=0)
    if broken:
        conf -= 25

    return {
        "best": best,
        "confidence": max(0, min(conf, 95)),
        "decision": "✅ ĐÁNH" if conf >= AI6_MIN_CONF and not broken else "⛔ DỪNG",
        "broken": broken
    }

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3 (AI V6)")

raw = st.text_area("📥 Nhập kết quả – mỗi dòng 1 số 5 chữ số")
if st.button("💾 LƯU"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        st.success(f"Đã lưu {save_results(nums)} kỳ")
    else:
        st.error("Sai định dạng")

df = load_csv(RESULT_FILE, ["ky", "time", "result"])

if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu để AI hoạt động")
    st.stop()

st.divider()
st.subheader("🧠 AI V6 – QUYẾT ĐỊNH CUỐI")

ai6 = ai6_predict(df)

if ai6["broken"]:
    st.error("⚠️ CẦU GÃY – AI CẤM ĐÁNH")

for x in ai6["best"]:
    st.write(f"• {x['pair']} | Điểm: {x['score']}")

st.metric("📊 ĐỘ TIN CẬY", f"{ai6['confidence']}%")
st.markdown(f"### 📌 QUYẾT ĐỊNH: **{ai6['decision']}**")

st.caption("🚀 LOTOBET AUTO PRO V3 – AI V6 | Đánh theo kỳ – Có kỷ luật – Không đoán mò")
