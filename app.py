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
LOSS_FILE = "ai_loss_memory.csv"
MIN_DATA = 30

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

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

def ai_recent_losses():
    return ai_load_loss().tail(5)["pair"].tolist()

# ================= LAYER FUNCTIONS =================
def layer_filter_duplicate(df, pairs):
    last = df.iloc[-1]["result"][-2:]
    return [p for p in pairs if p != last]

def layer_bet_streak(df, pairs, min_bet=2, max_bet=5):
    last20 = df.tail(20)["result"].astype(str).str.zfill(5)
    out = []
    for p in pairs:
        c = sum(1 for x in last20 if x[-2:] == p)
        if min_bet <= c <= max_bet:
            out.append(p)
    return out

def layer_cau_age_filter(age_map, pairs, min_age=1, max_age=6):
    return [p for p in pairs if min_age <= age_map.get(p, 0) <= max_age]

def layer_cold_filter(df, pairs, cold_limit=8):
    results = df["result"].astype(str).str.zfill(5)
    out = []
    for p in pairs:
        miss = 0
        for r in reversed(results):
            if r[-2:] == p:
                break
            miss += 1
        if miss < cold_limit:
            out.append(p)
    return out

def layer_return_cycle(df, pairs, min_gap=3, max_gap=8):
    results = df["result"].astype(str).str.zfill(5)
    ok = []
    for p in pairs:
        pos = [i for i, r in enumerate(results) if r[-2:] == p]
        if len(pos) >= 3:
            gaps = [pos[i] - pos[i-1] for i in range(1, len(pos))]
            avg = sum(gaps[-3:]) / len(gaps[-3:])
            if min_gap <= avg <= max_gap:
                ok.append(p)
    return ok

def layer_rhythm_check(df):
    last = df.tail(5)["result"].astype(str).str.zfill(5)
    pairs = [x[-2:] for x in last]
    return len(set(pairs)) <= 3

def layer_detect_broken(df):
    last = df.tail(5)["result"].astype(str).str.zfill(5)
    pairs = [x[-2:] for x in last]
    return len(set(pairs[-3:])) == 3

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

def ai6_predict(df):
    bet, hot = ai_analyze_trend(df)
    good_digits = ai_predict_digits(df)
    bad_digits = ai_blacklist_digits(df)
    losses = ai_recent_losses()
    age_map = ai6_cau_age(df)

    candidates = list(dict.fromkeys(bet + hot))
    if not candidates:
        return {"best": [], "confidence": 0, "decision": "⛔ KHÔNG CÓ CẦU", "broken": True}

    if not layer_rhythm_check(df):
        return {"best": [], "confidence": 0, "decision": "⛔ NHỊP XẤU – DỪNG", "broken": True}

    candidates = layer_filter_duplicate(df, candidates)
    candidates = layer_bet_streak(df, candidates)
    candidates = layer_cau_age_filter(age_map, candidates)
    candidates = layer_cold_filter(df, candidates)
    candidates = layer_return_cycle(df, candidates)

    if not candidates:
        return {"best": [], "confidence": 0, "decision": "⛔ BỊ LỌC HẾT", "broken": True}

    broken = layer_detect_broken(df)

    scored = []
    for p in candidates:
        score = 50
        for d in p:
            if d in good_digits:
                score += 10
            if d in bad_digits:
                score -= 20
        if p in losses:
            score -= 35
        a = age_map.get(p, 1)
        if a <= 3:
            score += 15
        elif a >= 8:
            score -= 25
        scored.append({"pair": p, "score": score})

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
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
st.title("🎯 LOTOBET AUTO PRO – V3 (AI V6 FULL TẦNG)")

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
st.caption("🚀 LOTOBET AUTO PRO V3 – AI V6 | Đánh có kỷ luật – Không đoán mò")
