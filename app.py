import streamlit as st
import pandas as pd
import os, re
from datetime import datetime
from collections import Counter

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – AI V7",
    layout="wide",
    page_icon="🎯"
)

RESULT_FILE = "results.csv"
STATE_FILE = "ai_state.csv"
WIN_FILE = "ai_win_memory.csv"
LOSS_FILE = "ai_loss_memory.csv"

MIN_DATA = 30

# ===== RISK CONFIG =====
BANKROLL = 100
BASE_STAKE = 1
MAX_LOSS_STREAK = 3
DAILY_TAKE_PROFIT = 15
DAILY_STOP_LOSS = -10

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

# ================= ANALYSIS =================
def analyze_pairs(df, n=20):
    last = df.tail(n)["result"].astype(str).str.zfill(5)
    return Counter([x[-2:] for x in last])

def digit_stats(df):
    nums = df["result"].astype(str).str.zfill(5)
    return Counter("".join(nums))

def cau_age(df):
    last = df["result"].astype(str).str.zfill(5)
    age = {}
    for r in reversed(last):
        p = r[-2:]
        age[p] = age.get(p, 0) + 1
    return age

# ================= LAYERS =================
def layer_filter_duplicate(df, pairs):
    last = df.tail(3)["result"].astype(str).str.zfill(5)
    recent = set([x[-2:] for x in last])
    return [p for p in pairs if p not in recent]

def layer_bet_streak(df, pairs):
    freq = analyze_pairs(df, 15)
    return [p for p in pairs if 2 <= freq.get(p, 0) <= 4]

def layer_cau_age(age, pairs):
    return [p for p in pairs if 2 <= age.get(p, 1) <= 7]

def layer_cold(df, pairs):
    results = df["result"].astype(str).str.zfill(5)
    ok = []
    for p in pairs:
        miss = 0
        for r in reversed(results):
            if r[-2:] == p:
                break
            miss += 1
        if miss <= 10:
            ok.append(p)
    return ok

def layer_cycle(df, pairs):
    results = df["result"].astype(str).str.zfill(5)
    ok = []
    for p in pairs:
        pos = [i for i, r in enumerate(results) if r[-2:] == p]
        if len(pos) >= 3:
            gaps = [pos[i]-pos[i-1] for i in range(1,len(pos))]
            avg = sum(gaps[-3:]) / len(gaps[-3:])
            if 3 <= avg <= 8:
                ok.append(p)
    return ok

def detect_broken(df):
    last = df.tail(5)["result"].astype(str).str.zfill(5)
    return len(set([x[-2:] for x in last])) >= 4

def rhythm_ok(df):
    last = df.tail(6)["result"].astype(str).str.zfill(5)
    return len(set([x[-2:] for x in last])) <= 3

# ================= MONEY MANAGEMENT =================
def stake_by_confidence(conf):
    if conf >= 85:
        return BASE_STAKE * 3
    if conf >= 75:
        return BASE_STAKE * 2
    if conf >= 65:
        return BASE_STAKE
    return 0

# ================= AI V7 =================
def ai_v7(df):
    pairs = analyze_pairs(df, 20)
    hot = [k for k,v in pairs.items() if v >= 3]
    warm = [k for k,v in pairs.items() if v == 2]
    candidates = list(dict.fromkeys(hot + warm))

    if not candidates:
        return {"bet":[], "conf":0, "decision":"⛔ KHÔNG CÓ CẦU"}

    age = cau_age(df)
    digits = digit_stats(df)
    good_digits = [d for d,_ in digits.most_common(5)]
    bad_digits = [d for d,_ in digits.most_common()[-3:]]

    # ===== FULL FILTER =====
    candidates = layer_filter_duplicate(df, candidates)
    candidates = layer_bet_streak(df, candidates)
    candidates = layer_cau_age(age, candidates)
    candidates = layer_cold(df, candidates)
    candidates = layer_cycle(df, candidates)

    if not candidates or not rhythm_ok(df):
        return {"bet":[], "conf":0, "decision":"⛔ NHỊP XẤU – DỪNG"}

    broken = detect_broken(df)

    scored = []
    for p in candidates:
        score = 50
        for d in p:
            if d in good_digits: score += 10
            if d in bad_digits: score -= 15
        if p in recent_memory(LOSS_FILE): score -= 30
        if p in recent_memory(WIN_FILE): score += 20
        score += max(0, 10 - age.get(p,1))
        scored.append({"pair":p, "score":score})

    scored = sorted(scored, key=lambda x:x["score"], reverse=True)
    best = scored[:2]
    conf = max([x["score"] for x in best])

    if broken:
        conf -= 25

    stake = stake_by_confidence(conf)

    return {
        "bet": best,
        "conf": max(0,min(conf,95)),
        "stake": stake,
        "decision": "✅ ĐÁNH" if stake>0 else "⛔ DỪNG"
    }

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – AI V7")

raw = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)")
if st.button("💾 LƯU"):
    nums = re.findall(r"\d{5}", raw)
    if nums:
        st.success(f"Đã lưu {save_results(nums)} kỳ")
    else:
        st.error("Sai định dạng")

df = load_csv(RESULT_FILE, ["ky","time","result"])
if len(df) < MIN_DATA:
    st.warning("Chưa đủ dữ liệu")
    st.stop()

st.divider()
ai = ai_v7(df)

st.subheader("🧠 AI V7 – KẾT LUẬN")
for x in ai["bet"]:
    st.write(f"• {x['pair']} | Điểm {x['score']}")

st.metric("📊 Độ tin cậy", f"{ai['conf']}%")
st.metric("💰 Mức cược đề xuất", ai.get("stake",0))
st.markdown(f"### 📌 QUYẾT ĐỊNH: **{ai['decision']}**")

st.caption("⚠️ AI chỉ hỗ trợ xác suất – thắng thua phụ thuộc kỷ luật & quản trị vốn")
