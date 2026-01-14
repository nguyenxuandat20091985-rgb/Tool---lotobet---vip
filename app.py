import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3.1 STABLE",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
AI_FILE = "ai_weight.csv"

# ================= UTIL =================
def load_csv(path, cols):
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def save_csv(df, path):
    df.to_csv(path, index=False)

def safe_pair(p):
    try:
        p = str(int(p)).zfill(2)
        return p if len(p) == 2 else None
    except:
        return None

# ================= SAVE DATA =================
def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for p in pairs:
        sp = safe_pair(p)
        if sp:
            rows.append({"time": now, "pair": sp})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_csv(df, DATA_FILE)

# ================= AI MEMORY =================
def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, win=True):
    pair = safe_pair(pair)
    if not pair:
        return
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.2 if win else -0.1
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    save_csv(ai, AI_FILE)

# ================= ANALYSIS =================
def analyze_v3(df):
    df = df.copy()
    df["pair"] = df["pair"].apply(safe_pair)
    df = df.dropna()

    if len(df) < 20:
        return []

    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    ai = load_ai()
    ai_map = dict(zip(ai["pair"], ai["weight"]))

    results = []
    for pair in cnt_all:
        base = (
            (cnt10[pair] / 10) * 0.5 +
            (cnt20[pair] / 20) * 0.3 +
            (cnt_all[pair] / total) * 0.2
        )
        weight = ai_map.get(pair, 1.0)
        score = round(base * weight * 100, 2)

        if cnt10[pair] >= 3:
            status = "🔥 HOT"
            advice = "🟢 ĐÁNH MẠNH"
        elif cnt10[pair] == 2:
            status = "🌤 WARM"
            advice = "🟡 ĐÁNH NHẸ"
        else:
            status = "❄️ COLD"
            advice = "🔴 BỎ"

        results.append({
            "pair": pair,
            "score": score,
            "status": status,
            "advice": advice
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ================= BACKTEST =================
def backtest(df, pair, lookback=30):
    pair = safe_pair(pair)
    if not pair:
        return 0, 0
    last = df.tail(lookback)["pair"].tolist()
    hit = last.count(pair)
    rate = round(hit / len(last) * 100, 2) if last else 0
    return hit, rate

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3.1 (ỔN ĐỊNH)")

raw = st.text_area("📥 Nhập kết quả (mỗi dòng 1 số 5 chữ số)", height=120)

if st.button("💾 LƯU KỲ"):
    nums = re.findall(r"\d{5}", raw)
    pairs = [n[-2:] for n in nums]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Sai dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
df["pair"] = df["pair"].apply(safe_pair)
df = df.dropna()

st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ================= RESULT =================
analysis = analyze_v3(df)

if not analysis:
    st.warning("⛔ Chưa đủ dữ liệu hoặc dữ liệu lỗi – AI tạm dừng")
    st.stop()

st.subheader("🔥 TOP CẶP ĐỀ XUẤT")
st.table(pd.DataFrame(analysis[:5]))

best = analysis[0]
pair = safe_pair(best["pair"])

if not pair:
    st.error("⛔ Cặp lỗi – AI dừng an toàn")
    st.stop()

dau, duoi = pair[0], pair[1]
hit, rate = backtest(df, pair)

st.subheader("🧠 KẾT LUẬN AI")
st.markdown(f"""
**Cặp đề xuất:** `{pair}`  
**Xác suất AI:** `{best['score']}%`  
**Backtest:** `{rate}%`  
**Trạng thái:** {best['status']}  
**Khuyến nghị:** {best['advice']}  
**Khả năng về tay:** `{dau}` – `{duoi}`
""")

if rate >= 25:
    st.success("✅ Có thể vào tiền")
else:
    st.warning("⚠️ Nên quan sát thêm")

if st.button("📌 AI HỌC KỲ"):
    update_ai(pair, win=(rate >= 25))
    st.success("AI đã học xong")
