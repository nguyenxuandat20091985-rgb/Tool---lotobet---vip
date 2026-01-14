import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V3 FIX",
    layout="centered",
    page_icon="🎯"
)

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
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

def normalize_pair(p):
    return str(p).zfill(2)

# ================= SAVE DATA =================
def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"time": now, "pair": normalize_pair(p)} for p in pairs]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save_csv(df, DATA_FILE)

# ================= AI MEMORY =================
def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, win=True):
    pair = normalize_pair(pair)
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.2 if win else -0.1
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    save_csv(ai, AI_FILE)

# ================= ANALYSIS CORE =================
def analyze_v3(df):
    df["pair"] = df["pair"].apply(normalize_pair)

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
            "appear_10": cnt10[pair],
            "appear_20": cnt20[pair],
            "score": score,
            "status": status,
            "advice": advice
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ================= BACKTEST =================
def backtest(df, test_pair, lookback=30):
    test_pair = normalize_pair(test_pair)
    last = df.tail(lookback)["pair"].tolist()
    hit = last.count(test_pair)
    rate = round(hit / len(last) * 100, 2) if last else 0
    return hit, rate

# ================= UI =================
st.title("🎯 LOTOBET AUTO PRO – V3 (STABLE)")

raw = st.text_area("📥 Dán kết quả (mỗi dòng 1 số 5 chữ số)", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    nums = re.findall(r"\d{5}", raw)
    pairs = [n[-2:] for n in nums]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Không nhận diện được dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
df["pair"] = df["pair"].apply(normalize_pair)

st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ================= ANALYZE =================
if len(df) >= 40:
    analysis = analyze_v3(df)

    st.subheader("🔥 TOP CẶP ĐỀ XUẤT")
    st.table(pd.DataFrame(analysis[:5]))

    best = analysis[0]
    pair_str = normalize_pair(best["pair"])
    dau, duoi = pair_str[0], pair_str[1]

    hit, rate = backtest(df, pair_str)

    st.subheader("🧠 KẾT LUẬN AI")
    st.markdown(f"""
    **Cặp đề xuất:** `{pair_str}`  
    **Xác suất AI:** `{best['score']}%`  
    **Backtest:** `{rate}%`  
    **Trạng thái:** {best['status']}  
    **Khuyến nghị:** {best['advice']}  
    **Khả năng về tay:** `{dau}` – `{duoi}`
    """)

    if rate >= 25:
        st.success("✅ Có thể xuống tiền")
    else:
        st.warning("⚠️ Nên theo dõi thêm")

    if st.button("📌 AI HỌC KỲ NÀY"):
        update_ai(pair_str, win=(rate >= 25))
        st.success("AI đã cập nhật trí nhớ")

    st.subheader("🎯 DÀN GỢI Ý")
    st.write("Dàn 1:", [x["pair"] for x in analysis[:1]])
    st.write("Dàn 3:", [x["pair"] for x in analysis[:3]])
    st.write("Dàn 5:", [x["pair"] for x in analysis[:5]])

else:
    st.warning("Cần tối thiểu 40 kỳ để AI phân tích")
