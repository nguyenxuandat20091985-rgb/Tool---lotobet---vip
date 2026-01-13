import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET AUTO PRO – V4",
    layout="centered",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "data.csv"
LOG_FILE = "predict_log.csv"
AI_FILE = "ai_weight.csv"

# ================= STYLE =================
st.markdown("""
<style>
.big-title {font-size:32px;font-weight:800;color:#00ff99;text-align:center;}
.card {background:#0f172a;padding:20px;border-radius:14px;margin-bottom:15px;}
.metric {font-size:20px;font-weight:700;color:#38bdf8;}
.good {color:#22c55e;font-weight:700;}
.warn {color:#eab308;font-weight:700;}
.bad {color:#ef4444;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ================= LOAD / SAVE =================
def load_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def save_pairs(pairs):
    df = load_csv(DATA_FILE, ["time", "pair"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([{"time": now, "pair": p} for p in pairs])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def load_ai():
    return load_csv(AI_FILE, ["pair", "weight"])

def update_ai(pair, good=True):
    ai = load_ai()
    if pair not in ai["pair"].values:
        ai.loc[len(ai)] = [pair, 1.0]
    idx = ai[ai["pair"] == pair].index[0]
    ai.loc[idx, "weight"] += 0.15 if good else -0.1
    ai.loc[idx, "weight"] = max(0.2, ai.loc[idx, "weight"])
    ai.to_csv(AI_FILE, index=False)

# ================= CORE ANALYSIS =================
def analyze_v4(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)

    ai = load_ai()
    ai_map = dict(zip(ai["pair"], ai["weight"]))

    rows = []
    for p in cnt_all:
        base = (
            (cnt10[p]/10)*0.45 +
            (cnt20[p]/20)*0.35 +
            (cnt_all[p]/total)*0.20
        )
        score = round(base * ai_map.get(p, 1.0) * 100, 2)

        rows.append({
            "pair": p,
            "10k": cnt10[p],
            "20k": cnt20[p],
            "score": score
        })

    df_rs = pd.DataFrame(rows)
    df_rs = df_rs.sort_values("score", ascending=False)

    # loại cold sâu
    df_rs = df_rs[df_rs["10k"] >= 1]

    return df_rs

def backtest(df, pair, lookback=30):
    hits = 0
    for i in range(1, min(lookback, len(df))):
        if df.iloc[-i]["pair"] == pair:
            hits += 1
    rate = round(hits / lookback * 100, 2)
    return rate

# ================= UI =================
st.markdown('<div class="big-title">🟢 LOTOBET AUTO PRO – V4</div>', unsafe_allow_html=True)

raw = st.text_area("📥 Dán kết quả 5 tỉnh", height=110)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2]+r[-1]) for r in rows if len(r)==5]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Sai định dạng dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
st.markdown(f"📊 **Tổng dữ liệu:** `{len(df)}` kỳ")

# ================= MAIN LOGIC =================
if len(df) >= 40:
    rs = analyze_v4(df)

    top_pair = rs.iloc[0]
    rate = backtest(df, top_pair["pair"])

    # ===== KẾT LUẬN =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚦 KẾT LUẬN AI")
    st.markdown(f"""
    **Cặp đánh chính:** `{top_pair['pair']}`  
    **Score AI:** `{top_pair['score']}%`  
    **Backtest 30 kỳ:** `{rate}%`
    """)

    if rate >= 25:
        st.markdown("🟢 **KHUYẾN NGHỊ: ĐÁNH CHÍNH**", unsafe_allow_html=True)
        update_ai(top_pair["pair"], good=True)
    else:
        st.markdown("🟡 **THEO DÕI – KHÔNG ÉP TIỀN**", unsafe_allow_html=True)
        update_ai(top_pair["pair"], good=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== DÀN 5 =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 DÀN 5 THÔNG MINH")
    dan5 = rs.head(5)
    st.table(dan5[["pair", "score", "10k", "20k"]])
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== TOP CẶP =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔝 TOP CẶP ĐÁNG CHÚ Ý")
    st.table(rs.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("⚠️ Cần tối thiểu 40 kỳ dữ liệu để AI hoạt động chuẩn")
