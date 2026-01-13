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

# ================= ANALYSIS CORE =================
def analyze_v4(df):
    total = len(df)
    last10 = df.tail(10)["pair"].tolist()
    last20 = df.tail(20)["pair"].tolist()
    last50 = df.tail(50)["pair"].tolist()

    cnt_all = Counter(df["pair"])
    cnt10 = Counter(last10)
    cnt20 = Counter(last20)
    cnt50 = Counter(last50)

    results = []
    for pair in cnt_all:
        base = (
            cnt10[pair]/10 * 0.5 +
            cnt20[pair]/20 * 0.3 +
            cnt50[pair]/50 * 0.2
        )

        score = round(base * 100, 2)

        if cnt10[pair] >= 3:
            group = "🔥 HOT"
            action = "ĐÁNH CHÍNH"
        elif cnt10[pair] == 2:
            group = "🌤 ỔN ĐỊNH"
            action = "ĐÁNH PHỤ"
        elif cnt20[pair] >= 2 and cnt10[pair] == 0:
            group = "🎯 BÙNG LẠI"
            action = "GÀI NHẸ"
        else:
            group = "❄️ COLD"
            action = "BỎ"

        results.append({
            "pair": pair,
            "10k": cnt10[pair],
            "20k": cnt20[pair],
            "score": score,
            "group": group,
            "action": action
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return [x for x in results if x["action"] != "BỎ"]

# ================= UI =================
st.markdown(
    "<h2 style='text-align:center;color:#00ff99'>🟢 LOTOBET AUTO PRO – V4</h2>",
    unsafe_allow_html=True
)

raw = st.text_area("📥 Dán kết quả 5 tỉnh", height=120)

if st.button("💾 LƯU KỲ MỚI"):
    digits = re.findall(r"\d", raw)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
    pairs = [int(r[-2]+r[-1]) for r in rows if len(r)==5]
    if pairs:
        save_pairs(pairs)
        st.success(f"Đã lưu {len(pairs)} kỳ")
    else:
        st.error("Không đọc được dữ liệu")

df = load_csv(DATA_FILE, ["time", "pair"])
st.info(f"📊 Tổng dữ liệu: {len(df)} kỳ")

# ================= DASHBOARD =================
if len(df) >= 50:
    data = analyze_v4(df)
    dan5 = data[:5]

    st.markdown("## 🎯 DÀN 5 SỐ AI ĐỀ XUẤT")

    for d in dan5:
        color = "#00ff99" if "CHÍNH" in d["action"] else "#ffd966"
        st.markdown(
            f"""
            <div style="
                border:1px solid {color};
                border-radius:12px;
                padding:12px;
                margin-bottom:10px;
                background-color:#0f1117">
                <h3 style="color:{color}">Cặp {d['pair']} – {d['group']}</h3>
                <p>📊 Score: <b>{d['score']}%</b></p>
                <p>📌 Khuyến nghị: <b>{d['action']}</b></p>
                <p>10 kỳ: {d['10k']} | 20 kỳ: {d['20k']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================= KẾT LUẬN =================
    danh_chinh = [x["pair"] for x in dan5 if x["action"] == "ĐÁNH CHÍNH"]

    st.markdown("## 🚦 KẾT LUẬN CUỐI")
    if danh_chinh:
        st.success(
            f"✅ KỲ NÀY LÊN ĐÁNH CHÍNH: {', '.join(map(str, danh_chinh))}\n\n"
            f"🎯 ĐÁNH DÀN 5: {', '.join(str(x['pair']) for x in dan5)}"
        )
    else:
        st.warning("⚠️ Không có cặp đủ mạnh → NÊN GIỮ TIỀN")

else:
    st.warning("⏳ Cần tối thiểu 50 kỳ để chạy V4 chuẩn")
