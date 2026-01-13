import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config("LOTOBET V7 – TRỢ LÝ", layout="wide")

DATA = "data.csv"
LOG  = "log.csv"

# ===== INIT FILE =====
if not os.path.exists(DATA):
    pd.DataFrame(columns=["Time","Result"]).to_csv(DATA, index=False)

if not os.path.exists(LOG):
    pd.DataFrame(columns=["Time","Main","Backup","Decision","SAFE","Note"]).to_csv(LOG, index=False)

# ===== LOAD + FIX DATA =====
df = pd.read_csv(DATA)

# FIX DATA CŨ (V3, V4, V5)
if "Kết quả" in df.columns:
    df.rename(columns={"Kết quả": "Result"}, inplace=True)
    df.to_csv(DATA, index=False)

if "Result" not in df.columns:
    st.error("❌ File dữ liệu lỗi. Hãy xoá data.csv và chạy lại.")
    st.stop()

df["Result"] = df["Result"].astype(str)

# ===== CORE AI =====
def analyze(df):
    total = len(df)
    rows = []

    for i in range(100):
        p = f"{i:02d}"

        mask = df["Result"].str.contains(p, na=False)
        hits = df[mask]

        freq = len(hits)
        gap = total - hits.index[-1] - 1 if freq else total

        # bệt
        streak = 0
        for r in reversed(df.tail(7)["Result"]):
            if p in r:
                streak += 1
            else:
                break

        prob = round(freq / total * 100, 2)

        safe = (
            100
            - gap * 5
            - max(0, streak - 2) * 12
            + prob * 2
        )

        rows.append({
            "Cặp": p,
            "Gap": gap,
            "Bệt": streak,
            "%": prob,
            "SAFE": round(max(0, min(100, safe)), 2)
        })

    return pd.DataFrame(rows).sort_values("SAFE", ascending=False)

def assistant_decision(safe):
    if safe >= 70:
        return "🟢 ĐÁNH"
    elif safe >= 55:
        return "🟡 GIẢM TIỀN"
    else:
        return "🔴 NGHỈ"

# ===== UI =====
st.title("🤖 LOTOBET V7 – TRỢ LÝ KIẾM TIỀN AN TOÀN")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📥 Nhập kết quả 5 tinh")
    r = st.text_input("Ví dụ: 57221")

    if st.button("LƯU"):
        if r.isdigit() and len(r) == 5:
            pd.DataFrame({
                "Time": [datetime.now()],
                "Result": [r]
            }).to_csv(DATA, mode="a", header=False, index=False)
            st.success("✅ Đã lưu")
            st.rerun()
        else:
            st.error("❌ Cần đúng 5 số")

with col2:
    if len(df) < 20:
        st.warning("⚠️ Dữ liệu < 20 kỳ → TRỢ LÝ KHUYÊN NGHỈ")
    else:
        ana = analyze(df)

        main = ana.iloc[0]
        backup = ana.iloc[1]

        decision = assistant_decision(main["SAFE"])
        note = ""

        if main["Bệt"] >= 3:
            decision = "🔴 NGHỈ"
            note = "Bệt sâu – rủi ro cao"

        st.success(f"""
🎯 **Cặp chính:** {main['Cặp']}  
🎯 **Cặp phụ:** {backup['Cặp']}  

🧠 **TRỢ LÝ:** {decision}  
📊 **SAFE:** {main['SAFE']}  
💰 **Vốn:** 5–10% / tay  
📌 **Luật:** Thua 2 tay → DỪNG
""")

        pd.DataFrame({
            "Time": [datetime.now()],
            "Main": [main["Cặp"]],
            "Backup": [backup["Cặp"]],
            "Decision": [decision],
            "SAFE": [main["SAFE"]],
            "Note": [note]
        }).to_csv(LOG, mode="a", header=False, index=False)

        st.subheader("📊 Top cặp an toàn")
        st.dataframe(ana.head(10), use_container_width=True)

st.subheader("🕒 Nhật ký trợ lý")
st.dataframe(pd.read_csv(LOG).tail(10), use_container_width=True)
