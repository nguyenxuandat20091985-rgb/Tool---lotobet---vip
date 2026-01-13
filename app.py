import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config("LOTOBET V7 – TRỢ LÝ", layout="wide")

DATA = "data.csv"
LOG  = "log.csv"

for f, cols in [
    (DATA, ["Time","Result"]),
    (LOG, ["Time","Main","Backup","Decision","SAFE","Note"])
]:
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

# ===== CORE AI =====
def analyze(df):
    total = len(df)
    res = []

    for i in range(100):
        p = f"{i:02d}"
        hits = df[df["Result"].str.contains(p)]
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

        res.append({
            "Cặp": p,
            "Gap": gap,
            "Bệt": streak,
            "%": prob,
            "SAFE": round(max(0, min(100, safe)), 2)
        })

    return pd.DataFrame(res).sort_values("SAFE", ascending=False)

def assistant_decision(row):
    if row["SAFE"] >= 70:
        return "🟢 ĐÁNH"
    elif row["SAFE"] >= 55:
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
            st.success("Đã lưu kết quả")
            st.rerun()
        else:
            st.error("Sai định dạng (cần đúng 5 số)")

with col2:
    df = pd.read_csv(DATA)

    if len(df) < 20:
        st.warning("⚠️ Dữ liệu < 20 kỳ → TRỢ LÝ KHUYÊN NGHỈ")
    else:
        ana = analyze(df)
        pick = ana.head(2)

        decision = assistant_decision(pick.iloc[0])

        note = ""
        if pick.iloc[0]["Bệt"] >= 3:
            decision = "🔴 NGHỈ"
            note = "Bệt quá sâu – rủi ro cao"

        st.success(f"""
🎯 **Cặp chính:** {pick.iloc[0]['Cặp']}  
🎯 **Cặp phụ:** {pick.iloc[1]['Cặp']}  

🧠 **TRỢ LÝ:** {decision}  
📊 **SAFE:** {pick.iloc[0]['SAFE']}  
💰 **Vốn:** 5–10% / tay  
📌 **Luật:** Thua 2 tay → DỪNG
""")

        pd.DataFrame({
            "Time": [datetime.now()],
            "Main": [pick.iloc[0]["Cặp"]],
            "Backup": [pick.iloc[1]["Cặp"]],
            "Decision": [decision],
            "SAFE": [pick.iloc[0]["SAFE"]],
            "Note": [note]
        }).to_csv(LOG, mode="a", header=False, index=False)

        st.subheader("📊 Top cặp an toàn nhất")
        st.dataframe(ana.head(10), use_container_width=True)

st.subheader("🕒 Nhật ký trợ lý")
st.dataframe(pd.read_csv(LOG).tail(10), use_container_width=True)
