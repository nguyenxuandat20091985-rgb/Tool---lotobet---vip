import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ================= CONFIG =================
st.set_page_config("🤖 LOTOBET V8 – TRỢ LÝ", layout="wide")

DATA = "data.csv"
LOG  = "log.csv"

# ================= INIT FILES =================
def init_files():
    if not os.path.exists(DATA):
        pd.DataFrame({"time":[], "result":[]}).to_csv(DATA, index=False)
    if not os.path.exists(LOG):
        pd.DataFrame({
            "time":[], "main":[], "backup":[],
            "safe":[], "decision":[], "note":[]
        }).to_csv(LOG, index=False)

init_files()

# ================= LOAD DATA (SAFE) =================
df = pd.read_csv(DATA)

# Chuẩn hóa cột
df.columns = [c.lower().strip() for c in df.columns]

# Tự phát hiện cột kết quả
if "result" not in df.columns:
    st.error("❌ Không tìm thấy cột kết quả. Dữ liệu bị lỗi.")
    st.stop()

df["result"] = df["result"].astype(str)

# ================= CORE ANALYSIS =================
def analyze(df):
    total = len(df)
    rows = []

    for i in range(100):
        pair = f"{i:02d}"

        hits_idx = df[df["result"].str.contains(pair, na=False)].index.tolist()
        freq = len(hits_idx)
        gap = total - hits_idx[-1] - 1 if freq else total

        # Bệt (liên tiếp 7 kỳ)
        streak = 0
        for r in reversed(df.tail(7)["result"].tolist()):
            if pair in r:
                streak += 1
            else:
                break

        prob = round(freq / total * 100, 2) if total else 0

        safe = (
            100
            - gap * 4
            - max(0, streak - 2) * 15
            + prob * 2
        )

        rows.append({
            "Cặp": pair,
            "Gap": gap,
            "Bệt": streak,
            "%": prob,
            "SAFE": round(max(0, min(100, safe)), 2)
        })

    return pd.DataFrame(rows).sort_values("SAFE", ascending=False)

def assistant(safe, streak):
    if streak >= 3:
        return "🔴 NGHỈ (BỆT)"
    if safe >= 70:
        return "🟢 ĐÁNH"
    if safe >= 55:
        return "🟡 GIẢM TIỀN"
    return "🔴 NGHỈ"

# ================= UI =================
st.title("🤖 LOTOBET V8 – TRỢ LÝ KIẾM TIỀN AN TOÀN")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📥 Nhập kết quả 5 tinh")
    r = st.text_input("Ví dụ: 57221")

    if st.button("LƯU KỲ"):
        if r.isdigit() and len(r) == 5:
            pd.DataFrame({
                "time":[datetime.now()],
                "result":[r]
            }).to_csv(DATA, mode="a", header=False, index=False)
            st.success("✅ Đã lưu")
            st.rerun()
        else:
            st.error("❌ Cần đúng 5 chữ số")

with col2:
    if len(df) < 25:
        st.warning("⚠️ Ít dữ liệu → TRỢ LÝ KHUYÊN NGHỈ")
    else:
        ana = analyze(df)

        main = ana.iloc[0]
        backup = ana.iloc[1]

        decision = assistant(main["SAFE"], main["Bệt"])

        st.success(f"""
🎯 **Cặp chính:** {main['Cặp']}  
🎯 **Cặp phụ:** {backup['Cặp']}  

📊 **SAFE:** {main['SAFE']}  
🔥 **Bệt:** {main['Bệt']}  
🧠 **TRỢ LÝ:** {decision}

💰 **Vốn:** 5–10%  
⛔ **Luật:** Thua 2 tay → DỪNG
""")

        pd.DataFrame({
            "time":[datetime.now()],
            "main":[main["Cặp"]],
            "backup":[backup["Cặp"]],
            "safe":[main["SAFE"]],
            "decision":[decision],
            "note":[""]
        }).to_csv(LOG, mode="a", header=False, index=False)

        st.subheader("📊 Top cặp an toàn")
        st.dataframe(ana.head(10), use_container_width=True)

st.subheader("🕒 Nhật ký trợ lý")
st.dataframe(pd.read_csv(LOG).tail(10), use_container_width=True)
