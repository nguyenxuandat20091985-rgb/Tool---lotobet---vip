import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config("🤖 LOTOBET V8.1 – TRỢ LÝ", layout="wide")

DATA = "data.csv"
LOG  = "log.csv"

# ================= INIT =================
def init_files():
    if not os.path.exists(DATA):
        pd.DataFrame([["",""]], columns=["time","result"]).to_csv(DATA, index=False)
    if not os.path.exists(LOG):
        pd.DataFrame(columns=["time","main","backup","safe","decision","note"]).to_csv(LOG, index=False)

init_files()

# ================= SAFE LOAD =================
def load_data():
    try:
        df = pd.read_csv(DATA, header=None)
    except:
        return pd.DataFrame(columns=["time","result"])

    # Nếu file có header chuẩn
    if df.iloc[0].astype(str).str.contains("result|kết|ket", case=False).any():
        df = pd.read_csv(DATA)
        df.columns = [c.lower().strip() for c in df.columns]
    else:
        # Không có header → gán cứng
        df.columns = ["time","result"]

    # Lọc dữ liệu hợp lệ
    df = df[df["result"].astype(str).str.match(r"^\d{5}$", na=False)]
    df["result"] = df["result"].astype(str)

    return df

df = load_data()

# ================= CORE =================
def analyze(df):
    total = len(df)
    rows = []

    for i in range(100):
        pair = f"{i:02d}"
        hits = df[df["result"].str.contains(pair, na=False)]

        freq = len(hits)
        gap = total - hits.index[-1] - 1 if freq else total

        # Bệt
        streak = 0
        for r in reversed(df.tail(7)["result"].tolist()):
            if pair in r:
                streak += 1
            else:
                break

        pct = round(freq / total * 100, 2) if total else 0

        safe = max(0, min(100,
            60
            + pct * 2
            - gap * 4
            - max(0, streak - 2) * 20
        ))

        rows.append({
            "Cặp": pair,
            "Gap": gap,
            "Bệt": streak,
            "%": pct,
            "SAFE": round(safe,2)
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
st.title("🤖 LOTOBET V8.1 – TRỢ LÝ KIẾM TIỀN AN TOÀN")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📥 Nhập kết quả 5 tinh")
    r = st.text_input("Ví dụ: 57221")

    if st.button("LƯU KỲ"):
        if r.isdigit() and len(r) == 5:
            pd.DataFrame([[datetime.now(), r]], columns=["time","result"])\
              .to_csv(DATA, mode="a", header=False, index=False)
            st.success("✅ Đã lưu kỳ")
            st.rerun()
        else:
            st.error("❌ Cần đúng 5 chữ số")

with col2:
    if len(df) < 20:
        st.warning("⚠️ Ít dữ liệu → TRỢ LÝ KHUYÊN NGHỈ")
    else:
        ana = analyze(df)
        main, backup = ana.iloc[0], ana.iloc[1]

        decision = assistant(main["SAFE"], main["Bệt"])

        st.success(f"""
🎯 **Cặp chính:** {main['Cặp']}  
🎯 **Cặp phụ:** {backup['Cặp']}  

📊 SAFE: {main['SAFE']}  
🔥 Bệt: {main['Bệt']}  
🧠 Trợ lý: {decision}

💰 Vốn: 5–10%  
⛔ Thua 2 tay → NGHỈ
""")

        pd.DataFrame([[datetime.now(), main["Cặp"], backup["Cặp"], main["SAFE"], decision, ""]],
            columns=["time","main","backup","safe","decision","note"])\
            .to_csv(LOG, mode="a", header=False, index=False)

        st.dataframe(ana.head(10), use_container_width=True)

st.subheader("🕒 Nhật ký trợ lý")
st.dataframe(pd.read_csv(LOG).tail(10), use_container_width=True)
