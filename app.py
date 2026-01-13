import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
import cv2
import re
from PIL import Image
from collections import Counter
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="LOTOBET AI – 2 SỐ 5 TINH",
    layout="centered"
)

DATA_DIR = "data"
ALL_DATA = f"{DATA_DIR}/data_all.csv"
NEW_DATA = f"{DATA_DIR}/data_new.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ================== STYLE ==================
st.markdown("""
<style>
body {background:#0e1117;color:white;}
.big-title {font-size:22px;font-weight:700;color:#00e676;text-align:center;}
.card {background:#1e1e2f;padding:15px;border-radius:14px;margin-top:12px;}
.num {font-size:32px;color:#00e5ff;font-weight:bold;text-align:center;}
.warn {background:#4b0000;color:#ff4b4b;padding:10px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ================== DATA ==================
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["pair"])

def save_data(new_pairs):
    df_new = pd.DataFrame(new_pairs, columns=["pair"])
    df_new.to_csv(NEW_DATA, mode="a", header=not os.path.exists(NEW_DATA), index=False)

    df_all = load_csv(ALL_DATA)
    df_all = pd.concat([df_all, df_new]).drop_duplicates()
    df_all.to_csv(ALL_DATA, index=False)

# ================== OCR ==================
def ocr_2so_5tinh(image):
    img = np.array(image.convert("L"))
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        img,
        config="--psm 6 -c tessedit_char_whitelist=0123456789"
    )

    digits = re.findall(r"\d", text)
    rows = [digits[i:i+5] for i in range(0, len(digits), 5)]

    pairs = []
    for r in rows:
        if len(r) == 5:
            pairs.append(int(r[-2] + r[-1]))
    return pairs

# ================== ANALYSIS ==================
def analyze_top_3(df):
    nums = []
    for p in df["pair"]:
        nums.extend([p // 10, p % 10])

    counter = Counter(nums)
    hot = [n for n, _ in counter.most_common(6)]

    pairs = []
    for i in range(0, 6, 2):
        pairs.append((hot[i], hot[i+1]))
    return pairs

def detect_bet(df):
    recent = df.tail(10)["pair"].tolist()
    c = Counter(recent)
    return [k for k, v in c.items() if v >= 3]

# ================== UI ==================
st.markdown("<div class='big-title'>🎯 LOTOBET AI – 2 SỐ 5 TINH</div>", unsafe_allow_html=True)

# -------- INPUT TEXT --------
with st.expander("📥 NẠP DỮ LIỆU TEXT"):
    raw = st.text_area("Dán kết quả 5 tinh (mỗi dòng 1 kỳ)", height=120)
    if st.button("🚀 NẠP TEXT"):
        digits = re.findall(r"\d", raw)
        rows = [digits[i:i+5] for i in range(0, len(digits), 5)]
        pairs = [int(r[-2] + r[-1]) for r in rows if len(r) == 5]
        if pairs:
            save_data(pairs)
            st.success(f"Đã nạp {len(pairs)} kỳ")

# -------- INPUT IMAGE --------
with st.expander("📷 NẠP DỮ LIỆU HÌNH ẢNH"):
    img = st.file_uploader("Upload ảnh Lotobet", type=["png","jpg","jpeg"])
    if img:
        image = Image.open(img)
        pairs = ocr_2so_5tinh(image)
        if pairs:
            save_data(pairs)
            st.success(f"Đã quét {len(pairs)} kỳ từ ảnh")
        else:
            st.error("Không nhận diện được ảnh")

# -------- DATA INFO --------
df_all = load_csv(ALL_DATA)
df_new = load_csv(NEW_DATA)

st.info(f"📊 Tổng dữ liệu: {len(df_all)} kỳ | 🆕 Mới: {len(df_new)}")

# -------- ANALYZE --------
if st.button("🔮 PHÂN TÍCH KỲ TIẾP"):
    if len(df_all) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
    else:
        bet = detect_bet(df_all)
        if bet:
            st.markdown(f"<div class='warn'>🚨 CẦU BỆT: {', '.join(map(str, bet))}</div>", unsafe_allow_html=True)

        top3 = analyze_top_3(df_all)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 3 CẶP 2 SỐ 5 TINH MẠNH NHẤT")
        for a, b in top3:
            st.markdown(f"<div class='num'>{a} - {b}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚠️ Công cụ thống kê – không đảm bảo trúng. Quản lý vốn là trên hết.")
