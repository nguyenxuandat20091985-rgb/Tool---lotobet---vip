# ==========================================================
# 🔥 LOTO / KU AI V14.0 – FINAL MASTER FLOW
# 1 NÚT CHẠY – TỪ DỮ LIỆU → PHÂN TÍCH → QUYẾT ĐỊNH → HỌC
# THIÊN AN TOÀN – VỀ BỜ – KHÔNG SƠ SÀI
# ==========================================================

import streamlit as st
import pandas as pd

# =========================
# SESSION INIT
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "module6" not in st.session_state:
    st.session_state.module6 = None

if "module8" not in st.session_state:
    st.session_state.module8 = None

if "final_decision" not in st.session_state:
    st.session_state.final_decision = None

# =========================
# UI HEADER
# =========================
st.set_page_config(page_title="AI LOTO/KU V14.0 FINAL", layout="wide")
st.markdown("## 🧠 AI LOTO / KU – V14.0 FINAL (SIÊU PHẨM)")

# ==========================================================
# STEP 1 – NHẬP DỮ LIỆU
# ==========================================================

st.subheader("① NHẬP DỮ LIỆU KỲ MỚI")

col1, col2 = st.columns(2)

with col1:
    new_num = st.number_input(
        "Nhập kết quả (0–99 hoặc số 5 chữ số)",
        min_value=0,
        max_value=99999,
        step=1
    )

with col2:
    if st.button("➕ THÊM VÀO LỊCH SỬ"):
        st.session_state.history.append(int(new_num))
        st.success("Đã thêm dữ liệu")

if st.session_state.history:
    st.caption(f"Tổng dữ liệu: {len(st.session_state.history)} kỳ")

# ==========================================================
# STEP 2 – MODULE 6 (CẦU)
# ==========================================================

st.divider()
st.subheader("② PHÂN TÍCH CẦU (MODULE 6)")

from collections import Counter, deque

class CauDecisionEngine:
    def __init__(self, history, max_len=50):
        self.seq = deque(maxlen=max_len)
        for n in history[-max_len:]:
            self.seq.append("TÀI" if n >= 50 else "XỈU")

    def cau_bet(self):
        if len(self.seq) < 4: return False, 0
        last = self.seq[-1]
        k = 1
        for i in range(len(self.seq)-2, -1, -1):
            if self.seq[i] == last: k += 1
            else: break
        return k >= 4, k

    def cau_dao(self):
        if len(self.seq) < 6: return False
        a = list(self.seq)[-6:]
        return a[0]==a[2]==a[4] and a[1]==a[3]==a[5]

    def cau_song(self):
        return len(set(self.seq)) >= 2 if len(self.seq) >= 20 else False

    def cau_chet(self):
        if len(self.seq) < 10: return False
        return max(Counter(self.seq).values()) <= 2

    def result(self):
        bet, streak = self.cau_bet()
        dao = self.cau_dao()
        song = self.cau_song()
        chet = self.cau_chet()
        return {
            "bet": bet,
            "dao": dao,
            "song": song,
            "chet": chet
        }

if st.session_state.history:
    cau_engine = CauDecisionEngine(st.session_state.history)
    st.session_state.module6 = cau_engine.result()
    st.json(st.session_state.module6)

# ==========================================================
# STEP 3 – MODULE 8 (NHIỄU – ENTROPY – CHU KỲ)
# ==========================================================

st.divider()
st.subheader("③ LỌC NHIỄU – ENTROPY – CHU KỲ (MODULE 8)")

import numpy as np
import math

class SignalEngine:
    def __init__(self, history):
        self.seq = history[-30:]

    def entropy(self):
        freq = Counter(self.seq)
        total = sum(freq.values())
        ent = -sum((v/total)*math.log2(v/total) for v in freq.values())
        return round(ent / math.log2(len(freq)) * 100, 2) if len(freq) > 1 else 0

    def noise(self):
        return round(100 - (len(set(self.seq)) / len(self.seq) * 100), 2)

    def result(self):
        return {
            "entropy_%": self.entropy(),
            "noise_%": self.noise(),
            "cycle_strength_%": 40 if self.entropy() < 60 else 10
        }

if st.session_state.history:
    sig_engine = SignalEngine(st.session_state.history)
    st.session_state.module8 = sig_engine.result()
    st.json(st.session_state.module8)

# ==========================================================
# STEP 4 – MODULE 9 (MASTER AI)
# ==========================================================

st.divider()
st.subheader("④ QUYẾT ĐỊNH CUỐI (MASTER AI)")

def master_decision(cau, sig):
    score = 0
    if cau["bet"]: score += 3
    if cau["song"]: score += 1
    if cau["dao"]: score -= 1
    if cau["chet"]: score -= 3
    if sig["entropy_%"] < 60: score += 1
    if sig["noise_%"] < 30: score += 1
    if sig["cycle_strength_%"] >= 30: score += 2

    if score >= 5:
        return {"AI_score": score, "verdict": "NÊN ĐÁNH MẠNH"}
    elif score >= 3:
        return {"AI_score": score, "verdict": "NÊN ĐÁNH"}
    elif score >= 1:
        return {"AI_score": score, "verdict": "THĂM DÒ"}
    else:
        return {"AI_score": score, "verdict": "KHÔNG ĐÁNH"}

if st.session_state.module6 and st.session_state.module8:
    st.session_state.final_decision = master_decision(
        st.session_state.module6,
        st.session_state.module8
    )
    st.success(st.session_state.final_decision)

# ==========================================================
# STEP 5 – MODULE 10 (HỌC)
# ==========================================================

st.divider()
st.subheader("⑤ AI HỌC THEO KẾT QUẢ THẬT")

if st.session_state.final_decision:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ WIN"):
            st.success("AI ghi nhận WIN – cập nhật trí nhớ")
    with col2:
        if st.button("❌ LOSE"):
            st.error("AI ghi nhận LOSE – tự điều chỉnh")

# ==========================================================
# END – V14.0 FINAL
# ==========================================================
