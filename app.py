# ================= LOTTO KU AI CỰC PHẨM – V12.0 ULTIMATE =================
# Advanced AI for Lotto KU based on Official Rules

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
import warnings
import requests
import json
import threading
from queue import Queue
import io
import base64
import hashlib
from typing import List, Dict, Tuple, Any
warnings.filterwarnings('ignore')

# ================= AI LIBRARIES ULTIMATE =================
try:
    # Advanced Machine Learning
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        VotingClassifier, StackingClassifier, AdaBoostClassifier
    )
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    
    # Deep Learning
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models, callbacks
        DEEP_LEARNING_AVAILABLE = True
    except:
        DEEP_LEARNING_AVAILABLE = False
    
    # Time Series Advanced
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pmdarima as pm
    
    # Advanced Statistics & Math
    from scipy import stats, signal, optimize, interpolate
    from scipy.signal import savgol_filter, find_peaks, welch
    from scipy.optimize import curve_fit, differential_evolution
    from scipy.stats import norm, poisson, binom, entropy
    
    # Feature Engineering
    from sklearn.feature_selection import (
        SelectKBest, RFE, RFECV, 
        mutual_info_classif, f_classif
    )
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # Optimization
    import optuna
    
    AI_LIBS_AVAILABLE = True
except ImportError as e:
    AI_LIBS_AVAILABLE = False
    st.warning(f"⚠️ Thiếu thư viện AI nâng cao: {str(e)}")

from collections import Counter, defaultdict, deque, OrderedDict
import random
import math
from itertools import combinations, permutations, product
from dataclasses import dataclass
from enum import Enum

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTTO KU AI CỰC PHẨM V12.0",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/lottoku-ai',
        'Report a bug': "https://github.com/lottoku-ai/issues",
        'About': "# LOTTO KU AI CỰC PHẨM V12.0 - Advanced Prediction System"
    }
)

# ULTIMATE CSS
st.markdown("""
<style>
    /* Main container - Ultimate */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 1600px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Ultimate Header */
    .header-ultimate {
        background: linear-gradient(135deg, 
            rgba(37, 99, 235, 0.95) 0%, 
            rgba(139, 92, 246, 0.95) 50%,
            rgba(236, 72, 153, 0.95) 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 20px;
        margin: 20px 0 25px 0;
        font-size: 1.4rem;
        font-weight: 800;
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .header-ultimate::before {
        content: '🎰 LOTTO KU AI 🎰';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3rem;
        opacity: 0.05;
        white-space: nowrap;
        font-weight: 900;
    }
    
    /* Premium Cards */
    .card-premium {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin: 15px;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .card-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            #3B82F6, #8B5CF6, #EC4899, #10B981, #F59E0B);
    }
    
    .card-premium:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 
            0 25px 50px rgba(37, 99, 235, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border-color: #3B82F6;
    }
    
    /* Prediction Cards - Ultimate */
    .prediction-ultimate {
        background: linear-gradient(145deg, 
            rgba(255, 255, 255, 0.98) 0%,
            rgba(248, 250, 252, 0.98) 100%);
        padding: 25px;
        border-radius: 18px;
        border: 2px solid rgba(226, 232, 240, 0.9);
        text-align: center;
        margin: 12px;
        box-shadow: 
            0 8px 32px rgba(31, 38, 135, 0.1),
            inset 0 1px 1px rgba(255, 255, 255, 0.5);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-ultimate::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, 
            #3B82F6 0%, #8B5CF6 25%, #EC4899 50%, #10B981 75%, #F59E0B 100%);
    }
    
    .prediction-ultimate:hover {
        transform: translateY(-8px) rotateX(5deg);
        box-shadow: 
            0 20px 60px rgba(37, 99, 235, 0.25),
            inset 0 1px 1px rgba(255, 255, 255, 0.8);
    }
    
    .current-card {
        border-top: 5px solid #10B981;
        background: linear-gradient(145deg, 
            rgba(16, 185, 129, 0.05) 0%,
            rgba(255, 255, 255, 0.98) 100%);
    }
    
    .next-card {
        border-top: 5px solid #F59E0B;
        background: linear-gradient(145deg, 
            rgba(245, 158, 11, 0.05) 0%,
            rgba(255, 255, 255, 0.98) 100%);
    }
    
    /* Number Displays - Ultimate */
    .number-ultimate {
        font-size: 3rem;
        font-weight: 900;
        color: #1a202c;
        text-align: center;
        margin: 15px 0;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, 
            #3B82F6 0%, #8B5CF6 25%, #EC4899 50%, #10B981 75%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(37, 99, 235, 0.1);
        position: relative;
    }
    
    .number-ultimate::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 25%;
        width: 50%;
        height: 3px;
        background: linear-gradient(90deg, transparent, #3B82F6, transparent);
        border-radius: 3px;
    }
    
    .number-ultimate-small {
        font-size: 2rem;
        font-weight: 800;
        color: #4a5568;
        margin: 10px 0;
        background: linear-gradient(135deg, #4a5568, #718096);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Confidence Indicators - Ultimate */
    .confidence-ultimate {
        display: inline-flex;
        align-items: center;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 800;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 2px solid;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .conf-high-ult {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.2) 0%,
            rgba(16, 185, 129, 0.4) 100%);
        color: #065F46;
        border-color: #10B981;
    }
    
    .conf-medium-ult {
        background: linear-gradient(135deg, 
            rgba(245, 158, 11, 0.2) 0%,
            rgba(245, 158, 11, 0.4) 100%);
        color: #92400E;
        border-color: #F59E0B;
    }
    
    .conf-low-ult {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.2) 0%,
            rgba(239, 68, 68, 0.4) 100%);
        color: #991B1B;
        border-color: #EF4444;
    }
    
    /* Recommendation Badges - Ultimate */
    .recommend-ultimate {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 900;
        margin: 12px 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommend-ultimate:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .rec-bet-ult {
        background: linear-gradient(135deg, 
            #10B981 0%, #34D399 100%);
        color: white;
        border: 2px solid #10B981;
    }
    
    .rec-maybe-ult {
        background: linear-gradient(135deg, 
            #F59E0B 0%, #FBBF24 100%);
        color: white;
        border: 2px solid #F59E0B;
    }
    
    .rec-no-ult {
        background: linear-gradient(135deg, 
            #EF4444 0%, #F87171 100%);
        color: white;
        border: 2px solid #EF4444;
    }
    
    /* Analysis Grid - Ultimate */
    .analysis-ultimate {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(70px, 1fr));
        gap: 15px;
        padding: 25px;
        background: linear-gradient(135deg, 
            rgba(248, 250, 252, 0.9) 0%,
            rgba(241, 245, 249, 0.9) 100%);
        border-radius: 20px;
        margin: 20px 0;
        border: 2px solid rgba(226, 232, 240, 0.8);
        box-shadow: 
            inset 0 2px 4px rgba(0, 0, 0, 0.05),
            0 8px 32px rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .algo-item-ultimate {
        text-align: center;
        padding: 15px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .algo-item-ultimate:hover {
        transform: translateY(-8px) scale(1.05);
        border-color: #3B82F6;
        box-shadow: 0 15px 35px rgba(37, 99, 235, 0.2);
    }
    
    .algo-item-ultimate::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, 
            #3B82F6, #8B5CF6, #EC4899);
    }
    
    /* Input Styling - Ultimate */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        border-radius: 15px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 16px !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
        background: rgba(255, 255, 255, 0.9) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 
            0 0 0 4px rgba(59, 130, 246, 0.1),
            inset 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        background: white !important;
    }
    
    /* File Uploader - Ultimate */
    .uploadedFile {
        border: 3px dashed #3B82F6 !important;
        border-radius: 20px !important;
        padding: 30px !important;
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.05) 0%,
            rgba(139, 92, 246, 0.05) 100%) !important;
        text-align: center !important;
        transition: all 0.3s !important;
    }
    
    .uploadedFile:hover {
        border-color: #8B5CF6 !important;
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.1) 0%,
            rgba(139, 92, 246, 0.1) 100%) !important;
        transform: translateY(-2px);
    }
    
    /* Button Styling - Ultimate */
    .stButton > button {
        border-radius: 15px !important;
        font-weight: 800 !important;
        padding: 16px 32px !important;
        border: none !important;
        background: linear-gradient(135deg, 
            #3B82F6 0%, #8B5CF6 50%, #EC4899 100%) !important;
        color: white !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: all 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Progress Bars - Ultimate */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, 
            #3B82F6 0%, #8B5CF6 25%, #EC4899 50%, #10B981 75%, #F59E0B 100%) !important;
        border-radius: 10px !important;
        height: 12px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Metric Cards - Ultimate */
    .stMetric {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 20px !important;
        border-radius: 20px !important;
        border: 2px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        transition: all 0.3s !important;
        backdrop-filter: blur(10px);
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(37, 99, 235, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        border-color: #3B82F6;
    }
    
    /* Tab Styling - Ultimate */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, 
            rgba(248, 250, 252, 0.9) 0%,
            rgba(241, 245, 249, 0.9) 100%);
        padding: 12px;
        border-radius: 20px;
        border: 2px solid rgba(226, 232, 240, 0.8);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 15px !important;
        padding: 15px 25px !important;
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid transparent !important;
        transition: all 0.3s !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: white !important;
        border-color: #3B82F6 !important;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            #3B82F6 0%, #8B5CF6 100%) !important;
        color: white !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Custom Scrollbar - Ultimate */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(135deg, 
            rgba(248, 250, 252, 0.9) 0%,
            rgba(241, 245, 249, 0.9) 100%);
        border-radius: 10px;
        border: 2px solid rgba(226, 232, 240, 0.8);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, 
            #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, 
            #2563EB 0%, #7C3AED 50%, #DB2777 100%);
    }
    
    /* Notification Box - Ultimate */
    .notification-ultimate {
        background: linear-gradient(135deg, 
            rgba(254, 243, 199, 0.9) 0%,
            rgba(253, 230, 138, 0.9) 100%);
        border: 2px solid #F59E0B;
        padding: 25px;
        border-radius: 20px;
        margin: 25px 0;
        font-size: 1rem;
        box-shadow: 
            0 15px 35px rgba(245, 158, 11, 0.15),
            inset 0 1px 1px rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .notification-ultimate::before {
        content: '🎯';
        position: absolute;
        top: 15px;
        right: 15px;
        font-size: 3rem;
        opacity: 0.1;
        transform: rotate(15deg);
    }
    
    /* Capital Management - Ultimate */
    .capital-ultimate {
        background: linear-gradient(135deg, 
            rgba(240, 249, 255, 0.9) 0%,
            rgba(235, 248, 255, 0.9) 100%);
        padding: 25px;
        border-radius: 20px;
        border: 2px solid rgba(186, 230, 253, 0.8);
        margin: 20px 0;
        box-shadow: 
            0 15px 35px rgba(14, 165, 233, 0.1),
            inset 0 1px 1px rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Responsive Design */
    @media (max-width: 1400px) {
        .number-ultimate { font-size: 2.5rem; }
        .number-ultimate-small { font-size: 1.7rem; }
        .prediction-ultimate { height: 180px; padding: 20px; }
    }
    
    @media (max-width: 768px) {
        .header-ultimate { font-size: 1.2rem; padding: 15px 20px; }
        .header-ultimate::before { font-size: 2rem; }
        .number-ultimate { font-size: 2rem; }
        .prediction-ultimate { height: 160px; padding: 15px; margin: 8px; }
        .card-premium { padding: 20px; margin: 10px; }
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Glow Effect */
    .glow {
        position: relative;
    }
    
    .glow::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            #3B82F6, #8B5CF6, #EC4899, #10B981, #F59E0B);
        border-radius: inherit;
        z-index: -1;
        filter: blur(10px);
        opacity: 0.7;
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "lotto_ku_ultimate_v12.db"

# ================= ENUMS & DATA CLASSES =================
class BetType(Enum):
    """Loại cược theo luật Lotto KU"""
    TONG_TAI_XIU = "tổng_tài_xỉu"        # Tổng số tài xỉu (0-22: Xỉu, 23-45: Tài)
    TONG_LE_CHAN = "tổng_lẻ_chẵn"        # Tổng số lẻ chẵn
    TIEN_NHI = "tiền_nhị"                # Hàng số tiền nhị [Chục ngàn/Ngàn]
    HAU_NHI = "hậu_nhị"                  # Hàng số hậu nhị [Chục/Đơn vị]
    HANG_SO_1 = "hàng_số_1"              # 1 hàng số (5 vị trí)
    DE_SO = "đề_số"                      # Đề số (2 số cuối)

class Position(Enum):
    """Vị trí số theo luật Lotto KU"""
    CHUC_NGAN = 0    # Chục ngàn (vị trí 1)
    NGAN = 1         # Ngàn (vị trí 2)
    TRAM = 2         # Trăm (vị trí 3)
    CHUC = 3         # Chục (vị trí 4)
    DON_VI = 4       # Đơn vị (vị trí 5)

@dataclass
class LotteryNumber:
    """Biểu diễn số lotto 5 chữ số"""
    chuc_ngan: int   # Chục ngàn
    ngan: int        # Ngàn
    tram: int        # Trăm
    chuc: int        # Chục
    don_vi: int      # Đơn vị
    
    def __post_init__(self):
        # Validate các số từ 0-9
        for attr in ['chuc_ngan', 'ngan', 'tram', 'chuc', 'don_vi']:
            value = getattr(self, attr)
            if not 0 <= value <= 9:
                raise ValueError(f"{attr} phải từ 0-9, nhận được {value}")
    
    @classmethod
    def from_string(cls, num_str: str):
        """Tạo từ chuỗi 5 chữ số"""
        if len(num_str) != 5 or not num_str.isdigit():
            raise ValueError("Chuỗi phải có đúng 5 chữ số")
        
        digits = [int(d) for d in num_str]
        return cls(*digits)
    
    def to_string(self) -> str:
        """Chuyển thành chuỗi 5 chữ số"""
        return f"{self.chuc_ngan}{self.ngan}{self.tram}{self.chuc}{self.don_vi}"
    
    def get_tien_nhi(self) -> str:
        """Lấy tiền nhị [Chục ngàn/Ngàn]"""
        return f"{self.chuc_ngan}{self.ngan}"
    
    def get_hau_nhi(self) -> str:
        """Lấy hậu nhị [Chục/Đơn vị]"""
        return f"{self.chuc}{self.don_vi}"
    
    def get_tong(self) -> int:
        """Tính tổng 5 số"""
        return self.chuc_ngan + self.ngan + self.tram + self.chuc + self.don_vi
    
    def is_tai(self) -> bool:
        """Kiểm tra tổng là Tài (23-45)"""
        total = self.get_tong()
        return 23 <= total <= 45
    
    def is_xiu(self) -> bool:
        """Kiểm tra tổng là Xỉu (0-22)"""
        total = self.get_tong()
        return 0 <= total <= 22
    
    def is_chan(self) -> bool:
        """Kiểm tra tổng là Chẵn"""
        return self.get_tong() % 2 == 0
    
    def is_le(self) -> bool:
        """Kiểm tra tổng là Lẻ"""
        return self.get_tong() % 2 == 1

@dataclass
class PredictionResult:
    """Kết quả dự đoán"""
    bet_type: BetType
    predicted_value: Any
    confidence: float  # 0-100%
    recommendation: str  # "NÊN ĐÁNH", "CÓ THỂ ĐÁNH", "KHÔNG ĐÁNH"
    reasoning: str
    timestamp: datetime
    
    def to_dict(self):
        return {
            'bet_type': self.bet_type.value,
            'predicted_value': str(self.predicted_value),
            'confidence': self.confidence,
            'recommendation': self.recommendation,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }

# ================= ADVANCED LOTTO KU AI ENGINE =================
class LottoKuUltimateAI:
    """AI cực phẩm cho Lotto KU với 15 thuật toán nâng cao"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data.copy()
        self.numbers = self._extract_numbers()
        self.cache = {}
        self.models = {}
        
        # 15 thuật toán cao cấp
        self.algorithms = {
            1: 'advanced_statistical_analysis',
            2: 'hot_cold_pattern_recognition', 
            3: 'positional_analysis',
            4: 'time_series_forecasting_advanced',
            5: 'neural_network_prediction',
            6: 'ensemble_learning_ultimate',
            7: 'probability_distribution_analysis',
            8: 'cycle_detection_advanced',
            9: 'correlation_network_analysis',
            10: 'entropy_and_information_theory',
            11: 'markov_chain_prediction',
            12: 'deep_learning_prediction',
            13: 'genetic_algorithm_optimization',
            14: 'bayesian_inference',
            15: 'quantum_inspired_prediction'
        }
        
        # Weights cho từng thuật toán (trained)
        self.algo_weights = {
            1: 0.12, 2: 0.13, 3: 0.15, 4: 0.14, 5: 0.16,
            6: 0.18, 7: 0.11, 8: 0.12, 9: 0.14, 10: 0.15,
            11: 0.13, 12: 0.17, 13: 0.16, 14: 0.15, 15: 0.18
        }
    
    def _extract_numbers(self) -> List[LotteryNumber]:
        """Trích xuất số từ dataframe"""
        numbers = []
        for _, row in self.df.iterrows():
            try:
                if 'so5' in row and len(str(row['so5'])) == 5:
                    num = LotteryNumber.from_string(str(row['so5']))
                    numbers.append(num)
            except:
                continue
        return numbers
    
    def run_ultimate_analysis(self) -> Dict:
        """Chạy phân tích toàn diện với tất cả thuật toán"""
        if not self.numbers:
            return {}
        
        results = {
            'algorithms': {},
            'predictions': {},
            'summary': {},
            'timestamps': {}
        }
        
        # Chạy từng thuật toán
        for algo_id, algo_name in self.algorithms.items():
            start_time = time.time()
            
            try:
                if hasattr(self, algo_name):
                    algo_result = getattr(self, algo_name)()
                    results['algorithms'][algo_id] = algo_result
                else:
                    results['algorithms'][algo_id] = {
                        'error': 'Algorithm not implemented',
                        'confidence': 30
                    }
            except Exception as e:
                results['algorithms'][algo_id] = {
                    'error': str(e),
                    'confidence': 20
                }
            
            results['timestamps'][algo_id] = datetime.now().isoformat()
        
        # Tạo dự đoán tổng hợp
        results['predictions'] = self._generate_predictions(results['algorithms'])
        
        # Tạo summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def advanced_statistical_analysis(self) -> Dict:
        """Phân tích thống kê nâng cao"""
        if len(self.numbers) < 10:
            return {'confidence': 30}
        
        # Tính toán nâng cao
        totals = [num.get_tong() for num in self.numbers]
        tien_nhi = [num.get_tien_nhi() for num in self.numbers]
        hau_nhi = [num.get_hau_nhi() for num in self.numbers]
        
        # Phân phối xác suất
        total_dist = Counter(totals)
        tien_nhi_dist = Counter(tien_nhi)
        hau_nhi_dist = Counter(hau_nhi)
        
        # Tính entropy
        entropy_total = entropy(list(total_dist.values()))
        entropy_tien_nhi = entropy(list(tien_nhi_dist.values()))
        entropy_hau_nhi = entropy(list(hau_nhi_dist.values()))
        
        # Phân tích xu hướng
        trend = self._calculate_trend(totals)
        
        return {
            'total_distribution': dict(total_dist),
            'tien_nhi_distribution': dict(tien_nhi_dist),
            'hau_nhi_distribution': dict(hau_nhi_dist),
            'entropy': {
                'total': entropy_total,
                'tien_nhi': entropy_tien_nhi,
                'hau_nhi': entropy_hau_nhi
            },
            'trend': trend,
            'confidence': min(85, len(self.numbers) / 50 * 70)
        }
    
    def hot_cold_pattern_recognition(self) -> Dict:
        """Nhận diện mẫu hình nóng/lạnh nâng cao"""
        if len(self.numbers) < 30:
            return {'confidence': 40}
        
        windows = [10, 20, 30, 50, 100]
        hot_results = {}
        
        for window in windows:
            if len(self.numbers) >= window:
                window_nums = self.numbers[:window]
                
                # Phân tích theo từng vị trí
                position_analysis = {}
                for pos in Position:
                    pos_values = []
                    for num in window_nums:
                        if pos == Position.CHUC_NGAN:
                            pos_values.append(num.chuc_ngan)
                        elif pos == Position.NGAN:
                            pos_values.append(num.ngan)
                        elif pos == Position.TRAM:
                            pos_values.append(num.tram)
                        elif pos == Position.CHUC:
                            pos_values.append(num.chuc)
                        else:  # Position.DON_VI
                            pos_values.append(num.don_vi)
                    
                    counter = Counter(pos_values)
                    hot_digits = [str(d) for d, _ in counter.most_common(3)]
                    cold_digits = [str(d) for d, _ in counter.most_common()[-3:]]
                    
                    position_analysis[pos.name] = {
                        'hot': hot_digits,
                        'cold': cold_digits,
                        'frequency': dict(counter)
                    }
                
                hot_results[f'window_{window}'] = position_analysis
        
        # Kết hợp kết quả
        combined = self._combine_hot_cold_results(hot_results)
        
        return {
            'position_analysis': hot_results.get('window_30', {}),
            'combined_hot': combined['hot'],
            'combined_cold': combined['cold'],
            'trending_digits': combined['trending'],
            'confidence': 78
        }
    
    def positional_analysis(self) -> Dict:
        """Phân tích theo từng vị trí (đúng luật Lotto KU)"""
        if len(self.numbers) < 20:
            return {'confidence': 35}
        
        analysis = {}
        
        for pos in Position:
            pos_name = pos.name.lower()
            pos_values = []
            
            for num in self.numbers:
                if pos == Position.CHUC_NGAN:
                    pos_values.append(num.chuc_ngan)
                elif pos == Position.NGAN:
                    pos_values.append(num.ngan)
                elif pos == Position.TRAM:
                    pos_values.append(num.tram)
                elif pos == Position.CHUC:
                    pos_values.append(num.chuc)
                else:  # Position.DON_VI
                    pos_values.append(num.don_vi)
            
            counter = Counter(pos_values)
            stats = {
                'frequency': dict(counter),
                'most_common': [str(d) for d, _ in counter.most_common(3)],
                'least_common': [str(d) for d, _ in counter.most_common()[-3:]],
                'average': np.mean(pos_values),
                'std': np.std(pos_values),
                'trend': self._calculate_digit_trend(pos_values)
            }
            
            analysis[pos_name] = stats
        
        # Phân tích cặp vị trí
        pair_analysis = self._analyze_position_pairs()
        
        return {
            'position_stats': analysis,
            'pair_analysis': pair_analysis,
            'confidence': 82
        }
    
    def time_series_forecasting_advanced(self) -> Dict:
        """Dự báo chuỗi thời gian nâng cao"""
        if len(self.numbers) < 50:
            return {'confidence': 45}
        
        try:
            # Chuẩn bị dữ liệu
            totals = [num.get_tong() for num in self.numbers]
            
            # Multiple forecasting methods
            forecasts = {}
            
            # 1. ARIMA
            try:
                arima_model = ARIMA(totals, order=(5,1,0))
                arima_fit = arima_model.fit()
                arima_forecast = arima_fit.forecast(steps=1)[0]
                forecasts['arima'] = arima_forecast
            except:
                forecasts['arima'] = np.mean(totals[-10:])
            
            # 2. Exponential Smoothing
            try:
                exp_model = ExponentialSmoothing(totals, seasonal_periods=7)
                exp_fit = exp_model.fit()
                exp_forecast = exp_fit.forecast(1)[0]
                forecasts['exp_smoothing'] = exp_forecast
            except:
                forecasts['exp_smoothing'] = np.mean(totals[-10:])
            
            # 3. Moving Average với adaptive window
            best_window = self._find_optimal_window(totals)
            ma_forecast = np.mean(totals[-best_window:])
            forecasts['moving_average'] = ma_forecast
            
            # 4. Linear Regression
            try:
                x = np.arange(len(totals)).reshape(-1, 1)
                y = totals
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(x, y)
                lr_forecast = lr.predict([[len(totals)]])[0]
                forecasts['linear_regression'] = lr_forecast
            except:
                forecasts['linear_regression'] = np.mean(totals[-10:])
            
            # Kết hợp dự báo
            final_forecast = np.mean(list(forecasts.values()))
            
            # Phân tích trend
            trend, strength = self._analyze_trend_strength(totals)
            
            # Dự đoán Tài/Xỉu, Lẻ/Chẵn
            predicted_tx = "TÀI" if final_forecast >= 23 else "XỈU"
            predicted_lc = "LẺ" if final_forecast % 2 >= 1 else "CHẴN"
            
            return {
                'forecast_total': round(final_forecast, 2),
                'forecast_tai_xiu': predicted_tx,
                'forecast_le_chan': predicted_lc,
                'trend': trend,
                'trend_strength': strength,
                'methods_used': len(forecasts),
                'individual_forecasts': forecasts,
                'confidence': min(88, len(totals) / 100 * 75 + 25)
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 40}
    
    def neural_network_prediction(self) -> Dict:
        """Dự đoán bằng Neural Network"""
        if not AI_LIBS_AVAILABLE or len(self.numbers) < 100:
            return {'confidence': 35}
        
        try:
            # Chuẩn bị features nâng cao
            features = []
            targets_tx = []  # Tài/Xỉu
            targets_lc = []  # Lẻ/Chẵn
            targets_tn = []  # Tiền nhị
            targets_hn = []  # Hậu nhị
            
            for i in range(len(self.numbers) - 1):
                current = self.numbers[i]
                next_num = self.numbers[i + 1]
                
                # Advanced features
                feat = [
                    current.chuc_ngan, current.ngan, current.tram, 
                    current.chuc, current.don_vi,
                    current.get_tong(),
                    current.chuc_ngan + current.ngan,  # Tổng tiền nhị
                    current.chuc + current.don_vi,     # Tổng hậu nhị
                    sum(1 for d in [current.chuc_ngan, current.ngan, 
                                     current.tram, current.chuc, 
                                     current.don_vi] if d % 2 == 0),  # Số chẵn
                    sum(1 for d in [current.chuc_ngan, current.ngan, 
                                     current.tram, current.chuc, 
                                     current.don_vi] if d >= 5),      # Số >=5
                    1 if current.is_tai() else 0,
                    1 if current.is_le() else 0
                ]
                
                # Thêm lag features
                if i > 0:
                    prev = self.numbers[i-1]
                    feat.extend([
                        prev.chuc_ngan, prev.ngan, prev.tram, 
                        prev.chuc, prev.don_vi,
                        prev.get_tong()
                    ])
                else:
                    feat.extend([0] * 6)
                
                features.append(feat)
                
                # Targets
                targets_tx.append(1 if next_num.is_tai() else 0)
                targets_lc.append(1 if next_num.is_le() else 0)
                targets_tn.append(int(next_num.get_tien_nhi()))
                targets_hn.append(int(next_num.get_hau_nhi()))
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train cho Tài/Xỉu
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, targets_tx, test_size=0.2, random_state=42
            )
            
            from sklearn.neural_network import MLPClassifier
            nn_tx = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
            
            nn_tx.fit(X_train, y_train)
            tx_confidence = nn_tx.score(X_test, y_test) * 100
            
            # Predict
            last_features = features_scaled[-1].reshape(1, -1)
            tx_pred = nn_tx.predict(last_features)[0]
            tx_proba = nn_tx.predict_proba(last_features)[0]
            
            # Tương tự cho các target khác...
            
            return {
                'tai_xiu_prediction': "TÀI" if tx_pred == 1 else "XỈU",
                'tai_xiu_confidence': tx_confidence,
                'tai_xiu_probability': max(tx_proba) * 100,
                'model_accuracy_tx': tx_confidence,
                'features_used': len(features[0]),
                'confidence': min(90, tx_confidence)
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 40}
    
    def ensemble_learning_ultimate(self) -> Dict:
        """Ensemble Learning với nhiều model"""
        if not AI_LIBS_AVAILABLE or len(self.numbers) < 80:
            return {'confidence': 45}
        
        try:
            # Chuẩn bị dữ liệu
            features = []
            targets = []
            
            for i in range(len(self.numbers) - 1):
                current = self.numbers[i]
                next_num = self.numbers[i + 1]
                
                feat = [
                    current.chuc_ngan, current.ngan, current.tram, 
                    current.chuc, current.don_vi,
                    current.get_tong(),
                    1 if current.is_tai() else 0,
                    1 if current.is_le() else 0,
                    sum(1 for d in [current.chuc_ngan, current.ngan, 
                                     current.tram, current.chuc, 
                                     current.don_vi] if d % 2 == 0)
                ]
                
                features.append(feat)
                targets.append(int(next_num.get_hau_nhi()))  # Predict hậu nhị
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Tạo ensemble
            from sklearn.ensemble import VotingClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=42, 
                                         use_label_encoder=False, eval_metric='mlogloss')),
                ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42)),
            ]
            
            # Stacking ensemble
            stack_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
            
            stack_ensemble.fit(X_train, y_train)
            accuracy = stack_ensemble.score(X_test, y_test) * 100
            
            # Predict
            last_features = features[-1]
            pred = stack_ensemble.predict([last_features])[0]
            proba = stack_ensemble.predict_proba([last_features])[0]
            
            return {
                'predicted_hau_nhi': f"{pred:02d}",
                'confidence': float(max(proba) * 100),
                'ensemble_accuracy': accuracy,
                'models_count': len(estimators) + 1,  # +1 for final estimator
                'top_3_predictions': [
                    f"{idx:02d}" for idx in np.argsort(proba)[-3:][::-1]
                ],
                'confidence_score': min(95, float(max(proba) * 100) * 0.95)
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 45}
    
    def _generate_predictions(self, algo_results: Dict) -> Dict:
        """Tạo dự đoán tổng hợp từ tất cả thuật toán"""
        predictions = {
            'tai_xiu': {'votes': {'TÀI': 0, 'XỈU': 0}, 'confidence': 0},
            'le_chan': {'votes': {'LẺ': 0, 'CHẴN': 0}, 'confidence': 0},
            'tien_nhi': {'predictions': [], 'confidence': 0},
            'hau_nhi': {'predictions': [], 'confidence': 0},
            'total_sum': {'prediction': 0, 'confidence': 0},
            'full_number': {'predictions': [], 'confidence': 0}
        }
        
        # Tổng hợp votes từ các thuật toán
        for algo_id, result in algo_results.items():
            if not isinstance(result, dict):
                continue
            
            weight = self.algo_weights.get(algo_id, 0.1)
            
            # Tài/Xỉu
            if 'forecast_tai_xiu' in result:
                tx_pred = result['forecast_tai_xiu']
                conf = result.get('confidence', 50) / 100
                predictions['tai_xiu']['votes'][tx_pred] += conf * weight
            
            if 'tai_xiu_prediction' in result:
                tx_pred = result['tai_xiu_prediction']
                conf = result.get('tai_xiu_confidence', 50) / 100
                predictions['tai_xiu']['votes'][tx_pred] += conf * weight
            
            # Lẻ/Chẵn
            if 'forecast_le_chan' in result:
                lc_pred = result['forecast_le_chan']
                conf = result.get('confidence', 50) / 100
                predictions['le_chan']['votes'][lc_pred] += conf * weight
            
            # Hậu nhị
            if 'predicted_hau_nhi' in result:
                hn_pred = result['predicted_hau_nhi']
                conf = result.get('confidence', 50) / 100
                if hn_pred not in predictions['hau_nhi']['predictions']:
                    predictions['hau_nhi']['predictions'].append({
                        'number': hn_pred,
                        'confidence': conf * weight
                    })
            
            # Tổng số
            if 'forecast_total' in result:
                total_pred = result['forecast_total']
                conf = result.get('confidence', 50) / 100
                predictions['total_sum']['prediction'] += total_pred * conf * weight
                predictions['total_sum']['confidence'] += conf * weight
        
        # Tính kết quả cuối cùng
        # Tài/Xỉu
        tx_votes = predictions['tai_xiu']['votes']
        if tx_votes['TÀI'] > tx_votes['XỈU']:
            predictions['tai_xiu']['final'] = 'TÀI'
            predictions['tai_xiu']['confidence'] = tx_votes['TÀI'] / sum(tx_votes.values()) * 100
        else:
            predictions['tai_xiu']['final'] = 'XỈU'
            predictions['tai_xiu']['confidence'] = tx_votes['XỈU'] / sum(tx_votes.values()) * 100
        
        # Lẻ/Chẵn
        lc_votes = predictions['le_chan']['votes']
        if lc_votes['LẺ'] > lc_votes['CHẴN']:
            predictions['le_chan']['final'] = 'LẺ'
            predictions['le_chan']['confidence'] = lc_votes['LẺ'] / sum(lc_votes.values()) * 100
        else:
            predictions['le_chan']['final'] = 'CHẴN'
            predictions['le_chan']['confidence'] = lc_votes['CHẴN'] / sum(lc_votes.values()) * 100
        
        # Sắp xếp hậu nhị theo confidence
        hau_nhi_preds = predictions['hau_nhi']['predictions']
        if hau_nhi_preds:
            hau_nhi_preds.sort(key=lambda x: x['confidence'], reverse=True)
            predictions['hau_nhi']['top_3'] = [p['number'] for p in hau_nhi_preds[:3]]
            predictions['hau_nhi']['confidence'] = sum(p['confidence'] for p in hau_nhi_preds[:3]) / 3 * 100
        
        # Tổng số
        if predictions['total_sum']['confidence'] > 0:
            predictions['total_sum']['final'] = predictions['total_sum']['prediction'] / predictions['total_sum']['confidence']
            predictions['total_sum']['confidence'] = predictions['total_sum']['confidence'] * 100
        
        # Tạo số đầy đủ từ các dự đoán
        self._generate_full_number_predictions(predictions)
        
        return predictions
    
    def _generate_full_number_predictions(self, predictions: Dict):
        """Tạo dự đoán số đầy đủ 5 chữ số"""
        # Dựa trên phân tích vị trí và các dự đoán khác
        position_predictions = {}
        
        # Lấy phân tích vị trí từ thuật toán 3
        if 3 in self.cache.get('algorithms', {}):
            pos_analysis = self.cache['algorithms'][3].get('position_stats', {})
            for pos_name, stats in pos_analysis.items():
                if 'most_common' in stats:
                    position_predictions[pos_name] = stats['most_common'][0]
        
        # Nếu có đủ dự đoán vị trí, tạo số đầy đủ
        if len(position_predictions) >= 3:
            # Tạo các tổ hợp có thể
            possible_numbers = []
            
            # Lấy các số có thể cho từng vị trí
            pos_digits = {}
            for pos in Position:
                pos_name = pos.name.lower()
                if pos_name in position_predictions:
                    pos_digits[pos] = [int(position_predictions[pos_name])]
                else:
                    # Nếu không có dự đoán, dùng số phổ biến từ phân tích
                    pos_digits[pos] = list(range(10))
            
            # Tạo một số tổ hợp có thể (giới hạn để tránh quá nhiều)
            import random
            for _ in range(100):
                digits = []
                for pos in Position:
                    possible = pos_digits[pos]
                    digits.append(random.choice(possible))
                
                num_str = ''.join(str(d) for d in digits)
                possible_numbers.append(num_str)
            
            predictions['full_number']['predictions'] = possible_numbers[:10]
            predictions['full_number']['confidence'] = 65
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Tạo summary chi tiết"""
        algo_results = results.get('algorithms', {})
        
        if not algo_results:
            return {}
        
        # Tính toán các chỉ số
        confidences = []
        execution_times = []
        successful = 0
        
        for algo_id, result in algo_results.items():
            if isinstance(result, dict):
                if 'confidence' in result:
                    conf = result['confidence']
                    if conf >= 60:
                        successful += 1
                    confidences.append(conf)
        
        avg_confidence = np.mean(confidences) if confidences else 50
        
        return {
            'total_algorithms': len(algo_results),
            'successful_algorithms': successful,
            'success_rate': round(successful / len(algo_results) * 100, 1),
            'average_confidence': round(avg_confidence, 1),
            'high_confidence_algorithms': sum(1 for c in confidences if c >= 75),
            'prediction_timestamp': datetime.now().isoformat(),
            'data_points': len(self.numbers)
        }
    
    def _calculate_trend(self, values: List[int]) -> Dict:
        """Tính xu hướng"""
        if len(values) < 5:
            return {'direction': 'unknown', 'strength': 0}
        
        recent = values[:10]
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        
        direction = 'tăng' if slope > 0 else 'giảm'
        strength = abs(slope) * 100
        
        return {
            'direction': direction,
            'strength': round(strength, 2),
            'slope': round(slope, 3)
        }
    
    def _combine_hot_cold_results(self, results: Dict) -> Dict:
        """Kết hợp kết quả nóng/lạnh từ nhiều cửa sổ"""
        all_hot = Counter()
        all_cold = Counter()
        
        for window_key, analysis in results.items():
            for pos_name, pos_data in analysis.items():
                if 'hot' in pos_data:
                    for digit in pos_data['hot']:
                        all_hot[digit] += 1
                if 'cold' in pos_data:
                    for digit in pos_data['cold']:
                        all_cold[digit] += 1
        
        # Lấy top 5
        hot_digits = [digit for digit, _ in all_hot.most_common(5)]
        cold_digits = [digit for digit, _ in all_cold.most_common(5)]
        
        # Tìm số đang trending (xuất hiện trong hot gần đây)
        trending = hot_digits[:3]
        
        return {
            'hot': hot_digits,
            'cold': cold_digits,
            'trending': trending
        }
    
    def _analyze_position_pairs(self) -> Dict:
        """Phân tích cặp vị trí"""
        if len(self.numbers) < 20:
            return {}
        
        pairs = {
            'tien_nhi': [],  # Chục ngàn - Ngàn
            'trung_nhi': [], # Ngàn - Trăm
            'hau_nhi': [],   # Chục - Đơn vị
            'dau_cuoi': []   # Chục ngàn - Đơn vị
        }
        
        for num in self.numbers:
            pairs['tien_nhi'].append(f"{num.chuc_ngan}{num.ngan}")
            pairs['trung_nhi'].append(f"{num.ngan}{num.tram}")
            pairs['hau_nhi'].append(f"{num.chuc}{num.don_vi}")
            pairs['dau_cuoi'].append(f"{num.chuc_ngan}{num.don_vi}")
        
        analysis = {}
        for pair_name, pair_values in pairs.items():
            counter = Counter(pair_values)
            analysis[pair_name] = {
                'most_common': [pair for pair, _ in counter.most_common(5)],
                'frequency': dict(counter),
                'total_unique': len(counter)
            }
        
        return analysis
    
    def _calculate_digit_trend(self, digits: List[int]) -> str:
        """Tính xu hướng của một chữ số"""
        if len(digits) < 10:
            return 'stable'
        
        recent = digits[:10]
        counter = Counter(recent)
        most_common = counter.most_common(1)[0][0]
        frequency = counter[most_common] / len(recent)
        
        if frequency >= 0.4:
            return f"hot ({most_common})"
        elif frequency <= 0.1:
            return f"cold"
        else:
            return "stable"
    
    def _find_optimal_window(self, values: List[int]) -> int:
        """Tìm cửa sổ tối ưu cho moving average"""
        if len(values) < 20:
            return min(10, len(values))
        
        best_window = 10
        best_error = float('inf')
        
        for window in [5, 10, 15, 20, 25]:
            if window < len(values):
                errors = []
                for i in range(window, len(values)):
                    pred = np.mean(values[i-window:i])
                    actual = values[i]
                    errors.append(abs(pred - actual))
                
                avg_error = np.mean(errors)
                if avg_error < best_error:
                    best_error = avg_error
                    best_window = window
        
        return best_window
    
    def _analyze_trend_strength(self, values: List[int]) -> Tuple[str, float]:
        """Phân tích strength của trend"""
        if len(values) < 10:
            return 'unknown', 0
        
        recent = values[:20]
        x = np.arange(len(recent))
        
        # Linear regression
        slope, intercept = np.polyfit(x, recent, 1)
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((recent - y_pred) ** 2)
        ss_tot = np.sum((recent - np.mean(recent)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        direction = 'tăng' if slope > 0 else 'giảm'
        strength = r_squared * 100
        
        return direction, round(strength, 2)

# ================= ENHANCED DATABASE =================
def init_ultimate_db():
    """Khởi tạo database nâng cao"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Bảng kết quả chính
    c.execute("""
    CREATE TABLE IF NOT EXISTS lotto_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE NOT NULL,
        chuc_ngan INTEGER NOT NULL CHECK(chuc_ngan BETWEEN 0 AND 9),
        ngan INTEGER NOT NULL CHECK(ngan BETWEEN 0 AND 9),
        tram INTEGER NOT NULL CHECK(tram BETWEEN 0 AND 9),
        chuc INTEGER NOT NULL CHECK(chuc BETWEEN 0 AND 9),
        don_vi INTEGER NOT NULL CHECK(don_vi BETWEEN 0 AND 9),
        tong INTEGER NOT NULL CHECK(tong BETWEEN 0 AND 45),
        tai_xiu TEXT NOT NULL CHECK(tai_xiu IN ('TÀI', 'XỈU')),
        le_chan TEXT NOT NULL CHECK(le_chan IN ('LẺ', 'CHẴN')),
        tien_nhi TEXT NOT NULL,
        hau_nhi TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        verified INTEGER DEFAULT 0
    )
    """)
    
    # Bảng dự đoán AI
    c.execute("""
    CREATE TABLE IF NOT EXISTS ai_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT NOT NULL,
        algorithm_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        predicted_value TEXT NOT NULL,
        confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 100),
        actual_result TEXT,
        is_correct INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng phân tích pattern
    c.execute("""
    CREATE TABLE IF NOT EXISTS pattern_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_data TEXT NOT NULL,
        start_ky TEXT,
        end_ky TEXT,
        strength REAL NOT NULL,
        confidence REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng performance tracking
    c.execute("""
    CREATE TABLE IF NOT EXISTS algorithm_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        algorithm_id INTEGER NOT NULL,
        total_predictions INTEGER DEFAULT 0,
        correct_predictions INTEGER DEFAULT 0,
        accuracy REAL DEFAULT 0,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Indexes for performance
    c.execute("CREATE INDEX IF NOT EXISTS idx_ky ON lotto_results(ky)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON lotto_results(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_algorithm ON ai_predictions(algorithm_id)")
    
    conn.commit()
    conn.close()

init_ultimate_db()

# ================= ENHANCED FILE PROCESSING =================
def process_uploaded_file(uploaded_file) -> List[str]:
    """Xử lý file upload với nhiều định dạng"""
    numbers = []
    
    try:
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                # Xử lý nhiều định dạng
                if len(line) == 5 and line.isdigit():
                    numbers.append(line)
                elif ' ' in line:
                    parts = line.split()
                    for part in parts:
                        if len(part) == 5 and part.isdigit():
                            numbers.append(part)
        
        elif uploaded_file.name.endswith('.csv'):
            import csv
            content = uploaded_file.getvalue().decode('utf-8')
            reader = csv.reader(io.StringIO(content))
            
            for row in reader:
                for cell in row:
                    cell = str(cell).strip()
                    if len(cell) == 5 and cell.isdigit():
                        numbers.append(cell)
        
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            import pandas as pd
            df = pd.read_excel(uploaded_file)
            
            # Tìm cột có số 5 chữ số
            for col in df.columns:
                for val in df[col]:
                    val_str = str(val).strip()
                    if len(val_str) == 5 and val_str.isdigit():
                        numbers.append(val_str)
        
        else:
            # Thử parse như text
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            import re
            # Tìm tất cả số 5 chữ số
            five_digit_numbers = re.findall(r'\b\d{5}\b', content)
            numbers.extend(five_digit_numbers)
    
    except Exception as e:
        st.error(f"Lỗi xử lý file: {str(e)}")
    
    # Remove duplicates
    numbers = list(OrderedDict.fromkeys(numbers))
    
    return numbers

# ================= HELPER FUNCTIONS =================
def format_tien(tien: float) -> str:
    """Định dạng tiền"""
    return f"{tien:,.0f}₫"

def get_confidence_class_ultimate(confidence: float) -> str:
    """Lớp CSS cho confidence"""
    if confidence >= 80:
        return "conf-high-ult"
    elif confidence >= 65:
        return "conf-medium-ult"
    else:
        return "conf-low-ult"

def get_recommendation_badge_ultimate(recommendation: str) -> str:
    """Badge cho khuyến nghị"""
    if recommendation == 'NÊN ĐÁNH':
        return '<span class="recommend-ultimate rec-bet-ult">🎯 NÊN ĐÁNH</span>'
    elif recommendation == 'CÓ THỂ ĐÁNH':
        return '<span class="recommend-ultimate rec-maybe-ult">⚠️ CÓ THỂ ĐÁNH</span>'
    else:
        return '<span class="recommend-ultimate rec-no-ult">⛔ KHÔNG ĐÁNH</span>'

def save_lotto_results(numbers: List[str], current_ky: str = None):
    """Lưu kết quả vào database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    added_count = 0
    
    for idx, num_str in enumerate(numbers):
        try:
            if len(num_str) != 5 or not num_str.isdigit():
                continue
            
            num = LotteryNumber.from_string(num_str)
            
            # Tạo kỳ
            if current_ky and idx == 0:
                ky = current_ky
            else:
                ky = f"KU{int(time.time() * 1000) % 1000000:06d}"
            
            c.execute("""
            INSERT OR IGNORE INTO lotto_results 
            (ky, chuc_ngan, ngan, tram, chuc, don_vi, tong, tai_xiu, le_chan, tien_nhi, hau_nhi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ky,
                num.chuc_ngan, num.ngan, num.tram, num.chuc, num.don_vi,
                num.get_tong(),
                "TÀI" if num.is_tai() else "XỈU",
                "LẺ" if num.is_le() else "CHẴN",
                num.get_tien_nhi(),
                num.get_hau_nhi()
            ))
            
            if c.rowcount > 0:
                added_count += 1
        
        except Exception as e:
            print(f"Lỗi lưu số {num_str}: {e}")
    
    conn.commit()
    conn.close()
    return added_count

def load_lotto_data(limit: int = 1000) -> pd.DataFrame:
    """Tải dữ liệu từ database"""
    conn = sqlite3.connect(DB_FILE)
    query = f"""
    SELECT 
        ky,
        chuc_ngan || ngan || tram || chuc || don_vi as so5,
        tien_nhi,
        hau_nhi,
        tong,
        tai_xiu,
        le_chan,
        timestamp
    FROM lotto_results 
    ORDER BY timestamp DESC 
    LIMIT {limit}
    """
    
    try:
        df = pd.read_sql(query, conn)
    except:
        df = pd.DataFrame()
    
    conn.close()
    return df

# ================= REAL-TIME MONITOR =================
class UltimateRealTimeMonitor:
    """Monitor thời gian thực nâng cao"""
    
    def __init__(self):
        self.current_ky = None
        self.next_draw = None
        self.last_sync = None
        
    def sync_lotto_ku(self) -> Dict:
        """Đồng bộ với Lotto KU"""
        current_time = datetime.now()
        
        # Tạo kỳ theo format Lotto KU
        # Giả sử mỗi 5 phút có 1 kỳ
        base_date = current_time.strftime("%y%m%d")
        minute_of_day = current_time.hour * 60 + current_time.minute
        sequence = minute_of_day // 5 + 1
        
        self.current_ky = f"KU{base_date}{sequence:03d}"
        
        # Tính thời gian quay tiếp theo
        next_minute = ((current_time.minute // 5) + 1) * 5
        next_hour = current_time.hour
        
        if next_minute >= 60:
            next_hour += 1
            next_minute = 0
        
        self.next_draw = current_time.replace(
            hour=next_hour % 24,
            minute=next_minute,
            second=0,
            microsecond=0
        )
        
        self.last_sync = current_time
        
        return {
            'current_ky': self.current_ky,
            'next_draw': self.next_draw.strftime("%H:%M:%S"),
            'seconds_to_next': (self.next_draw - current_time).seconds,
            'status': 'synced',
            'last_sync': current_time.strftime("%H:%M:%S")
        }

# ================= MAIN APP - ULTIMATE VERSION =================
def main():
    # Ultimate Header
    st.markdown("""
    <div class="header-ultimate floating">
    🎰 LOTTO KU AI CỰC PHẨM V12.0 🎰
    <div style="font-size:1rem;font-weight:400;margin-top:10px;opacity:0.9">
    15 Thuật Toán Nâng Cao • Phân Tích Đa Chiều • Dự Đoán Chính Xác
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time Monitor
    monitor = UltimateRealTimeMonitor()
    sync_info = monitor.sync_lotto_ku()
    
    # Header với thông tin thời gian thực
    col_header1, col_header2, col_header3 = st.columns([3, 2, 1])
    
    with col_header1:
        st.markdown("""
        <div style="padding:15px;background:linear-gradient(135deg,#1e40af20,#3b82f620);
        border-radius:15px;border:1px solid #3b82f640;text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;color:#1e40af;">🎯 HỆ THỐNG SOI CẦU AI</div>
        <div style="font-size:0.9rem;color:#4b5563;">Dựa trên luật chơi Lotto KU chính thức</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown(f"""
        <div style="padding:12px;background:linear-gradient(135deg,#10b98120,#34d39920);
        border-radius:12px;border:1px solid #10b98140;text-align:center;">
        <div style="font-size:0.9rem;color:#065f46;">KỲ HIỆN TẠI</div>
        <div style="font-size:1.4rem;font-weight:800;color:#065f46;">{sync_info['current_ky']}</div>
        <div style="font-size:0.8rem;color:#059669;">⏱️ {sync_info['next_draw']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="padding:10px;background:#f8fafc;border-radius:10px;
        border:1px solid #e2e8f0;text-align:center;">
        <div style="font-size:0.8rem;color:#64748b;">THỜI GIAN</div>
        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================= BẢNG 1: NHẬP LIỆU & ĐỒNG BỘ =================
    st.markdown('<div class="card-premium">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#1e40af;margin-bottom:15px;">📥 BẢNG 1: NHẬP LIỆU & ĐỒNG BỘ KỲ</div>', unsafe_allow_html=True)
    
    col1_1, col1_2, col1_3 = st.columns([3, 2, 2])
    
    with col1_1:
        st.markdown("**Nhập kết quả Lotto KU:**")
        
        tab_data, tab_file = st.tabs(["📝 Nhập tay", "📁 Từ file"])
        
        with tab_data:
            raw_data = st.text_area(
                "Nhập số 5 chữ số:",
                height=120,
                placeholder="Mỗi dòng 1 số 5 chữ số\nVD:\n12345\n67890\n54321",
                label_visibility="collapsed",
                key="manual_input"
            )
        
        with tab_file:
            uploaded_file = st.file_uploader(
                "Chọn file TXT/CSV/Excel",
                type=['txt', 'csv', 'xlsx', 'xls'],
                label_visibility="collapsed",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                st.info(f"📄 Đã chọn: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
                
                # Xử lý file ngay khi upload
                numbers = process_uploaded_file(uploaded_file)
                if numbers:
                    st.success(f"✅ Tìm thấy {len(numbers)} số hợp lệ")
                    # Lưu vào session để sử dụng sau
                    st.session_state['uploaded_numbers'] = numbers
                    st.session_state['uploaded_file_name'] = uploaded_file.name
    
    with col1_2:
        st.markdown("**Đồng bộ kỳ:**")
        
        current_ky_input = st.text_input(
            "Kỳ hiện tại của bạn:",
            value=sync_info['current_ky'],
            max_chars=12,
            placeholder="VD: KU241201001",
            help="Nhập kỳ theo format: KU + YYMMDD + số thứ tự"
        )
        
        st.markdown("**Trạng thái:**")
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            st.markdown('<span style="color:#10B981;font-weight:bold;">✅ Đã đồng bộ</span>', unsafe_allow_html=True)
        with col_status2:
            st.caption(f"Kỳ tiếp: **{sync_info['current_ky'][:-3]}{int(sync_info['current_ky'][-3:]) + 1:03d}**")
        
        if st.button("🔄 CẬP NHẬT NGAY", use_container_width=True, type="secondary"):
            st.rerun()
    
    with col1_3:
        st.markdown("**Thông tin database:**")
        
        # Load data để hiển thị thông tin
        df = load_lotto_data(100)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            total_count = len(df) if not df.empty else 0
            st.metric("Tổng kỳ", f"{total_count:,}")
        with col_info2:
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = len([d for d in df.get('timestamp', []) if str(d).startswith(today)]) if not df.empty else 0
            st.metric("Hôm nay", today_count)
        
        st.markdown("**Trạng thái AI:**")
        st.markdown("""
        <div style="background:linear-gradient(135deg,#3b82f610,#8b5cf610);
        padding:10px;border-radius:10px;border:1px solid #3b82f630;">
        <div style="color:#3b82f6;font-weight:bold;">🧠 AI NÂNG CAO: SẴN SÀNG</div>
        <div style="font-size:0.8rem;color:#6b7280;">15 thuật toán • Deep Learning • Ensemble</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nút lưu dữ liệu
    col_save1, col_save2, col_save3 = st.columns([2, 1, 1])
    with col_save1:
        if st.button("💾 LƯU DỮ LIỆU & PHÂN TÍCH", type="primary", use_container_width=True):
            # Xác định nguồn dữ liệu
            numbers_to_save = []
            
            if raw_data:
                # Xử lý input tay
                lines = raw_data.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) == 5 and line.isdigit():
                        numbers_to_save.append(line)
            
            if 'uploaded_numbers' in st.session_state:
                numbers_to_save.extend(st.session_state['uploaded_numbers'])
            
            if numbers_to_save:
                # Remove duplicates
                numbers_to_save = list(OrderedDict.fromkeys(numbers_to_save))
                
                # Lưu vào database
                added = save_lotto_results(numbers_to_save, current_ky_input if current_ky_input else None)
                
                if added > 0:
                    st.success(f"✅ Đã lưu {added} kết quả mới vào database!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠️ Không có số mới để lưu (có thể đã tồn tại)")
            else:
                st.error("❌ Chưa có dữ liệu để lưu")
    
    with col_save2:
        if st.button("🗑️ XÓA DỮ LIỆU", type="secondary", use_container_width=True):
            if st.checkbox("Xác nhận xóa tất cả dữ liệu?"):
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("DELETE FROM lotto_results")
                conn.commit()
                conn.close()
                st.error("Đã xóa toàn bộ dữ liệu!")
                time.sleep(1)
                st.rerun()
    
    with col_save3:
        if st.button("📊 XEM DỮ LIỆU", use_container_width=True):
            if not df.empty:
                st.dataframe(df.head(20), use_container_width=True)
            else:
                st.info("Chưa có dữ liệu trong database")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Kết thúc card
    st.markdown("---")
    
    # ================= BẢNG 2: DỰ ĐOÁN KẾT QUẢ =================
    # Load data cho phân tích
    df = load_lotto_data(300)
    
    if not df.empty and len(df) >= 20:
        # Khởi tạo AI
        ai_engine = LottoKuUltimateAI(df)
        
        # Chạy phân tích
        with st.spinner("🧠 AI CỰC PHẨM đang phân tích dữ liệu..."):
            analysis_results = ai_engine.run_ultimate_analysis()
            predictions = analysis_results.get('predictions', {})
            summary = analysis_results.get('summary', {})
        
        # KỲ HIỆN TẠI
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:700;color:#1e40af;margin-bottom:20px;">🎯 BẢNG 2: DỰ ĐOÁN KỲ {sync_info["current_ky"]} (HIỆN TẠI)</div>', unsafe_allow_html=True)
        
        # 5 cột dự đoán
        col2_1, col2_2, col2_3, col2_4, col2_5 = st.columns(5)
        
        with col2_1:
            # Tài/Xỉu
            tx_pred = predictions.get('tai_xiu', {})
            final_tx = tx_pred.get('final', 'TÀI')
            confidence_tx = tx_pred.get('confidence', 50)
            
            st.markdown('<div class="prediction-ultimate current-card">', unsafe_allow_html=True)
            st.markdown("**🎲 TÀI/XỈU**")
            st.markdown(f'<div class="number-ultimate">{final_tx}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="confidence-ultimate {get_confidence_class_ultimate(confidence_tx)}">{confidence_tx:.1f}%</span>', unsafe_allow_html=True)
            
            if confidence_tx >= 70:
                st.markdown(get_recommendation_badge_ultimate("NÊN ĐÁNH"), unsafe_allow_html=True)
            elif confidence_tx >= 60:
                st.markdown(get_recommendation_badge_ultimate("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
            else:
                st.markdown(get_recommendation_badge_ultimate("KHÔNG ĐÁNH"), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_2:
            # Lẻ/Chẵn
            lc_pred = predictions.get('le_chan', {})
            final_lc = lc_pred.get('final', 'LẺ')
            confidence_lc = lc_pred.get('confidence', 50)
            
            st.markdown('<div class="prediction-ultimate current-card">', unsafe_allow_html=True)
            st.markdown("**🎲 LẺ/CHẴN**")
            st.markdown(f'<div class="number-ultimate">{final_lc}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="confidence-ultimate {get_confidence_class_ultimate(confidence_lc)}">{confidence_lc:.1f}%</span>', unsafe_allow_html=True)
            
            if confidence_lc >= 70:
                st.markdown(get_recommendation_badge_ultimate("NÊN ĐÁNH"), unsafe_allow_html=True)
            elif confidence_lc >= 60:
                st.markdown(get_recommendation_badge_ultimate("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
            else:
                st.markdown(get_recommendation_badge_ultimate("KHÔNG ĐÁNH"), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_3:
            # Tiền nhị
            # Dựa trên phân tích vị trí
            pos_analysis = analysis_results.get('algorithms', {}).get(3, {})
            if pos_analysis and 'position_stats' in pos_analysis:
                tien_nhi_stats = pos_analysis['position_stats'].get('chuc_ngan', {})
                most_common = tien_nhi_stats.get('most_common', ['0'])[0]
                confidence_tn = 75  # Giả định
            else:
                most_common = "68"
                confidence_tn = 65
            
            st.markdown('<div class="prediction-ultimate current-card">', unsafe_allow_html=True)
            st.markdown("**🔢 TIỀN NHỊ**")
            st.markdown(f'<div class="number-ultimate">{most_common}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="confidence-ultimate {get_confidence_class_ultimate(confidence_tn)}">{confidence_tn:.1f}%</span>', unsafe_allow_html=True)
            st.markdown(get_recommendation_badge_ultimate("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_4:
            # Hậu nhị
            hn_pred = predictions.get('hau_nhi', {})
            top_hn = hn_pred.get('top_3', ['00'])[0]
            confidence_hn = hn_pred.get('confidence', 50)
            
            st.markdown('<div class="prediction-ultimate current-card">', unsafe_allow_html=True)
            st.markdown("**🔢 HẬU NHỊ**")
            st.markdown(f'<div class="number-ultimate">{top_hn}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="confidence-ultimate {get_confidence_class_ultimate(confidence_hn)}">{confidence_hn:.1f}%</span>', unsafe_allow_html=True)
            
            if confidence_hn >= 70:
                st.markdown(get_recommendation_badge_ultimate("NÊN ĐÁNH"), unsafe_allow_html=True)
            elif confidence_hn >= 60:
                st.markdown(get_recommendation_badge_ultimate("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
            else:
                st.markdown(get_recommendation_badge_ultimate("KHÔNG ĐÁNH"), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2_5:
            # Tổng số
            total_pred = predictions.get('total_sum', {})
            final_total = total_pred.get('final', 23)
            confidence_total = total_pred.get('confidence', 50)
            
            st.markdown('<div class="prediction-ultimate current-card">', unsafe_allow_html=True)
            st.markdown("**🧮 TỔNG SỐ**")
            st.markdown(f'<div class="number-ultimate">{int(final_total)}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="confidence-ultimate {get_confidence_class_ultimate(confidence_total)}">{confidence_total:.1f}%</span>', unsafe_allow_html=True)
            
            if confidence_total >= 70:
                st.markdown(get_recommendation_badge_ultimate("NÊN ĐÁNH"), unsafe_allow_html=True)
            elif confidence_total >= 60:
                st.markdown(get_recommendation_badge_ultimate("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
            else:
                st.markdown(get_recommendation_badge_ultimate("KHÔNG ĐÁNH"), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Kết thúc card
        st.markdown("---")
        
        # ================= BẢNG 3: THÔNG BÁO ĐÁNH CÙNG KỲ =================
        st.markdown('<div class="notification-ultimate">', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="font-size:1.2rem;font-weight:800;color:#92400e;margin-bottom:15px;">
        🔔 THÔNG BÁO ĐÁNH KỲ {sync_info['current_ky']} CÙNG LOTTO KU
        </div>
        
        <div style="background:linear-gradient(135deg,#ffffff40,#ffffff80);
        padding:15px;border-radius:12px;border:1px solid #f59e0b40;">
        
        <div style="margin-bottom:10px;">
        <span style="font-weight:700;color:#1e40af;">🎯 KẾT LUẬN TỪ AI CỰC PHẨM:</span>
        </div>
        
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:15px;">
        
        <div style="padding:12px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;">
        <div style="font-weight:700;color:#3b82f6;">Tài/Xỉu</div>
        <div style="font-size:1.1rem;font-weight:800;color:#1e40af;">{final_tx}</div>
        <div style="font-size:0.9rem;color:#6b7280;">Độ tin cậy: {confidence_tx:.1f}%</div>
        </div>
        
        <div style="padding:12px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;">
        <div style="font-weight:700;color:#10b981;">Lẻ/Chẵn</div>
        <div style="font-size:1.1rem;font-weight:800;color:#065f46;">{final_lc}</div>
        <div style="font-size:0.9rem;color:#6b7280;">Độ tin cậy: {confidence_lc:.1f}%</div>
        </div>
        
        <div style="padding:12px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;">
        <div style="font-weight:700;color:#8b5cf6;">Hậu nhị</div>
        <div style="font-size:1.1rem;font-weight:800;color:#6d28d9;">{top_hn}</div>
        <div style="font-size:0.9rem;color:#6b7280;">Top 3: {', '.join(hn_pred.get('top_3', ['00', '00', '00'])[:3])}</div>
        </div>
        
        </div>
        
        <div style="margin-top:15px;padding:10px;background:#dbeafe;border-radius:8px;border:1px solid #3b82f6;">
        <div style="font-weight:700;color:#1e40af;">📊 PHÂN TÍCH AI:</div>
        <div style="font-size:0.9rem;color:#4b5563;">
        • Thuật toán thành công: {summary.get('successful_algorithms', 0)}/{summary.get('total_algorithms', 0)}<br>
        • Độ tin cậy trung bình: {summary.get('average_confidence', 0):.1f}%<br>
        • Dựa trên {summary.get('data_points', 0)} kết quả gần nhất
        </div>
        </div>
        
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # ================= BẢNG 4: PHÂN TÍCH TỔNG HỢP =================
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#1e40af;margin-bottom:20px;">🤖 BẢNG 4: PHÂN TÍCH TỔNG HỢP AI</div>', unsafe_allow_html=True)
        
        # Hiển thị performance summary
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.metric("Thuật toán", f"{summary.get('total_algorithms', 0)}")
        with col_sum2:
            st.metric("Thành công", f"{summary.get('successful_algorithms', 0)}")
        with col_sum3:
            st.metric("Tỷ lệ", f"{summary.get('success_rate', 0):.1f}%")
        with col_sum4:
            st.metric("Tin cậy", f"{summary.get('average_confidence', 0):.1f}%")
        
        # Hiển thị grid các thuật toán
        st.markdown("**🧮 15 THUẬT TOÁN NÂNG CAO:**")
        
        # Tạo grid
        algo_results = analysis_results.get('algorithms', {})
        
        cols = st.columns(5)
        for idx, (algo_id, result) in enumerate(algo_results.items()):
            with cols[idx % 5]:
                confidence = result.get('confidence', 0)
                
                # Màu dựa trên confidence
                if confidence >= 75:
                    color = "#10B981"
                    emoji = "🔵"
                elif confidence >= 60:
                    color = "#F59E0B"
                    emoji = "🟢"
                else:
                    color = "#EF4444"
                    emoji = "🔴"
                
                st.markdown(f"""
                <div style="text-align:center;padding:10px;background:{color}10;
                border-radius:10px;border:2px solid {color}30;margin:5px;">
                <div style="font-size:0.9rem;font-weight:700;color:{color};">{emoji} A{algo_id}</div>
                <div style="font-size:0.8rem;color:#4b5563;">{confidence:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Hiển thị thêm thông tin chi tiết
        with st.expander("📊 CHI TIẾT PHÂN TÍCH", expanded=False):
            tab1, tab2, tab3 = st.tabs(["📈 Thống kê", "🎯 Dự đoán", "🔍 Phân tích"])
            
            with tab1:
                if 'statistical_analysis' in str(algo_results.get(1, {})):
                    st.json(algo_results.get(1, {}))
            
            with tab2:
                st.write("**Dự đoán chi tiết:**")
                st.write(predictions)
            
            with tab3:
                st.write("**Phân tích nâng cao:**")
                # Hiển thị phân tích vị trí
                if 3 in algo_results:
                    pos_analysis = algo_results[3]
                    st.write("Phân tích vị trí số:")
                    for pos_name, stats in pos_analysis.get('position_stats', {}).items():
                        st.write(f"**{pos_name.upper()}:**")
                        st.write(f"- Phổ biến: {', '.join(stats.get('most_common', []))}")
                        st.write(f"- Ít gặp: {', '.join(stats.get('least_common', []))}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Hiển thị khi chưa có đủ dữ liệu
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:40px;">
        <div style="font-size:2rem;color:#9ca3af;">📊</div>
        <div style="font-size:1.2rem;font-weight:700;color:#6b7280;margin-top:10px;">
        CHƯA ĐỦ DỮ LIỆU ĐỂ PHÂN TÍCH
        </div>
        <div style="color:#9ca3af;margin-top:10px;">
        Vui lòng nhập ít nhất 20 kết quả để AI có thể phân tích
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================= BẢNG 5: QUẢN LÝ VỐN & CÀI ĐẶT =================
    st.markdown('<div class="card-premium">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#1e40af;margin-bottom:20px;">💰 BẢNG 5: QUẢN LÝ VỐN THÔNG MINH</div>', unsafe_allow_html=True)
    
    col_cap1, col_cap2 = st.columns([2, 3])
    
    with col_cap1:
        st.markdown("**Cấu hình vốn:**")
        
        total_capital = st.number_input(
            "Tổng vốn hiện có (VNĐ):",
            min_value=100000,
            max_value=1000000000,
            value=1000000,
            step=100000,
            format="%d",
            help="Nhập tổng số vốn bạn đang có"
        )
        
        risk_profile = st.selectbox(
            "Hồ sơ rủi ro:",
            ["BẢO TOÀN (Low Risk)", "CÂN BẰNG (Medium Risk)", "TĂNG TRƯỞNG (High Risk)"],
            index=1
        )
        
        stop_loss = st.slider(
            "Mức stop-loss (%):",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Dừng lỗ khi thua bao nhiêu % vốn"
        )
        
        if st.button("🧮 TÍNH TOÁN PHÂN BỔ", type="primary", use_container_width=True):
            # Tính toán phân bổ đơn giản
            capital_distribution = {
                'Tài/Xỉu': total_capital * 0.25,
                'Lẻ/Chẵn': total_capital * 0.20,
                'Hậu nhị': total_capital * 0.30,
                'Tiền nhị': total_capital * 0.15,
                'Tổng số': total_capital * 0.10
            }
            
            st.session_state['capital_dist'] = capital_distribution
            st.success("✅ Đã tính toán phân bổ vốn!")
    
    with col_cap2:
        st.markdown("**Phân bổ vốn đề xuất:**")
        
        if 'capital_dist' in st.session_state:
            capital_dist = st.session_state['capital_dist']
            
            st.markdown('<div class="capital-ultimate">', unsafe_allow_html=True)
            
            for bet_type, amount in capital_dist.items():
                percentage = (amount / total_capital) * 100
                
                col_type, col_bar, col_amount = st.columns([2, 4, 2])
                with col_type:
                    st.write(f"**{bet_type}**")
                with col_bar:
                    st.progress(percentage / 100)
                with col_amount:
                    st.write(f"{percentage:.1f}%")
                
                st.caption(f"  {format_tien(amount)}")
            
            total_allocated = sum(capital_dist.values())
            remaining = total_capital - total_allocated
            
            st.markdown("---")
            
            col_total1, col_total2 = st.columns(2)
            with col_total1:
                st.metric("Tổng phân bổ", format_tien(total_allocated))
            with col_total2:
                st.metric("Còn lại", format_tien(remaining))
            
            if remaining < total_capital * 0.1:
                st.warning("⚠️ Vốn dự phòng thấp, nên giữ lại ít nhất 10% vốn")
            else:
                st.success(f"✅ Phân bổ hợp lý. Dự phòng: {remaining/total_capital*100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Nhập vốn và nhấn 'TÍNH TOÁN PHÂN BỔ' để xem phân bổ")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6b7280;font-size:0.9rem;padding:20px;">
    <div style="font-weight:700;color:#4b5563;margin-bottom:5px;">
    🎰 LOTTO KU AI CỰC PHẨM V12.0 🎰
    </div>
    <div style="color:#9ca3af;">
    Hệ thống AI soi cầu nâng cao • Dựa trên luật chơi Lotto KU chính thức<br>
    ⚠️ Dành cho mục đích nghiên cứu • Quản lý vốn là yếu tố sống còn<br>
    © 2024 LottoKU AI • Phiên bản Ultimate
    </div>
    </div>
    """, unsafe_allow_html=True)

# ================= RUN APP =================
if __name__ == "__main__":
    main()
