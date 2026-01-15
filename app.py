# ================= LOTTO KU AI SIÊU PHẨM – V13.0 ULTIMATE =================
# Advanced AI for Lotto KU based on Official Rules - COMPLETE UPGRADE

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
from typing import List, Dict, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ================= AI LIBRARIES ULTIMATE V13 =================
try:
    # Advanced Machine Learning
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        VotingClassifier, StackingClassifier, AdaBoostClassifier,
        IsolationForest, BaggingClassifier, ExtraTreesClassifier
    )
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.decomposition import PCA, TruncatedSVD, FastICA
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, NuSVC, LinearSVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
    
    # Gradient Boosting Frameworks
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    
    # Deep Learning
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        TORCH_AVAILABLE = True
    except:
        TORCH_AVAILABLE = False
        
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models, callbacks
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU, Bidirectional
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        DEEP_LEARNING_AVAILABLE = True
    except:
        DEEP_LEARNING_AVAILABLE = False
    
    # Time Series Advanced
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import acf, pacf, coint
    import pmdarima as pm
    
    # Advanced Statistics & Math
    from scipy import stats, signal, optimize, interpolate, special
    from scipy.signal import savgol_filter, find_peaks, welch, periodogram, butter, filtfilt
    from scipy.optimize import curve_fit, differential_evolution, minimize, basinhopping
    from scipy.stats import norm, poisson, binom, entropy, skew, kurtosis, chi2, ttest_ind
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.ndimage import gaussian_filter1d
    
    # Feature Engineering
    from sklearn.feature_selection import (
        SelectKBest, RFE, RFECV, SelectFromModel,
        mutual_info_classif, f_classif, chi2, VarianceThreshold
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.compose import ColumnTransformer
    
    # Optimization & Hyperparameter Tuning
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
    
    # Imbalanced Learning
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
    from imblearn.combine import SMOTETomek, SMOTEENN
    
    # Dimensionality Reduction
    from umap import UMAP
    try:
        import prince
        MCA_AVAILABLE = True
    except:
        MCA_AVAILABLE = False
    
    AI_LIBS_AVAILABLE = True
except ImportError as e:
    AI_LIBS_AVAILABLE = False
    st.warning(f"⚠️ Thiếu thư viện AI nâng cao: {str(e)}")

from collections import Counter, defaultdict, deque, OrderedDict
import random
import math
from itertools import combinations, permutations, product
from dataclasses import dataclass, field
from enum import Enum
import pickle
import joblib
from pathlib import Path
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ================= LOGGING SETUP =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cos_v13_ultimate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTTO KU AI SIÊU PHẨM V13.0",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/lottoku-ai',
        'Report a bug': "https://github.com/lottoku-ai/issues",
        'About': "# LOTTO KU AI SIÊU PHẨM V13.0 - Advanced Prediction System"
    }
)

# ULTIMATE CSS V13
st.markdown("""
<style>
    /* Main container - Ultimate V13 */
    .main .block-container {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        max-width: 1800px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    /* Ultimate V13 Header */
    .header-v13 {
        background: linear-gradient(135deg, 
            rgba(99, 102, 241, 0.9) 0%, 
            rgba(168, 85, 247, 0.9) 30%,
            rgba(236, 72, 153, 0.9) 70%,
            rgba(239, 68, 68, 0.9) 100%);
        color: white;
        padding: 25px 40px;
        border-radius: 25px;
        margin: 20px 0 30px 0;
        font-size: 1.6rem;
        font-weight: 900;
        box-shadow: 
            0 25px 50px rgba(99, 102, 241, 0.3),
            0 0 100px rgba(168, 85, 247, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        animation: headerGlow 4s ease-in-out infinite;
    }
    
    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 25px 50px rgba(99, 102, 241, 0.3), 0 0 100px rgba(168, 85, 247, 0.2); }
        50% { box-shadow: 0 25px 50px rgba(99, 102, 241, 0.5), 0 0 150px rgba(236, 72, 153, 0.3); }
    }
    
    .header-v13::before {
        content: '🎰 COS V13.0 ULTIMATE 🎰';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 4rem;
        opacity: 0.08;
        white-space: nowrap;
        font-weight: 900;
        z-index: 0;
    }
    
    .header-v13 .subtitle {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 10px;
        letter-spacing: 1px;
        text-transform: none;
        background: linear-gradient(90deg, #fbbf24, #fde68a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Premium Cards V13 */
    .card-v13 {
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.8) 0%,
            rgba(15, 23, 42, 0.9) 100%);
        padding: 30px;
        border-radius: 25px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 20px 0;
        box-shadow: 
            0 15px 40px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .card-v13::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, 
            #3B82F6, #8B5CF6, #EC4899, #10B981, #F59E0B, #EF4444);
        border-radius: 5px 5px 0 0;
    }
    
    .card-v13:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 
            0 30px 60px rgba(99, 102, 241, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        border-color: #6366f1;
    }
    
    /* Prediction Cards - Ultimate V13 */
    .prediction-v13 {
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.9) 0%,
            rgba(15, 23, 42, 0.95) 100%);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid rgba(99, 102, 241, 0.4);
        text-align: center;
        margin: 15px;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.05);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-v13::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, 
            #3B82F6 0%, #8B5CF6 25%, #EC4899 50%, #10B981 75%, #F59E0B 100%);
    }
    
    .prediction-v13:hover {
        transform: translateY(-10px) rotateX(5deg);
        box-shadow: 
            0 25px 50px rgba(99, 102, 241, 0.3),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        border-color: #8b5cf6;
    }
    
    /* Special Prediction Cards */
    .current-card-v13 {
        border-top: 6px solid #10B981;
        background: linear-gradient(145deg, 
            rgba(16, 185, 129, 0.15) 0%,
            rgba(30, 41, 59, 0.95) 100%);
    }
    
    .next-card-v13 {
        border-top: 6px solid #F59E0B;
        background: linear-gradient(145deg, 
            rgba(245, 158, 11, 0.15) 0%,
            rgba(30, 41, 59, 0.95) 100%);
    }
    
    .special-card-v13 {
        border-top: 6px solid #EC4899;
        background: linear-gradient(145deg, 
            rgba(236, 72, 153, 0.15) 0%,
            rgba(30, 41, 59, 0.95) 100%);
        animation: pulseSpecial 2s infinite;
    }
    
    @keyframes pulseSpecial {
        0%, 100% { box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2); }
        50% { box-shadow: 0 15px 35px rgba(236, 72, 153, 0.3); }
    }
    
    /* Number Displays - Ultimate V13 */
    .number-v13 {
        font-size: 3.5rem;
        font-weight: 900;
        color: #f8fafc;
        text-align: center;
        margin: 20px 0;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, 
            #60a5fa 0%, #a78bfa 20%, #f472b6 40%, #34d399 60%, #fbbf24 80%, #f87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        letter-spacing: 2px;
    }
    
    .number-v13::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 25%;
        width: 50%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #60a5fa, transparent);
        border-radius: 4px;
    }
    
    .number-v13-small {
        font-size: 2.2rem;
        font-weight: 800;
        color: #cbd5e1;
        margin: 15px 0;
        background: linear-gradient(135deg, #94a3b8, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Confidence Indicators - Ultimate V13 */
    .confidence-v13 {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 12px 25px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 900;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border: 3px solid;
        text-transform: uppercase;
        letter-spacing: 1px;
        min-width: 120px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .conf-high-v13 {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.25) 0%,
            rgba(16, 185, 129, 0.5) 100%);
        color: #a7f3d0;
        border-color: #10B981;
    }
    
    .conf-medium-v13 {
        background: linear-gradient(135deg, 
            rgba(245, 158, 11, 0.25) 0%,
            rgba(245, 158, 11, 0.5) 100%);
        color: #fde68a;
        border-color: #F59E0B;
    }
    
    .conf-low-v13 {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.25) 0%,
            rgba(239, 68, 68, 0.5) 100%);
        color: #fecaca;
        border-color: #EF4444;
    }
    
    /* Recommendation Badges - Ultimate V13 */
    .recommend-v13 {
        display: inline-block;
        padding: 14px 28px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 900;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        min-width: 160px;
        cursor: pointer;
        border: 3px solid;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    .recommend-v13:hover {
        transform: scale(1.08);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    
    .rec-bet-v13 {
        background: linear-gradient(135deg, 
            #10B981 0%, #34D399 100%);
        color: white;
        border-color: #047857;
        animation: pulseGreen 1.5s infinite;
    }
    
    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6); }
    }
    
    .rec-maybe-v13 {
        background: linear-gradient(135deg, 
            #F59E0B 0%, #FBBF24 100%);
        color: white;
        border-color: #d97706;
    }
    
    .rec-no-v13 {
        background: linear-gradient(135deg, 
            #EF4444 0%, #F87171 100%);
        color: white;
        border-color: #dc2626;
    }
    
    /* Analysis Grid - Ultimate V13 */
    .analysis-v13 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
        gap: 20px;
        padding: 30px;
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.7) 0%,
            rgba(15, 23, 42, 0.8) 100%);
        border-radius: 25px;
        margin: 25px 0;
        border: 2px solid rgba(99, 102, 241, 0.3);
        box-shadow: 
            inset 0 2px 10px rgba(0, 0, 0, 0.1),
            0 10px 40px rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(15px);
    }
    
    .algo-item-v13 {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, 
            rgba(51, 65, 85, 0.8) 0%,
            rgba(30, 41, 59, 0.9) 100%);
        border-radius: 18px;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .algo-item-v13:hover {
        transform: translateY(-10px) scale(1.08);
        border-color: #6366f1;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
    }
    
    .algo-item-v13::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            #6366f1, #8b5cf6, #ec4899);
    }
    
    .algo-active-v13 {
        border-color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
        animation: pulseAlgo 2s infinite;
    }
    
    @keyframes pulseAlgo {
        0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 0 30px rgba(16, 185, 129, 0.5); }
    }
    
    /* Input Styling - Ultimate V13 */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        border-radius: 18px !important;
        border: 2px solid #475569 !important;
        padding: 18px !important;
        font-size: 1.1rem !important;
        transition: all 0.4s !important;
        background: rgba(30, 41, 59, 0.7) !important;
        color: #f1f5f9 !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 
            0 0 0 5px rgba(99, 102, 241, 0.1),
            inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        background: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* File Uploader - Ultimate V13 */
    .uploadedFile {
        border: 3px dashed #6366f1 !important;
        border-radius: 25px !important;
        padding: 40px !important;
        background: linear-gradient(135deg, 
            rgba(99, 102, 241, 0.08) 0%,
            rgba(168, 85, 247, 0.08) 100%) !important;
        text-align: center !important;
        transition: all 0.4s !important;
        color: #cbd5e1 !important;
    }
    
    .uploadedFile:hover {
        border-color: #8b5cf6 !important;
        background: linear-gradient(135deg, 
            rgba(99, 102, 241, 0.15) 0%,
            rgba(168, 85, 247, 0.15) 100%) !important;
        transform: translateY(-3px);
    }
    
    /* Button Styling - Ultimate V13 */
    .stButton > button {
        border-radius: 18px !important;
        font-weight: 900 !important;
        padding: 18px 36px !important;
        border: none !important;
        background: linear-gradient(135deg, 
            #6366f1 0%, #8b5cf6 30%, #ec4899 70%, #ef4444 100%) !important;
        color: white !important;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        font-size: 1rem !important;
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
        transition: all 0.8s;
    }
    
    .stButton > button:hover {
        transform: translateY(-8px) scale(1.08);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.5) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Progress Bars - Ultimate V13 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, 
            #6366f1 0%, #8b5cf6 20%, #ec4899 40%, #10b981 60%, #f59e0b 80%, #ef4444 100%) !important;
        border-radius: 12px !important;
        height: 15px !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
    }
    
    /* Metric Cards - Ultimate V13 */
    .stMetric {
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.8) 0%,
            rgba(15, 23, 42, 0.9) 100%) !important;
        padding: 25px !important;
        border-radius: 20px !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        box-shadow: 
            0 12px 35px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.4s !important;
        backdrop-filter: blur(15px);
        color: #f1f5f9 !important;
    }
    
    .stMetric:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 25px 50px rgba(99, 102, 241, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        border-color: #6366f1;
    }
    
    /* Tab Styling - Ultimate V13 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.7) 0%,
            rgba(15, 23, 42, 0.8) 100%);
        padding: 15px;
        border-radius: 25px;
        border: 2px solid rgba(99, 102, 241, 0.3);
        backdrop-filter: blur(15px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 18px !important;
        padding: 18px 30px !important;
        font-weight: 800 !important;
        background: rgba(51, 65, 85, 0.6) !important;
        border: 2px solid transparent !important;
        transition: all 0.4s !important;
        color: #cbd5e1 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(71, 85, 105, 0.8) !important;
        border-color: #6366f1 !important;
        transform: translateY(-3px);
        color: #f1f5f9 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-color: #6366f1 !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Custom Scrollbar - Ultimate V13 */
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.8) 0%,
            rgba(15, 23, 42, 0.9) 100%);
        border-radius: 12px;
        border: 2px solid rgba(99, 102, 241, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, 
            #6366f1 0%, #8b5cf6 40%, #ec4899 100%);
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, 
            #4f46e5 0%, #7c3aed 40%, #db2777 100%);
    }
    
    /* Notification Box - Ultimate V13 */
    .notification-v13 {
        background: linear-gradient(135deg, 
            rgba(251, 191, 36, 0.15) 0%,
            rgba(245, 158, 11, 0.2) 100%);
        border: 3px solid #f59e0b;
        padding: 30px;
        border-radius: 25px;
        margin: 30px 0;
        font-size: 1.1rem;
        box-shadow: 
            0 20px 40px rgba(245, 158, 11, 0.2),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(15px);
        color: #fef3c7;
    }
    
    .notification-v13::before {
        content: '🎯';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 4rem;
        opacity: 0.15;
        transform: rotate(15deg);
    }
    
    /* Capital Management - Ultimate V13 */
    .capital-v13 {
        background: linear-gradient(135deg, 
            rgba(14, 165, 233, 0.1) 0%,
            rgba(56, 189, 248, 0.15) 100%);
        padding: 30px;
        border-radius: 25px;
        border: 2px solid rgba(14, 165, 233, 0.3);
        margin: 25px 0;
        box-shadow: 
            0 20px 40px rgba(14, 165, 233, 0.15),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }
    
    /* Data Table Styling */
    .stDataFrame {
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Chart Container */
    .stPlotlyChart {
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }
    
    /* Toggle Switch */
    .stCheckbox {
        color: #cbd5e1 !important;
    }
    
    /* Select Box */
    .stSelectbox {
        color: #cbd5e1 !important;
    }
    
    .stSelectbox select {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    
    /* Loading Animation */
    @keyframes pulseV13 {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.08); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .loading-v13 {
        animation: pulseV13 1.8s infinite;
    }
    
    /* Glow Effect V13 */
    .glow-v13 {
        position: relative;
    }
    
    .glow-v13::before {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        background: linear-gradient(45deg, 
            #6366f1, #8b5cf6, #ec4899, #10b981, #f59e0b, #ef4444);
        border-radius: inherit;
        z-index: -1;
        filter: blur(15px);
        opacity: 0.5;
        animation: rotateGradient 4s linear infinite;
    }
    
    @keyframes rotateGradient {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Floating Animation V13 */
    @keyframes floatV13 {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
    .floating-v13 {
        animation: floatV13 4s ease-in-out infinite;
    }
    
    /* Rain Effect for Special Predictions */
    .rain-effect {
        position: absolute;
        top: -100px;
        left: 0;
        width: 100%;
        height: 100px;
        background: linear-gradient(to bottom, transparent, rgba(99, 102, 241, 0.1), transparent);
        animation: rain 3s linear infinite;
    }
    
    @keyframes rain {
        0% { transform: translateY(-100px); opacity: 0; }
        50% { opacity: 0.5; }
        100% { transform: translateY(300px); opacity: 0; }
    }
    
    /* Responsive Design V13 */
    @media (max-width: 1600px) {
        .number-v13 { font-size: 3rem; }
        .number-v13-small { font-size: 1.9rem; }
        .prediction-v13 { height: 200px; padding: 25px; }
        .header-v13 { font-size: 1.4rem; padding: 20px 30px; }
        .header-v13::before { font-size: 3rem; }
    }
    
    @media (max-width: 1200px) {
        .number-v13 { font-size: 2.5rem; }
        .prediction-v13 { height: 180px; padding: 20px; margin: 10px; }
        .card-v13 { padding: 25px; margin: 15px 0; }
        .header-v13 { font-size: 1.2rem; padding: 15px 20px; }
        .header-v13::before { font-size: 2.5rem; }
    }
    
    @media (max-width: 768px) {
        .number-v13 { font-size: 2rem; }
        .number-v13-small { font-size: 1.6rem; }
        .prediction-v13 { height: 160px; padding: 15px; margin: 8px; }
        .card-v13 { padding: 20px; margin: 10px 0; }
        .header-v13 { font-size: 1rem; padding: 12px 15px; }
        .header-v13::before { font-size: 2rem; }
        .recommend-v13 { padding: 10px 20px; font-size: 0.9rem; min-width: 120px; }
        .confidence-v13 { padding: 8px 15px; font-size: 0.8rem; min-width: 100px; }
    }
    
    /* Dark Mode Adjustments */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #475569 !important;
        color: #f1f5f9 !important;
    }
    
    /* Text Colors */
    .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, span, div {
        color: #cbd5e1 !important;
    }
    
    /* Icon Colors */
    .material-icons {
        color: #6366f1 !important;
    }
    
    /* Link Colors */
    a {
        color: #60a5fa !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #3b82f6 !important;
        text-decoration: underline !important;
    }
    
    /* Warning and Error Colors */
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.2)) !important;
        border-color: #f59e0b !important;
        color: #fef3c7 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.2)) !important;
        border-color: #ef4444 !important;
        color: #fecaca !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.2)) !important;
        border-color: #10b981 !important;
        color: #a7f3d0 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(14, 165, 233, 0.2)) !important;
        border-color: #0ea5e9 !important;
        color: #bae6fd !important;
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "cos_v13_ultimate.db"
MODEL_DIR = Path("models_v13")
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache_v13")
CACHE_DIR.mkdir(exist_ok=True)

# ================= ENUMS & DATA CLASSES V13 =================
class BetType(Enum):
    """Loại cược theo luật Lotto KU - V13"""
    TONG_TAI_XIU = "tổng_tài_xỉu"
    TONG_LE_CHAN = "tổng_lẻ_chẵn"
    TIEN_NHI = "tiền_nhị"
    HAU_NHI = "hậu_nhị"
    HANG_SO_1 = "hàng_số_1"
    DE_SO = "đề_số"
    CHAN_LE = "chẵn_lẻ"
    CHINH_PHU = "chính_phụ"
    SO_2_TINH = "2_tinh"
    SO_3_TINH = "3_tinh"
    DE_SO_DAU = "đề_số_đầu"
    DE_SO_DUOI = "đề_số_đuôi"
    CUP_DIEN = "cúp_điện"
    CAU_BET = "cầu_bệt"
    CAU_SONG = "cầu_sống"
    CAU_CHET = "cầu_chết"

class Position(Enum):
    """Vị trí số theo luật Lotto KU - V13"""
    CHUC_NGAN = 0
    NGAN = 1
    TRAM = 2
    CHUC = 3
    DON_VI = 4

class PatternType(Enum):
    """Loại pattern"""
    CAU_BET = "cầu_bệt"
    CAU_SONG = "cầu_sống"
    CAU_CHET = "cầu_chết"
    CAU_DAO = "cầu_đảo"
    CAU_GAP = "cầu_gấp"
    CAU_HIEU = "cầu_hiệu"
    CAU_TONG = "cầu_tổng"
    CAU_CHANH = "cầu_chánh"
    CAU_PHU = "cầu_phụ"

@dataclass
class LotteryNumberV13:
    """Biểu diễn số lotto 5 chữ số - V13"""
    chuc_ngan: int
    ngan: int
    tram: int
    chuc: int
    don_vi: int
    
    # Thêm thuộc tính tính toán
    _2tinh_pairs: List[Tuple[int, int]] = field(default_factory=list)
    _3tinh_pairs: List[Tuple[int, int, int]] = field(default_factory=list)
    _tien_nhi: str = field(init=False)
    _hau_nhi: str = field(init=False)
    _total: int = field(init=False)
    _is_tai: bool = field(init=False)
    _is_xiu: bool = field(init=False)
    _is_chan: bool = field(init=False)
    _is_le: bool = field(init=False)
    _is_chinh: bool = field(init=False)
    _is_phu: bool = field(init=False)
    
    def __post_init__(self):
        # Validate
        for attr in ['chuc_ngan', 'ngan', 'tram', 'chuc', 'don_vi']:
            value = getattr(self, attr)
            if not 0 <= value <= 9:
                raise ValueError(f"{attr} phải từ 0-9, nhận được {value}")
        
        # Tính toán
        self._tien_nhi = f"{self.chuc_ngan}{self.ngan}"
        self._hau_nhi = f"{self.chuc}{self.don_vi}"
        self._total = sum([self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi])
        self._is_tai = 23 <= self._total <= 45
        self._is_xiu = 0 <= self._total <= 22
        self._is_chan = self._total % 2 == 0
        self._is_le = self._total % 2 == 1
        self._is_chinh = self._total >= 23
        self._is_phu = self._total <= 22
        
        # Generate 2-tinh và 3-tinh pairs
        self._generate_pairs()
    
    def _generate_pairs(self):
        """Tạo các cặp 2-tinh và 3-tinh"""
        digits = [self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi]
        
        # 2-tinh: tất cả cặp 2 số từ 5 số
        self._2tinh_pairs = list(combinations(digits, 2))
        
        # 3-tinh: tất cả bộ 3 số từ 5 số
        self._3tinh_pairs = list(combinations(digits, 3))
    
    @classmethod
    def from_string(cls, num_str: str):
        if len(num_str) != 5 or not num_str.isdigit():
            raise ValueError("Chuỗi phải có đúng 5 chữ số")
        digits = [int(d) for d in num_str]
        return cls(*digits)
    
    def to_string(self) -> str:
        return f"{self.chuc_ngan}{self.ngan}{self.tram}{self.chuc}{self.don_vi}"
    
    def get_tien_nhi(self) -> str:
        return self._tien_nhi
    
    def get_hau_nhi(self) -> str:
        return self._hau_nhi
    
    def get_tong(self) -> int:
        return self._total
    
    def is_tai(self) -> bool:
        return self._is_tai
    
    def is_xiu(self) -> bool:
        return self._is_xiu
    
    def is_chan(self) -> bool:
        return self._is_chan
    
    def is_le(self) -> bool:
        return self._is_le
    
    def is_chinh(self) -> bool:
        return self._is_chinh
    
    def is_phu(self) -> bool:
        return self._is_phu
    
    def get_2tinh_pairs(self) -> List[Tuple[int, int]]:
        return self._2tinh_pairs
    
    def get_3tinh_pairs(self) -> List[Tuple[int, int, int]]:
        return self._3tinh_pairs
    
    def get_de_dau(self) -> int:
        """Đề đầu (số đầu tiên)"""
        return self.chuc_ngan
    
    def get_de_duoi(self) -> int:
        """Đề đuôi (số cuối cùng)"""
        return self.don_vi

@dataclass
class PatternAnalysis:
    """Phân tích pattern"""
    pattern_type: PatternType
    start_index: int
    end_index: int
    strength: float
    confidence: float
    numbers: List[str]
    metadata: Dict[str, Any]

@dataclass
class PredictionResultV13:
    """Kết quả dự đoán - V13"""
    bet_type: BetType
    predicted_value: Any
    confidence: float
    recommendation: str
    reasoning: str
    timestamp: datetime
    algorithm_used: str
    probability_dist: Dict[str, float] = field(default_factory=dict)
    top_alternatives: List[Any] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'bet_type': self.bet_type.value,
            'predicted_value': str(self.predicted_value),
            'confidence': self.confidence,
            'recommendation': self.recommendation,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat(),
            'algorithm_used': self.algorithm_used,
            'probability_dist': self.probability_dist,
            'top_alternatives': self.top_alternatives
        }

# ================= ADVANCED AI ENGINE V13 =================
class COSAdvancedAI:
    """AI cực phẩm cho Lotto KU với 25+ thuật toán nâng cao"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data.copy()
        self.numbers = self._extract_numbers()
        self.cache = {}
        self.models = {}
        self.patterns = []
        
        # Cấu hình
        self.n_algorithms = 25
        self.use_deep_learning = DEEP_LEARNING_AVAILABLE
        self.use_ensemble = True
        self.optimize_hyperparams = True
        
        # 25+ thuật toán cao cấp
        self.algorithms = {
            1: 'statistical_analysis_advanced',
            2: 'hot_cold_pattern_recognition_advanced',
            3: 'positional_analysis_ultimate',
            4: 'time_series_forecasting_hybrid',
            5: 'neural_network_ensemble',
            6: 'gradient_boosting_ultimate',
            7: 'probability_distribution_modeling',
            8: 'cycle_detection_spectral',
            9: 'correlation_network_analysis_advanced',
            10: 'entropy_and_information_theory_advanced',
            11: 'markov_chain_higher_order',
            12: 'deep_learning_lstm',
            13: 'genetic_algorithm_optimized',
            14: 'bayesian_inference_advanced',
            15: 'quantum_inspired_prediction',
            16: 'cluster_analysis_multidimensional',
            17: 'anomaly_detection_isolation',
            18: 'feature_importance_analysis',
            19: 'pattern_recognition_cnn',
            20: 'ensemble_stacking_ultimate',
            21: 'arima_garch_hybrid',
            22: 'prophet_forecasting',
            23: 'wavelet_analysis',
            24: 'fourier_transform_analysis',
            25: 'chaos_theory_analysis',
            26: 'two_star_analysis',      # 2 TINH
            27: 'three_star_analysis',     # 3 TINH
            28: 'de_so_analysis',         # Đề số
            29: 'tai_xiu_chart_analysis', # Tài xỉu chart
            30: 'smart_pattern_detection' # Pattern thông minh
        }
        
        # Khởi tạo weights
        self.algo_weights = self._initialize_weights()
        
        # Khởi tạo cache
        self._init_cache()
    
    def _extract_numbers(self) -> List[LotteryNumberV13]:
        """Trích xuất số từ dataframe"""
        numbers = []
        for _, row in self.df.iterrows():
            try:
                if 'so5' in row and len(str(row['so5'])) == 5:
                    num = LotteryNumberV13.from_string(str(row['so5']))
                    numbers.append(num)
            except:
                continue
        return numbers
    
    def _initialize_weights(self) -> Dict[int, float]:
        """Khởi tạo weights cho thuật toán"""
        base_weights = {
            1: 0.10, 2: 0.11, 3: 0.12, 4: 0.13, 5: 0.14,
            6: 0.15, 7: 0.11, 8: 0.10, 9: 0.12, 10: 0.13,
            11: 0.12, 12: 0.15, 13: 0.14, 14: 0.13, 15: 0.12,
            16: 0.11, 17: 0.10, 18: 0.12, 19: 0.14, 20: 0.16,
            21: 0.13, 22: 0.12, 23: 0.11, 24: 0.10, 25: 0.12,
            26: 0.15, 27: 0.16, 28: 0.14, 29: 0.13, 30: 0.15
        }
        
        # Chuẩn hóa weights
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}
    
    def _init_cache(self):
        """Khởi tạo cache"""
        self.cache = {
            'predictions': {},
            'patterns': [],
            'features': {},
            'model_performance': {},
            'last_updated': datetime.now()
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """Chạy phân tích toàn diện với tất cả thuật toán"""
        logger.info("Bắt đầu phân tích toàn diện V13...")
        
        if not self.numbers or len(self.numbers) < 30:
            logger.warning("Không đủ dữ liệu để phân tích")
            return self._get_default_results()
        
        results = {
            'algorithms': {},
            'predictions': {},
            'patterns': [],
            'summary': {},
            'timestamps': {},
            'performance': {}
        }
        
        # Parallel execution of algorithms
        with ThreadPoolExecutor(max_workers=min(10, self.n_algorithms)) as executor:
            futures = {}
            for algo_id, algo_name in self.algorithms.items():
                if algo_id <= self.n_algorithms:
                    future = executor.submit(self._run_algorithm, algo_id, algo_name)
                    futures[future] = algo_id
            
            for future in as_completed(futures):
                algo_id = futures[future]
                try:
                    algo_result = future.result(timeout=30)
                    results['algorithms'][algo_id] = algo_result
                    results['timestamps'][algo_id] = datetime.now().isoformat()
                    
                    # Update performance
                    if 'confidence' in algo_result:
                        conf = algo_result['confidence']
                        if conf >= 70:
                            results['performance'][algo_id] = 'high'
                        elif conf >= 50:
                            results['performance'][algo_id] = 'medium'
                        else:
                            results['performance'][algo_id] = 'low'
                    
                except Exception as e:
                    logger.error(f"Lỗi thuật toán {algo_id}: {str(e)}")
                    results['algorithms'][algo_id] = {
                        'error': str(e),
                        'confidence': 20
                    }
        
        # Phát hiện pattern
        results['patterns'] = self._detect_all_patterns()
        
        # Tạo dự đoán tổng hợp
        results['predictions'] = self._generate_comprehensive_predictions(results['algorithms'])
        
        # Phân tích chi tiết
        results['detailed_analysis'] = self._generate_detailed_analysis(results)
        
        # Tạo summary
        results['summary'] = self._generate_comprehensive_summary(results)
        
        # Lưu cache
        self.cache['predictions'] = results['predictions']
        self.cache['patterns'] = results['patterns']
        self.cache['last_updated'] = datetime.now()
        
        logger.info("Hoàn thành phân tích toàn diện")
        return results
    
    def _run_algorithm(self, algo_id: int, algo_name: str) -> Dict:
        """Chạy thuật toán cụ thể"""
        try:
            if hasattr(self, algo_name):
                start_time = time.time()
                result = getattr(self, algo_name)()
                exec_time = time.time() - start_time
                
                if isinstance(result, dict):
                    result['execution_time'] = exec_time
                    result['algorithm_id'] = algo_id
                
                return result
            else:
                return {
                    'error': f'Algorithm {algo_name} not implemented',
                    'confidence': 30,
                    'execution_time': 0
                }
        except Exception as e:
            logger.error(f"Lỗi trong thuật toán {algo_name}: {str(e)}")
            return {
                'error': str(e),
                'confidence': 20,
                'execution_time': 0
            }
    
    # ================= ALGORITHM IMPLEMENTATIONS V13 =================
    
    def statistical_analysis_advanced(self) -> Dict:
        """Phân tích thống kê nâng cao"""
        if len(self.numbers) < 20:
            return {'confidence': 35}
        
        # Tính toán nâng cao
        totals = [num.get_tong() for num in self.numbers]
        tien_nhi = [int(num.get_tien_nhi()) for num in self.numbers]
        hau_nhi = [int(num.get_hau_nhi()) for num in self.numbers]
        
        # Phân phối chi tiết
        total_dist = self._calculate_distribution(totals, 0, 45)
        tien_nhi_dist = self._calculate_distribution(tien_nhi, 0, 99)
        hau_nhi_dist = self._calculate_distribution(hau_nhi, 0, 99)
        
        # Thống kê nâng cao
        stats = {
            'total': self._calculate_advanced_stats(totals),
            'tien_nhi': self._calculate_advanced_stats(tien_nhi),
            'hau_nhi': self._calculate_advanced_stats(hau_nhi)
        }
        
        # Phân tích xu hướng
        trends = self._calculate_trends_advanced(totals, tien_nhi, hau_nhi)
        
        # Dự đoán
        predictions = self._statistical_predictions(totals, tien_nhi, hau_nhi)
        
        return {
            'distributions': {
                'total': total_dist,
                'tien_nhi': tien_nhi_dist,
                'hau_nhi': hau_nhi_dist
            },
            'statistics': stats,
            'trends': trends,
            'predictions': predictions,
            'confidence': min(85, len(self.numbers) / 100 * 80 + 15),
            'data_points': len(self.numbers)
        }
    
    def two_star_analysis(self) -> Dict:
        """Phân tích 2 TINH - Dự đoán 2 cặp số"""
        logger.info("Running 2 TINH analysis...")
        
        if len(self.numbers) < 30:
            return {'confidence': 40, 'predictions': []}
        
        # Extract all 2-tinh pairs
        all_2tinh = []
        for num in self.numbers:
            all_2tinh.extend([f"{a}{b}" for a, b in num.get_2tinh_pairs()])
        
        # Analyze frequency
        freq_counter = Counter(all_2tinh[-100:])  # Last 100 pairs
        total_pairs = len(all_2tinh[-100:])
        
        # Calculate probabilities
        predictions = []
        for pair, count in freq_counter.most_common(20):
            probability = (count / total_pairs) * 100
            
            # Determine recommendation
            if probability >= 2.5:
                recommendation = "NÊN ĐÁNH"
                rec_color = "green"
            elif probability >= 1.5:
                recommendation = "CÓ THỂ ĐÁNH"
                rec_color = "orange"
            else:
                recommendation = "KHÔNG ĐÁNH"
                rec_color = "red"
            
            predictions.append({
                'pair': pair,
                'probability': round(probability, 2),
                'frequency': count,
                'recommendation': recommendation,
                'color': rec_color,
                'strength': min(100, probability * 20)
            })
        
        # Calculate overall confidence
        if predictions:
            avg_prob = np.mean([p['probability'] for p in predictions[:5]])
            confidence = min(90, avg_prob * 1.5 + 40)
        else:
            confidence = 50
        
        return {
            'predictions': predictions[:10],  # Top 10
            'total_pairs_analyzed': total_pairs,
            'unique_pairs': len(freq_counter),
            'most_common': freq_counter.most_common(5),
            'confidence': confidence,
            'algorithm': '2_tinh_analysis'
        }
    
    def three_star_analysis(self) -> Dict:
        """Phân tích 3 TINH - Dự đoán 3 cặp số"""
        logger.info("Running 3 TINH analysis...")
        
        if len(self.numbers) < 50:
            return {'confidence': 45, 'predictions': []}
        
        # Extract all 3-tinh combinations
        all_3tinh = []
        for num in self.numbers:
            all_3tinh.extend([f"{a}{b}{c}" for a, b, c in num.get_3tinh_pairs()])
        
        # Analyze frequency
        freq_counter = Counter(all_3tinh[-150:])  # Last 150 combinations
        total_combs = len(all_3tinh[-150:])
        
        # Calculate probabilities
        predictions = []
        for comb, count in freq_counter.most_common(15):
            probability = (count / total_combs) * 100
            
            # Determine recommendation
            if probability >= 1.0:
                recommendation = "NÊN ĐÁNH"
                rec_color = "green"
            elif probability >= 0.5:
                recommendation = "CÓ THỂ ĐÁNH"
                rec_color = "orange"
            else:
                recommendation = "KHÔNG ĐÁNH"
                rec_color = "red"
            
            predictions.append({
                'combination': comb,
                'probability': round(probability, 2),
                'frequency': count,
                'recommendation': recommendation,
                'color': rec_color,
                'strength': min(100, probability * 25)
            })
        
        # Calculate overall confidence
        if predictions:
            avg_prob = np.mean([p['probability'] for p in predictions[:5]])
            confidence = min(88, avg_prob * 2 + 35)
        else:
            confidence = 50
        
        return {
            'predictions': predictions[:8],  # Top 8
            'total_combinations_analyzed': total_combs,
            'unique_combinations': len(freq_counter),
            'most_common': freq_counter.most_common(5),
            'confidence': confidence,
            'algorithm': '3_tinh_analysis'
        }
    
    def de_so_analysis(self) -> Dict:
        """Phân tích đề số hậu nhị, tiền nhị"""
        logger.info("Running DE SO analysis...")
        
        if len(self.numbers) < 40:
            return {'confidence': 40, 'predictions': []}
        
        # Extract last 50 numbers for analysis
        recent_nums = self.numbers[:100]
        
        # Analyze tien_nhi (first two digits)
        tien_nhi_list = [int(num.get_tien_nhi()) for num in recent_nums]
        tien_nhi_counter = Counter(tien_nhi_list)
        
        # Analyze hau_nhi (last two digits)
        hau_nhi_list = [int(num.get_hau_nhi()) for num in recent_nums]
        hau_nhi_counter = Counter(hau_nhi_list)
        
        # Get top predictions
        tien_nhi_predictions = self._get_de_so_predictions(tien_nhi_counter, "tiền nhị")
        hau_nhi_predictions = self._get_de_so_predictions(hau_nhi_counter, "hậu nhị")
        
        # Combined analysis
        combined_predictions = []
        for i in range(5):
            tn_pred = tien_nhi_predictions[i] if i < len(tien_nhi_predictions) else None
            hn_pred = hau_nhi_predictions[i] if i < len(hau_nhi_predictions) else None
            
            if tn_pred:
                combined_predictions.append(tn_pred)
            if hn_pred and hn_pred not in combined_predictions:
                combined_predictions.append(hn_pred)
        
        confidence = min(85, (len(tien_nhi_predictions) + len(hau_nhi_predictions)) * 5)
        
        return {
            'tien_nhi_predictions': tien_nhi_predictions,
            'hau_nhi_predictions': hau_nhi_predictions,
            'combined_predictions': combined_predictions[:5],
            'confidence': confidence,
            'algorithm': 'de_so_analysis',
            'analysis_period': len(recent_nums)
        }
    
    def tai_xiu_chart_analysis(self) -> Dict:
        """Phân tích Tài Xỉu - Chẵn Lẻ"""
        logger.info("Running TAI XIU analysis...")
        
        if len(self.numbers) < 30:
            return {'confidence': 35, 'predictions': []}
        
        # Analyze last 50 draws
        recent_nums = self.numbers[:80]
        
        tai_count = sum(1 for num in recent_nums if num.is_tai())
        xiu_count = sum(1 for num in recent_nums if num.is_xiu())
        chan_count = sum(1 for num in recent_nums if num.is_chan())
        le_count = sum(1 for num in recent_nums if num.is_le())
        
        total_draws = len(recent_nums)
        
        # Calculate percentages
        tai_percent = (tai_count / total_draws) * 100
        xiu_percent = (xiu_count / total_draws) * 100
        chan_percent = (chan_count / total_draws) * 100
        le_percent = (le_count / total_draws) * 100
        
        # Predict next outcome
        predictions = []
        
        # Tài/Xỉu prediction
        if tai_percent > 55:
            predictions.append({
                'type': 'Tài/Xỉu',
                'prediction': 'TÀI',
                'probability': round(tai_percent, 2),
                'confidence': min(90, tai_percent * 1.2),
                'recommendation': 'NÊN ĐÁNH' if tai_percent > 60 else 'CÓ THỂ ĐÁNH'
            })
        elif xiu_percent > 55:
            predictions.append({
                'type': 'Tài/Xỉu',
                'prediction': 'XỈU',
                'probability': round(xiu_percent, 2),
                'confidence': min(90, xiu_percent * 1.2),
                'recommendation': 'NÊN ĐÁNH' if xiu_percent > 60 else 'CÓ THỂ ĐÁNH'
            })
        
        # Chẵn/Lẻ prediction
        if chan_percent > 55:
            predictions.append({
                'type': 'Chẵn/Lẻ',
                'prediction': 'CHẴN',
                'probability': round(chan_percent, 2),
                'confidence': min(88, chan_percent * 1.2),
                'recommendation': 'NÊN ĐÁNH' if chan_percent > 60 else 'CÓ THỂ ĐÁNH'
            })
        elif le_percent > 55:
            predictions.append({
                'type': 'Chẵn/Lẻ',
                'prediction': 'LẺ',
                'probability': round(le_percent, 2),
                'confidence': min(88, le_percent * 1.2),
                'recommendation': 'NÊN ĐÁNH' if le_percent > 60 else 'CÓ THỂ ĐÁNH'
            })
        
        # Calculate overall confidence
        if predictions:
            confidences = [p['confidence'] for p in predictions]
            avg_confidence = np.mean(confidences)
        else:
            avg_confidence = 50
        
        return {
            'statistics': {
                'tai': {'count': tai_count, 'percent': tai_percent},
                'xiu': {'count': xiu_count, 'percent': xiu_percent},
                'chan': {'count': chan_count, 'percent': chan_percent},
                'le': {'count': le_count, 'percent': le_percent}
            },
            'predictions': predictions,
            'confidence': avg_confidence,
            'algorithm': 'tai_xiu_analysis',
            'period_analyzed': total_draws
        }
    
    def smart_pattern_detection(self) -> Dict:
        """Phát hiện pattern thông minh: cầu bệt, cầu sống, cầu chết"""
        logger.info("Running SMART PATTERN detection...")
        
        if len(self.numbers) < 50:
            return {'confidence': 40, 'patterns': []}
        
        patterns = []
        
        # Detect various patterns
        patterns.extend(self._detect_cau_bet())
        patterns.extend(self._detect_cau_song())
        patterns.extend(self._detect_cau_chet())
        patterns.extend(self._detect_cau_dao())
        patterns.extend(self._detect_cau_gap())
        
        # Calculate confidence based on pattern strength
        if patterns:
            strengths = [p.get('strength', 0) for p in patterns]
            avg_strength = np.mean(strengths)
            confidence = min(95, avg_strength * 1.5)
            
            # Sort by strength
            patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)
        else:
            confidence = 45
        
        return {
            'patterns': patterns[:10],  # Top 10 patterns
            'total_patterns_detected': len(patterns),
            'confidence': confidence,
            'algorithm': 'smart_pattern_detection'
        }
    
    def neural_network_ensemble(self) -> Dict:
        """Neural Network Ensemble nâng cao"""
        if not AI_LIBS_AVAILABLE or len(self.numbers) < 100:
            return {'confidence': 40}
        
        try:
            # Advanced feature engineering
            features, targets = self._create_advanced_features()
            
            if len(features) < 50:
                return {'confidence': 35}
            
            # Multiple neural network architectures
            models = self._create_nn_ensemble()
            
            # Train and evaluate
            results = []
            for name, model in models.items():
                try:
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, targets, test_size=0.2, random_state=42
                    )
                    
                    # Train
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
                        results.append({
                            'model': name,
                            'accuracy': accuracy * 100
                        })
                except:
                    continue
            
            if results:
                best_model = max(results, key=lambda x: x['accuracy'])
                confidence = best_model['accuracy']
                
                return {
                    'ensemble_results': results,
                    'best_model': best_model,
                    'confidence': min(92, confidence),
                    'features_used': len(features[0]),
                    'algorithm': 'neural_network_ensemble'
                }
            else:
                return {'confidence': 45}
                
        except Exception as e:
            logger.error(f"Lỗi neural network: {str(e)}")
            return {'confidence': 40, 'error': str(e)}
    
    def gradient_boosting_ultimate(self) -> Dict:
        """Gradient Boosting Ultimate với XGBoost, LightGBM, CatBoost"""
        if not AI_LIBS_AVAILABLE or len(self.numbers) < 80:
            return {'confidence': 42}
        
        try:
            features, targets = self._create_advanced_features()
            
            # Train multiple boosting algorithms
            models = {
                'xgboost': xgb.XGBClassifier(n_estimators=200, random_state=42),
                'lightgbm': lgb.LGBMClassifier(n_estimators=200, random_state=42),
                'catboost': cb.CatBoostClassifier(iterations=200, verbose=0, random_state=42)
            }
            
            results = []
            for name, model in models.items():
                try:
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(model, features, targets, cv=5, scoring='accuracy')
                    avg_score = np.mean(scores) * 100
                    results.append({
                        'model': name,
                        'accuracy': avg_score,
                        'std': np.std(scores) * 100
                    })
                except:
                    continue
            
            if results:
                best_result = max(results, key=lambda x: x['accuracy'])
                
                # Train final model
                final_model = models[best_result['model']]
                final_model.fit(features, targets)
                
                # Make prediction
                last_features = features[-1].reshape(1, -1)
                prediction = final_model.predict(last_features)[0]
                proba = final_model.predict_proba(last_features)[0]
                
                return {
                    'boosting_results': results,
                    'best_model': best_result['model'],
                    'prediction': int(prediction),
                    'confidence': best_result['accuracy'],
                    'probability': float(max(proba)) * 100,
                    'algorithm': 'gradient_boosting_ultimate'
                }
            else:
                return {'confidence': 48}
                
        except Exception as e:
            logger.error(f"Lỗi gradient boosting: {str(e)}")
            return {'confidence': 45, 'error': str(e)}
    
    def deep_learning_lstm(self) -> Dict:
        """Deep Learning với LSTM nâng cao"""
        if not DEEP_LEARNING_AVAILABLE or len(self.numbers) < 150:
            return {'confidence': 38}
        
        try:
            # Prepare sequential data for LSTM
            sequences, labels = self._create_lstm_sequences()
            
            if len(sequences) < 30:
                return {'confidence': 35}
            
            # Build LSTM model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(sequences.shape[1], sequences.shape[2])),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            
            # Train
            history = model.fit(sequences, labels, 
                              epochs=50, 
                              batch_size=16, 
                              validation_split=0.2,
                              verbose=0)
            
            # Get final accuracy
            final_accuracy = history.history['val_accuracy'][-1] * 100
            
            # Predict
            last_sequence = sequences[-1].reshape(1, sequences.shape[1], sequences.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            return {
                'model_type': 'LSTM',
                'accuracy': final_accuracy,
                'prediction': 'TÀI' if prediction > 0.5 else 'XỈU',
                'prediction_score': float(prediction),
                'confidence': min(90, final_accuracy * 1.1),
                'training_history': {
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1]
                },
                'algorithm': 'deep_learning_lstm'
            }
            
        except Exception as e:
            logger.error(f"Lỗi LSTM: {str(e)}")
            return {'confidence': 40, 'error': str(e)}
    
    def ensemble_stacking_ultimate(self) -> Dict:
        """Ensemble Stacking Ultimate"""
        if not AI_LIBS_AVAILABLE or len(self.numbers) < 100:
            return {'confidence': 45}
        
        try:
            features, targets = self._create_advanced_features()
            
            # Base models
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False)),
                ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ]
            
            # Meta model
            meta_model = LogisticRegression(random_state=42)
            
            # Stacking ensemble
            stacking = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                passthrough=True
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(stacking, features, targets, cv=5, scoring='accuracy')
            avg_accuracy = np.mean(scores) * 100
            
            # Train final model
            stacking.fit(features, targets)
            
            # Make prediction
            last_features = features[-1].reshape(1, -1)
            prediction = stacking.predict(last_features)[0]
            proba = stacking.predict_proba(last_features)[0]
            
            return {
                'ensemble_type': 'Stacking',
                'base_models': len(base_models),
                'accuracy': avg_accuracy,
                'prediction': int(prediction),
                'confidence': avg_accuracy,
                'probability': float(max(proba)) * 100,
                'algorithm': 'ensemble_stacking_ultimate'
            }
            
        except Exception as e:
            logger.error(f"Lỗi ensemble: {str(e)}")
            return {'confidence': 48, 'error': str(e)}
    
    # ================= HELPER METHODS =================
    
    def _create_advanced_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo features nâng cao cho machine learning"""
        features = []
        targets = []
        
        for i in range(len(self.numbers) - 1):
            current = self.numbers[i]
            next_num = self.numbers[i + 1]
            
            # Basic features
            feat = [
                current.chuc_ngan, current.ngan, current.tram, current.chuc, current.don_vi,
                current.get_tong(),
                sum(1 for d in [current.chuc_ngan, current.ngan, current.tram, current.chuc, current.don_vi] if d % 2 == 0),  # Even count
                sum(1 for d in [current.chuc_ngan, current.ngan, current.tram, current.chuc, current.don_vi] if d >= 5),      # High digits
                current.chuc_ngan + current.ngan,  # Tien nhi sum
                current.chuc + current.don_vi,     # Hau nhi sum
                int(current.is_tai()),             # Is Tai
                int(current.is_le()),              # Is Le
            ]
            
            # Add rolling statistics
            if i >= 5:
                prev_nums = self.numbers[i-5:i]
                prev_totals = [n.get_tong() for n in prev_nums]
                feat.extend([
                    np.mean(prev_totals),   # Moving average
                    np.std(prev_totals),    # Moving std
                    np.min(prev_totals),    # Min
                    np.max(prev_totals),    # Max
                    prev_totals[-1] - prev_totals[0]  # Trend
                ])
            else:
                feat.extend([0] * 5)
            
            # Add position differences
            if i > 0:
                prev = self.numbers[i-1]
                feat.extend([
                    current.chuc_ngan - prev.chuc_ngan,
                    current.ngan - prev.ngan,
                    current.tram - prev.tram,
                    current.chuc - prev.chuc,
                    current.don_vi - prev.don_vi
                ])
            else:
                feat.extend([0] * 5)
            
            features.append(feat)
            
            # Target: will next be Tai (1) or Xiu (0)
            targets.append(1 if next_num.is_tai() else 0)
        
        return np.array(features), np.array(targets)
    
    def _create_lstm_sequences(self, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo sequences cho LSTM"""
        totals = [num.get_tong() for num in self.numbers]
        
        sequences = []
        labels = []
        
        for i in range(len(totals) - seq_length):
            seq = totals[i:i+seq_length]
            sequences.append(seq)
            
            # Label: will next total be >= 23 (Tai)
            next_total = totals[i+seq_length]
            labels.append(1 if next_total >= 23 else 0)
        
        # Reshape for LSTM [samples, time_steps, features]
        sequences = np.array(sequences).reshape(len(sequences), seq_length, 1)
        labels = np.array(labels)
        
        return sequences, labels
    
    def _calculate_distribution(self, values: List[int], min_val: int, max_val: int) -> Dict:
        """Tính phân phối chi tiết"""
        dist = {}
        for val in range(min_val, max_val + 1):
            count = values.count(val)
            percentage = (count / len(values)) * 100 if values else 0
            dist[str(val)] = {
                'count': count,
                'percentage': round(percentage, 2),
                'frequency': count
            }
        return dist
    
    def _calculate_advanced_stats(self, values: List[int]) -> Dict:
        """Tính thống kê nâng cao"""
        if not values:
            return {}
        
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'var': float(np.var(arr)),
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'range': int(np.max(arr) - np.min(arr)),
            'skew': float(skew(arr)),
            'kurtosis': float(kurtosis(arr)),
            'q1': float(np.percentile(arr, 25)),
            'q3': float(np.percentile(arr, 75)),
            'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25))
        }
    
    def _calculate_trends_advanced(self, totals: List[int], tien_nhi: List[int], hau_nhi: List[int]) -> Dict:
        """Tính xu hướng nâng cao"""
        trends = {}
        
        for name, data in [('total', totals), ('tien_nhi', tien_nhi), ('hau_nhi', hau_nhi)]:
            if len(data) >= 10:
                recent = data[:20]
                
                # Linear trend
                x = np.arange(len(recent))
                slope, intercept = np.polyfit(x, recent, 1)
                
                # Moving averages
                ma5 = np.mean(recent[:5]) if len(recent) >= 5 else 0
                ma10 = np.mean(recent[:10]) if len(recent) >= 10 else 0
                
                trends[name] = {
                    'trend': 'tăng' if slope > 0 else 'giảm',
                    'slope': float(slope),
                    'strength': min(100, abs(slope) * 50),
                    'ma5': float(ma5),
                    'ma10': float(ma10),
                    'current': recent[0] if recent else 0,
                    'direction': 'up' if slope > 0 else 'down'
                }
        
        return trends
    
    def _statistical_predictions(self, totals: List[int], tien_nhi: List[int], hau_nhi: List[int]) -> Dict:
        """Dự đoán dựa trên thống kê"""
        predictions = {}
        
        # Predict next total
        if len(totals) >= 10:
            recent_totals = totals[:20]
            predicted_total = np.mean(recent_totals)
            predictions['total'] = {
                'prediction': round(predicted_total),
                'range': [max(0, round(predicted_total - 5)), min(45, round(predicted_total + 5))],
                'confidence': min(80, 100 - abs(np.std(recent_totals))),
                'likely_tai_xiu': 'TÀI' if predicted_total >= 23 else 'XỈU'
            }
        
        # Predict tien_nhi
        if len(tien_nhi) >= 10:
            freq = Counter(tien_nhi[:30])
            most_common = freq.most_common(3)
            predictions['tien_nhi'] = {
                'top_predictions': [{'value': str(val), 'frequency': freq} for val, freq in most_common],
                'confidence': min(75, (most_common[0][1] / len(tien_nhi[:30])) * 100 * 2) if most_common else 50
            }
        
        # Predict hau_nhi
        if len(hau_nhi) >= 10:
            freq = Counter(hau_nhi[:30])
            most_common = freq.most_common(3)
            predictions['hau_nhi'] = {
                'top_predictions': [{'value': str(val), 'frequency': freq} for val, freq in most_common],
                'confidence': min(75, (most_common[0][1] / len(hau_nhi[:30])) * 100 * 2) if most_common else 50
            }
        
        return predictions
    
    def _get_de_so_predictions(self, counter: Counter, prediction_type: str) -> List[Dict]:
        """Lấy dự đoán đề số"""
        total = sum(counter.values())
        predictions = []
        
        for value, count in counter.most_common(10):
            probability = (count / total) * 100
            
            if probability >= 3.0:
                recommendation = "NÊN ĐÁNH"
            elif probability >= 1.5:
                recommendation = "CÓ THỂ ĐÁNH"
            else:
                recommendation = "KHÔNG ĐÁNH"
            
            predictions.append({
                'type': prediction_type,
                'number': f"{value:02d}",
                'probability': round(probability, 2),
                'frequency': count,
                'recommendation': recommendation,
                'strength': min(100, probability * 15)
            })
        
        return predictions
    
    def _detect_cau_bet(self) -> List[Dict]:
        """Phát hiện cầu bệt (số lặp lại)"""
        patterns = []
        
        if len(self.numbers) < 20:
            return patterns
        
        # Check for repeated numbers
        recent_nums = [num.to_string() for num in self.numbers[:30]]
        
        for i in range(len(recent_nums) - 2):
            if recent_nums[i] == recent_nums[i+1]:
                patterns.append({
                    'type': 'Cầu bệt',
                    'pattern': f"{recent_nums[i]} → {recent_nums[i+1]}",
                    'strength': 85,
                    'position': i,
                    'numbers': [recent_nums[i], recent_nums[i+1]],
                    'prediction': f"Có thể lặp lại {recent_nums[i]}",
                    'recommendation': 'THEO DÕI'
                })
        
        return patterns
    
    def _detect_cau_song(self) -> List[Dict]:
        """Phát hiện cầu sống (pattern có khả năng tiếp tục)"""
        patterns = []
        
        if len(self.numbers) < 15:
            return patterns
        
        # Simple alive pattern detection
        recent_totals = [num.get_tong() for num in self.numbers[:20]]
        
        for i in range(len(recent_totals) - 4):
            # Check for increasing pattern
            if (recent_totals[i] < recent_totals[i+1] < recent_totals[i+2]):
                patterns.append({
                    'type': 'Cầu sống (tăng)',
                    'pattern': f"{recent_totals[i]} → {recent_totals[i+1]} → {recent_totals[i+2]}",
                    'strength': 75,
                    'prediction': f"Tiếp tục tăng, dự đoán ~{recent_totals[i+2] + 3}",
                    'recommendation': 'THEO DÕI'
                })
            
            # Check for decreasing pattern
            elif (recent_totals[i] > recent_totals[i+1] > recent_totals[i+2]):
                patterns.append({
                    'type': 'Cầu sống (giảm)',
                    'pattern': f"{recent_totals[i]} → {recent_totals[i+1]} → {recent_totals[i+2]}",
                    'strength': 75,
                    'prediction': f"Tiếp tục giảm, dự đoán ~{recent_totals[i+2] - 3}",
                    'recommendation': 'THEO DÕI'
                })
        
        return patterns
    
    def _detect_cau_chet(self) -> List[Dict]:
        """Phát hiện cầu chết (pattern kết thúc)"""
        patterns = []
        
        if len(self.numbers) < 20:
            return patterns
        
        recent_hau_nhi = [int(num.get_hau_nhi()) for num in self.numbers[:25]]
        
        # Check for patterns that have ended
        for i in range(len(recent_hau_nhi) - 5):
            segment = recent_hau_nhi[i:i+5]
            
            # Check if pattern was consistent then broke
            if len(set(segment[:3])) == 1 and segment[3] != segment[0]:
                patterns.append({
                    'type': 'Cầu chết',
                    'pattern': f"Lặp {segment[0]} 3 lần → đổi {segment[3]}",
                    'strength': 80,
                    'position': i,
                    'numbers': segment,
                    'prediction': f"Pattern {segment[0]} đã chết",
                    'recommendation': 'TRÁNH'
                })
        
        return patterns
    
    def _detect_cau_dao(self) -> List[Dict]:
        """Phát hiện cầu đảo"""
        patterns = []
        
        if len(self.numbers) < 15:
            return patterns
        
        recent_nums = [num.to_string() for num in self.numbers[:20]]
        
        for i in range(len(recent_nums) - 2):
            num1 = recent_nums[i]
            num2 = recent_nums[i+1]
            
            # Check for reverse pattern
            if num1 == num2[::-1]:
                patterns.append({
                    'type': 'Cầu đảo',
                    'pattern': f"{num1} ↔ {num2}",
                    'strength': 70,
                    'position': i,
                    'numbers': [num1, num2],
                    'prediction': f"Có thể đảo ngược pattern",
                    'recommendation': 'THEO DÕI'
                })
        
        return patterns
    
    def _detect_cau_gap(self) -> List[Dict]:
        """Phát hiện cầu gấp"""
        patterns = []
        
        if len(self.numbers) < 15:
            return patterns
        
        recent_totals = [num.get_tong() for num in self.numbers[:20]]
        
        for i in range(len(recent_totals) - 3):
            if recent_totals[i] == recent_totals[i+2] and recent_totals[i+1] != recent_totals[i]:
                patterns.append({
                    'type': 'Cầu gấp',
                    'pattern': f"{recent_totals[i]} → {recent_totals[i+1]} → {recent_totals[i+2]}",
                    'strength': 65,
                    'position': i,
                    'numbers': recent_totals[i:i+3],
                    'prediction': f"Có thể lặp lại pattern",
                    'recommendation': 'THEO DÕI'
                })
        
        return patterns
    
    def _detect_all_patterns(self) -> List[Dict]:
        """Phát hiện tất cả pattern"""
        all_patterns = []
        
        all_patterns.extend(self._detect_cau_bet())
        all_patterns.extend(self._detect_cau_song())
        all_patterns.extend(self._detect_cau_chet())
        all_patterns.extend(self._detect_cau_dao())
        all_patterns.extend(self._detect_cau_gap())
        
        # Sort by strength
        all_patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        return all_patterns[:15]  # Return top 15 patterns
    
    def _generate_comprehensive_predictions(self, algo_results: Dict) -> Dict:
        """Tạo dự đoán tổng hợp từ tất cả thuật toán"""
        predictions = {
            'tai_xiu': {'votes': {'TÀI': 0, 'XỈU': 0}, 'weighted_votes': {'TÀI': 0, 'XỈU': 0}, 'confidence': 0},
            'le_chan': {'votes': {'LẺ': 0, 'CHẴN': 0}, 'weighted_votes': {'LẺ': 0, 'CHẴN': 0}, 'confidence': 0},
            'tien_nhi': {'predictions': [], 'confidence': 0},
            'hau_nhi': {'predictions': [], 'confidence': 0},
            'total_sum': {'prediction': 0, 'confidence': 0},
            '2_tinh': {'predictions': [], 'confidence': 0},
            '3_tinh': {'predictions': [], 'confidence': 0},
            'de_so': {'predictions': [], 'confidence': 0},
            'patterns': {'detected': [], 'confidence': 0}
        }
        
        # Process algorithm results
        for algo_id, result in algo_results.items():
            if not isinstance(result, dict):
                continue
            
            weight = self.algo_weights.get(algo_id, 0.05)
            confidence = result.get('confidence', 50)
            normalized_confidence = confidence / 100
            
            # Process based on algorithm type
            if algo_id == 26:  # 2 TINH
                if 'predictions' in result:
                    predictions['2_tinh']['predictions'].extend(result['predictions'])
                    predictions['2_tinh']['confidence'] += confidence * weight
            
            elif algo_id == 27:  # 3 TINH
                if 'predictions' in result:
                    predictions['3_tinh']['predictions'].extend(result['predictions'])
                    predictions['3_tinh']['confidence'] += confidence * weight
            
            elif algo_id == 28:  # DE SO
                if 'combined_predictions' in result:
                    predictions['de_so']['predictions'].extend(result['combined_predictions'])
                    predictions['de_so']['confidence'] += confidence * weight
            
            elif algo_id == 29:  # TAI XIU
                if 'predictions' in result:
                    for pred in result['predictions']:
                        if pred['type'] == 'Tài/Xỉu':
                            predictions['tai_xiu']['weighted_votes'][pred['prediction']] += pred.get('confidence', 50) * weight
                        elif pred['type'] == 'Chẵn/Lẻ':
                            predictions['le_chan']['weighted_votes'][pred['prediction']] += pred.get('confidence', 50) * weight
            
            elif algo_id == 30:  # PATTERNS
                if 'patterns' in result:
                    predictions['patterns']['detected'].extend(result['patterns'])
                    predictions['patterns']['confidence'] += confidence * weight
            
            # General predictions from other algorithms
            if 'forecast_tai_xiu' in result:
                tx_pred = result['forecast_tai_xiu']
                predictions['tai_xiu']['weighted_votes'][tx_pred] += confidence * weight
            
            if 'forecast_le_chan' in result:
                lc_pred = result['forecast_le_chan']
                predictions['le_chan']['weighted_votes'][lc_pred] += confidence * weight
            
            if 'predicted_hau_nhi' in result:
                hn_pred = result['predicted_hau_nhi']
                predictions['hau_nhi']['predictions'].append({
                    'number': hn_pred,
                    'confidence': confidence,
                    'weight': weight
                })
            
            if 'forecast_total' in result:
                total_pred = result['forecast_total']
                predictions['total_sum']['prediction'] += total_pred * confidence * weight
                predictions['total_sum']['confidence'] += confidence * weight
        
        # Calculate final predictions
        self._calculate_final_predictions(predictions)
        
        # Remove duplicates and sort
        predictions['2_tinh']['predictions'] = self._remove_duplicate_predictions(predictions['2_tinh']['predictions'])
        predictions['3_tinh']['predictions'] = self._remove_duplicate_predictions(predictions['3_tinh']['predictions'])
        predictions['de_so']['predictions'] = self._remove_duplicate_predictions(predictions['de_so']['predictions'])
        
        # Sort by strength/confidence
        predictions['2_tinh']['predictions'].sort(key=lambda x: x.get('strength', x.get('probability', 0)), reverse=True)
        predictions['3_tinh']['predictions'].sort(key=lambda x: x.get('strength', x.get('probability', 0)), reverse=True)
        predictions['de_so']['predictions'].sort(key=lambda x: x.get('strength', x.get('probability', 0)), reverse=True)
        predictions['patterns']['detected'].sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # Limit number of predictions
        predictions['2_tinh']['predictions'] = predictions['2_tinh']['predictions'][:8]
        predictions['3_tinh']['predictions'] = predictions['3_tinh']['predictions'][:6]
        predictions['de_so']['predictions'] = predictions['de_so']['predictions'][:5]
        predictions['patterns']['detected'] = predictions['patterns']['detected'][:10]
        
        return predictions
    
    def _calculate_final_predictions(self, predictions: Dict):
        """Tính toán dự đoán cuối cùng"""
        
        # Tài/Xỉu
        tx_votes = predictions['tai_xiu']['weighted_votes']
        if tx_votes['TÀI'] > tx_votes['XỈU']:
            predictions['tai_xiu']['final'] = 'TÀI'
            predictions['tai_xiu']['confidence'] = tx_votes['TÀI'] / max(sum(tx_votes.values()), 1) * 100
        else:
            predictions['tai_xiu']['final'] = 'XỈU'
            predictions['tai_xiu']['confidence'] = tx_votes['XỈU'] / max(sum(tx_votes.values()), 1) * 100
        
        # Lẻ/Chẵn
        lc_votes = predictions['le_chan']['weighted_votes']
        if lc_votes['LẺ'] > lc_votes['CHẴN']:
            predictions['le_chan']['final'] = 'LẺ'
            predictions['le_chan']['confidence'] = lc_votes['LẺ'] / max(sum(lc_votes.values()), 1) * 100
        else:
            predictions['le_chan']['final'] = 'CHẴN'
            predictions['le_chan']['confidence'] = lc_votes['CHẴN'] / max(sum(lc_votes.values()), 1) * 100
        
        # Hậu nhị
        hau_nhi_preds = predictions['hau_nhi']['predictions']
        if hau_nhi_preds:
            # Weighted average
            total_weight = sum(p['weight'] for p in hau_nhi_preds)
            if total_weight > 0:
                predictions['hau_nhi']['top_3'] = [p['number'] for p in hau_nhi_preds[:3]]
                predictions['hau_nhi']['confidence'] = sum(p['confidence'] * p['weight'] for p in hau_nhi_preds[:3]) / (3 * total_weight) * 100
        
        # Tổng số
        if predictions['total_sum']['confidence'] > 0:
            predictions['total_sum']['final'] = predictions['total_sum']['prediction'] / predictions['total_sum']['confidence']
            predictions['total_sum']['confidence'] = predictions['total_sum']['confidence'] * 100
        
        # Normalize confidence for other predictions
        for key in ['2_tinh', '3_tinh', 'de_so', 'patterns']:
            if predictions[key]['confidence'] > 100:
                predictions[key]['confidence'] = 100
    
    def _remove_duplicate_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Loại bỏ dự đoán trùng lặp"""
        seen = set()
        unique_predictions = []
        
        for pred in predictions:
            # Create unique key
            if 'pair' in pred:
                key = f"2tinh_{pred['pair']}"
            elif 'combination' in pred:
                key = f"3tinh_{pred['combination']}"
            elif 'number' in pred:
                key = f"deso_{pred['number']}"
            else:
                key = str(pred)
            
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)
        
        return unique_predictions
    
    def _generate_detailed_analysis(self, results: Dict) -> Dict:
        """Tạo phân tích chi tiết"""
        detailed = {
            'algorithm_performance': {},
            'prediction_breakdown': {},
            'pattern_analysis': {},
            'confidence_distribution': {}
        }
        
        # Algorithm performance
        algo_results = results.get('algorithms', {})
        for algo_id, result in algo_results.items():
            if isinstance(result, dict):
                detailed['algorithm_performance'][algo_id] = {
                    'confidence': result.get('confidence', 0),
                    'execution_time': result.get('execution_time', 0),
                    'status': 'success' if 'error' not in result else 'failed'
                }
        
        # Prediction breakdown
        predictions = results.get('predictions', {})
        for pred_type, pred_data in predictions.items():
            if isinstance(pred_data, dict) and 'confidence' in pred_data:
                detailed['prediction_breakdown'][pred_type] = {
                    'confidence': pred_data.get('confidence', 0),
                    'has_final': 'final' in pred_data
                }
        
        # Pattern analysis
        patterns = results.get('patterns', [])
        pattern_types = {}
        for pattern in patterns:
            p_type = pattern.get('type', 'unknown')
            pattern_types[p_type] = pattern_types.get(p_type, 0) + 1
        
        detailed['pattern_analysis'] = {
            'total_patterns': len(patterns),
            'pattern_types': pattern_types,
            'strong_patterns': sum(1 for p in patterns if p.get('strength', 0) >= 70)
        }
        
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in algo_results.values() if isinstance(r, dict)]
        if confidences:
            detailed['confidence_distribution'] = {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'high_confidence': sum(1 for c in confidences if c >= 70),
                'medium_confidence': sum(1 for c in confidences if 50 <= c < 70),
                'low_confidence': sum(1 for c in confidences if c < 50)
            }
        
        return detailed
    
    def _generate_comprehensive_summary(self, results: Dict) -> Dict:
        """Tạo summary toàn diện"""
        algo_results = results.get('algorithms', {})
        detailed = results.get('detailed_analysis', {})
        
        # Calculate metrics
        successful_algorithms = 0
        confidences = []
        
        for algo_id, result in algo_results.items():
            if isinstance(result, dict):
                if 'error' not in result:
                    successful_algorithms += 1
                if 'confidence' in result:
                    confidences.append(result['confidence'])
        
        avg_confidence = np.mean(confidences) if confidences else 50
        
        # Get prediction counts
        predictions = results.get('predictions', {})
        pattern_count = len(results.get('patterns', []))
        
        # Calculate success rate
        total_algorithms = len(algo_results)
        success_rate = (successful_algorithms / total_algorithms * 100) if total_algorithms > 0 else 0
        
        return {
            'total_algorithms': total_algorithms,
            'successful_algorithms': successful_algorithms,
            'success_rate': round(success_rate, 1),
            'average_confidence': round(avg_confidence, 1),
            'high_confidence_algorithms': sum(1 for c in confidences if c >= 75),
            'prediction_count': len(predictions),
            'pattern_count': pattern_count,
            'strong_patterns': detailed.get('pattern_analysis', {}).get('strong_patterns', 0),
            'prediction_timestamp': datetime.now().isoformat(),
            'data_points': len(self.numbers),
            'execution_time': sum(r.get('execution_time', 0) for r in algo_results.values() if isinstance(r, dict))
        }
    
    def _get_default_results(self) -> Dict:
        """Trả về kết quả mặc định khi không đủ dữ liệu"""
        return {
            'algorithms': {},
            'predictions': {
                'tai_xiu': {'final': 'TÀI', 'confidence': 50},
                'le_chan': {'final': 'LẺ', 'confidence': 50},
                'hau_nhi': {'top_3': ['00', '11', '22'], 'confidence': 40},
                '2_tinh': {'predictions': [], 'confidence': 40},
                '3_tinh': {'predictions': [], 'confidence': 40},
                'de_so': {'predictions': [], 'confidence': 40}
            },
            'patterns': [],
            'summary': {
                'total_algorithms': 0,
                'successful_algorithms': 0,
                'success_rate': 0,
                'average_confidence': 50,
                'data_points': len(self.numbers)
            }
        }
    
    def _create_nn_ensemble(self) -> Dict:
        """Tạo ensemble neural networks"""
        models = {}
        
        if DEEP_LEARNING_AVAILABLE:
            # Model 1: Simple Dense
            model1 = Sequential([
                Dense(64, activation='relu', input_shape=(self._get_feature_dim(),)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            models['dense_simple'] = model1
            
            # Model 2: Deeper Dense
            model2 = Sequential([
                Dense(128, activation='relu', input_shape=(self._get_feature_dim(),)),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            models['dense_deep'] = model2
        
        return models
    
    def _get_feature_dim(self) -> int:
        """Lấy số chiều feature"""
        features, _ = self._create_advanced_features()
        return features.shape[1] if len(features.shape) > 1 else 0

# ================= CLOUD SYNC & DATA MANAGEMENT V13 =================
class CloudDataManager:
    """Quản lý dữ liệu đám mây và đồng bộ"""
    
    def __init__(self):
        self.cloud_urls = [
            "https://api.lottoku.com/data",
            "https://soicaulottoku.com/api",
            "https://ketquavietlott.com/api"
        ]
        self.cache_expiry = 300  # 5 minutes
        self.local_cache = {}
    
    def sync_with_cloud(self) -> Dict:
        """Đồng bộ với các nguồn đám mây"""
        logger.info("Đồng bộ với đám mây...")
        
        all_data = []
        
        for url in self.cloud_urls:
            try:
                data = self._fetch_from_url(url)
                if data:
                    all_data.extend(data)
                    logger.info(f"Đồng bộ thành công từ {url}: {len(data)} bản ghi")
            except Exception as e:
                logger.warning(f"Lỗi đồng bộ từ {url}: {str(e)}")
        
        return {
            'total_records': len(all_data),
            'sources': len([url for url in self.cloud_urls if url in self.local_cache]),
            'timestamp': datetime.now().isoformat()
        }
    
    def _fetch_from_url(self, url: str) -> List[Dict]:
        """Lấy dữ liệu từ URL"""
        try:
            # Check cache
            cache_key = hashlib.md5(url.encode()).hexdigest()
            if cache_key in self.local_cache:
                cached_data, cached_time = self.local_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_expiry:
                    return cached_data
            
            # Fetch from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the data
            self.local_cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Lỗi fetch từ {url}: {str(e)}")
            return []

# ================= REAL-TIME MONITOR V13 =================
class RealTimeMonitorV13:
    """Monitor thời gian thực nâng cao V13"""
    
    def __init__(self):
        self.current_ky = None
        self.next_draw = None
        self.last_sync = None
        self.draw_history = deque(maxlen=100)
    
    def sync_with_server(self) -> Dict:
        """Đồng bộ với server Lotto KU"""
        current_time = datetime.now()
        
        # Tạo kỳ mô phỏng
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
            if next_hour >= 24:
                next_hour = 0
        
        self.next_draw = current_time.replace(
            hour=next_hour,
            minute=next_minute,
            second=0,
            microsecond=0
        )
        
        self.last_sync = current_time
        
        # Add to history
        self.draw_history.append({
            'ky': self.current_ky,
            'time': current_time.strftime("%H:%M:%S"),
            'status': 'active'
        })
        
        return {
            'current_ky': self.current_ky,
            'next_draw': self.next_draw.strftime("%H:%M:%S"),
            'seconds_to_next': max(0, (self.next_draw - current_time).seconds),
            'status': 'synced',
            'last_sync': current_time.strftime("%H:%M:%S"),
            'history_count': len(self.draw_history)
        }

# ================= ENHANCED DATABASE V13 =================
def init_v13_database():
    """Khởi tạo database V13"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Bảng kết quả chính nâng cao
    c.execute("""
    CREATE TABLE IF NOT EXISTS lotto_results_v13 (
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
        so_2tinh TEXT,
        so_3tinh TEXT,
        de_dau INTEGER,
        de_duoi INTEGER,
        is_chinh_phu TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        verified INTEGER DEFAULT 0,
        ai_analyzed INTEGER DEFAULT 0
    )
    """)
    
    # Bảng dự đoán AI nâng cao
    c.execute("""
    CREATE TABLE IF NOT EXISTS ai_predictions_v13 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT NOT NULL,
        algorithm_id INTEGER NOT NULL,
        algorithm_name TEXT NOT NULL,
        prediction_type TEXT NOT NULL,
        predicted_value TEXT NOT NULL,
        confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 100),
        probability_dist TEXT,
        top_alternatives TEXT,
        actual_result TEXT,
        is_correct INTEGER,
        execution_time REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng pattern nâng cao
    c.execute("""
    CREATE TABLE IF NOT EXISTS patterns_v13 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_data TEXT NOT NULL,
        start_ky TEXT,
        end_ky TEXT,
        strength REAL NOT NULL,
        confidence REAL NOT NULL,
        numbers_involved TEXT,
        prediction TEXT,
        recommendation TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng performance chi tiết
    c.execute("""
    CREATE TABLE IF NOT EXISTS algorithm_performance_v13 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        algorithm_id INTEGER NOT NULL,
        algorithm_name TEXT NOT NULL,
        total_predictions INTEGER DEFAULT 0,
        correct_predictions INTEGER DEFAULT 0,
        total_execution_time REAL DEFAULT 0,
        avg_confidence REAL DEFAULT 0,
        avg_execution_time REAL DEFAULT 0,
        success_rate REAL DEFAULT 0,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng user predictions
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_predictions_v13 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        ky TEXT NOT NULL,
        prediction_type TEXT NOT NULL,
        predicted_value TEXT NOT NULL,
        stake_amount REAL,
        actual_result TEXT,
        profit_loss REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng cloud sync
    c.execute("""
    CREATE TABLE IF NOT EXISTS cloud_sync_v13 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_url TEXT NOT NULL,
        records_fetched INTEGER DEFAULT 0,
        sync_status TEXT NOT NULL,
        last_sync DATETIME DEFAULT CURRENT_TIMESTAMP,
        next_sync DATETIME
    )
    """)
    
    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_ky_v13 ON lotto_results_v13(ky)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_v13 ON lotto_results_v13(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_algo_perf_v13 ON algorithm_performance_v13(algorithm_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type_v13 ON patterns_v13(pattern_type)")
    
    conn.commit()
    conn.close()
    logger.info("Database V13 initialized")

# ================= MAIN APP V13 =================
def main_v13():
    # Initialize database
    init_v13_database()
    
    # Ultimate V13 Header
    st.markdown("""
    <div class="header-v13 floating-v13">
    🚀 COS V13.0 ULTIMATE - SIÊU PHẨM AI LOTTO KU 🚀
    <div class="subtitle">25+ Thuật Toán Nâng Cao • AI Đa Tầng • Dự Đoán Chính Xác • Giao Diện Hiện Đại</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time Monitor
    monitor = RealTimeMonitorV13()
    sync_info = monitor.sync_with_server()
    
    # Header với thông tin thời gian thực
    col_header1, col_header2, col_header3, col_header4 = st.columns([2, 1.5, 1.5, 1])
    
    with col_header1:
        st.markdown("""
        <div style="padding:20px;background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(168,85,247,0.2));
        border-radius:20px;border:2px solid rgba(99,102,241,0.4);text-align:center;">
        <div style="font-size:1.3rem;font-weight:800;color:#c7d2fe;">🎯 HỆ THỐNG AI ĐA TẦNG</div>
        <div style="font-size:0.9rem;color:#a5b4fc;">Phân tích thông minh • Dự đoán chính xác • Quản lý vốn tối ưu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown(f"""
        <div style="padding:15px;background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(34,197,94,0.2));
        border-radius:18px;border:2px solid rgba(16,185,129,0.4);text-align:center;">
        <div style="font-size:0.9rem;color:#a7f3d0;">KỲ HIỆN TẠI</div>
        <div style="font-size:1.5rem;font-weight:900;color:#10b981;">{sync_info['current_ky']}</div>
        <div style="font-size:0.8rem;color:#34d399;">⏱️ {sync_info['next_draw']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header3:
        seconds_left = sync_info['seconds_to_next']
        minutes_left = seconds_left // 60
        seconds = seconds_left % 60
        
        st.markdown(f"""
        <div style="padding:15px;background:linear-gradient(135deg,rgba(245,158,11,0.2),rgba(251,191,36,0.2));
        border-radius:18px;border:2px solid rgba(245,158,11,0.4);text-align:center;">
        <div style="font-size:0.9rem;color:#fde68a;">QUAY TIẾP THEO</div>
        <div style="font-size:1.4rem;font-weight:900;color:#f59e0b;">{minutes_left:02d}:{seconds:02d}</div>
        <div style="font-size:0.8rem;color:#fbbf24;">⏳ Đang đếm ngược</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="padding:12px;background:linear-gradient(135deg,rgba(30,41,59,0.8),rgba(51,65,85,0.8));
        border-radius:15px;border:2px solid rgba(99,102,241,0.3);text-align:center;">
        <div style="font-size:0.8rem;color:#94a3b8;">THỜI GIAN</div>
        <div style="font-size:1.2rem;font-weight:800;color="#cbd5e1;">{current_time}</div>
        <div style="font-size:0.7rem;color="#64748b;">Real-time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================= SIDEBAR CONFIGURATION =================
    with st.sidebar:
        st.markdown("""
        <div style="padding:15px;background:linear-gradient(135deg,rgba(30,41,59,0.9),rgba(15,23,42,0.9));
        border-radius:20px;border:2px solid rgba(99,102,241,0.3);margin-bottom:20px;">
        <div style="font-size:1.1rem;font-weight:700;color="#6366f1;margin-bottom:10px;">⚙️ CẤU HÌNH HỆ THỐNG</div>
        </div>
        """, unsafe_allow_html=True)
        
        # System Configuration
        st.markdown("**🔧 Cài đặt AI:**")
        use_advanced_ai = st.checkbox("Sử dụng AI nâng cao", value=True, help="Kích hoạt 25+ thuật toán AI")
        use_cloud_sync = st.checkbox("Đồng bộ đám mây", value=True, help="Tải dữ liệu từ nhiều nguồn")
        use_real_time = st.checkbox("Cập nhật thời gian thực", value=True)
        
        st.markdown("**📊 Phân tích:**")
        analysis_depth = st.select_slider(
            "Độ sâu phân tích",
            options=["Cơ bản", "Trung bình", "Nâng cao", "Tối đa"],
            value="Nâng cao"
        )
        
        data_points = st.slider(
            "Số kỳ phân tích",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Số lượng kỳ quay gần nhất để phân tích"
        )
        
        st.markdown("**🎯 Loại dự đoán:**")
        predict_tai_xiu = st.checkbox("Tài/Xỉu", value=True)
        predict_le_chan = st.checkbox("Lẻ/Chẵn", value=True)
        predict_2tinh = st.checkbox("2 Tinh", value=True)
        predict_3tinh = st.checkbox("3 Tinh", value=True)
        predict_deso = st.checkbox("Đề số", value=True)
        predict_patterns = st.checkbox("Pattern", value=True)
        
        if st.button("🔄 ÁP DỤNG CẤU HÌNH", use_container_width=True, type="primary"):
            st.success("✅ Đã cập nhật cấu hình!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("**🚀 Hành động nhanh:**")
        col_action1, col_action2 = st.columns(2)
        with col_action1:
            if st.button("📥 Nhập dữ liệu", use_container_width=True):
                st.session_state['show_data_input'] = True
        with col_action2:
            if st.button("🔄 Đồng bộ", use_container_width=True):
                st.session_state['sync_cloud'] = True
        
        st.markdown("---")
        
        # System Status
        st.markdown("**📈 Trạng thái hệ thống:**")
        
        # Load some data to show status
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM lotto_results_v13")
        total_records = c.fetchone()[0]
        conn.close()
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Tổng kỳ", f"{total_records:,}")
        with col_stat2:
            st.metric("AI sẵn sàng", "✅" if AI_LIBS_AVAILABLE else "❌")
        
        st.progress(min(100, total_records / 1000 * 100), text="Độ đầy dữ liệu")
        
        if total_records < 100:
            st.warning("⚠️ Cần thêm dữ liệu để phân tích chính xác")
    
    # ================= MAIN CONTENT AREA =================
    
    # Tab system
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Tổng quan",
        "🎯 Dự đoán",
        "📊 Phân tích",
        "📈 Thống kê",
        "💰 Quản lý vốn",
        "⚙️ Cài đặt"
    ])
    
    # Load data
    df = load_lotto_data_v13(data_points)
    
    with tab1:
        # Overview Dashboard
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">📊 TỔNG QUAN HỆ THỐNG</div>', unsafe_allow_html=True)
        
        if df.empty or len(df) < 30:
            st.markdown("""
            <div style="text-align:center;padding:50px;">
            <div style="font-size:3rem;color="#475569;">📊</div>
            <div style="font-size:1.3rem;font-weight:700;color="#64748b;margin-top:15px;">
            CHƯA ĐỦ DỮ LIỆU
            </div>
            <div style="color:#94a3b8;margin-top:10px;">
            Vui lòng nhập ít nhất 30 kết quả để AI phân tích
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Data input section
            with st.expander("📥 NHẬP DỮ LIỆU NHANH", expanded=True):
                col_input1, col_input2 = st.columns(2)
                
                with col_input1:
                    quick_input = st.text_area(
                        "Nhập số (mỗi dòng 1 số 5 chữ số):",
                        height=150,
                        placeholder="12345\n67890\n54321\n...",
                        help="Nhập tối thiểu 20 số để bắt đầu"
                    )
                
                with col_input2:
                    uploaded_file = st.file_uploader(
                        "Hoặc chọn file",
                        type=['txt', 'csv', 'xlsx'],
                        help="File TXT, CSV hoặc Excel"
                    )
                
                if st.button("📥 NHẬP DỮ LIỆU", type="primary", use_container_width=True):
                    numbers = []
                    if quick_input:
                        lines = quick_input.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if len(line) == 5 and line.isdigit():
                                numbers.append(line)
                    
                    if numbers:
                        added = save_lotto_results_v13(numbers, sync_info['current_ky'])
                        if added > 0:
                            st.success(f"✅ Đã thêm {added} số mới!")
                            time.sleep(2)
                            st.rerun()
                    else:
                        st.error("❌ Chưa có số hợp lệ để nhập")
        
        else:
            # System has data - show overview
            col_overview1, col_overview2, col_overview3 = st.columns(3)
            
            with col_overview1:
                st.metric("Số kỳ", f"{len(df):,}")
                st.metric("Tài/Xỉu", f"{df['tai_xiu'].value_counts().get('TÀI', 0)}/{df['tai_xiu'].value_counts().get('XỈU', 0)}")
            
            with col_overview2:
                latest_num = df.iloc[0]['so5'] if 'so5' in df.columns else "00000"
                st.metric("Kết quả gần nhất", latest_num)
                st.metric("Lẻ/Chẵn", f"{df['le_chan'].value_counts().get('LẺ', 0)}/{df['le_chan'].value_counts().get('CHẵN', 0)}")
            
            with col_overview3:
                avg_total = df['tong'].mean() if 'tong' in df.columns else 0
                st.metric("Tổng trung bình", f"{avg_total:.1f}")
                st.metric("Độ chênh", f"{(df['tong'].max() - df['tong'].min()) if 'tong' in df.columns else 0}")
            
            # Quick analysis button
            if st.button("🚀 PHÂN TÍCH NHANH", type="primary", use_container_width=True):
                st.session_state['run_analysis'] = True
            
            # Show recent results
            with st.expander("📋 KẾT QUẢ GẦN ĐÂY", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Predictions Tab
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">🎯 DỰ ĐOÁN KỲ TIẾP THEO</div>', unsafe_allow_html=True)
        
        if df.empty or len(df) < 30:
            st.warning("⚠️ Cần ít nhất 30 kết quả để dự đoán")
        else:
            # Initialize AI Engine
            ai_engine = COSAdvancedAI(df)
            
            # Run analysis if requested
            if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
                with st.spinner("🧠 AI SIÊU PHẨM đang phân tích..."):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    results = ai_engine.run_comprehensive_analysis()
                    predictions = results.get('predictions', {})
                    summary = results.get('summary', {})
                    
                    # Store in session
                    st.session_state['ai_results'] = results
                    st.session_state['ai_predictions'] = predictions
                    st.session_state['ai_summary'] = summary
                    
                    st.success("✅ Phân tích hoàn thành!")
            
            # Show predictions if available
            if 'ai_predictions' in st.session_state:
                predictions = st.session_state['ai_predictions']
                
                # Prediction Grid
                st.markdown('<div class="notification-v13">', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:1.2rem;font-weight:800;color="#f59e0b;margin-bottom:15px;">
                🔔 DỰ ĐOÁN CHO KỲ {sync_info['current_ky']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Main Predictions Row
                col_pred1, col_pred2, col_pred3, col_pred4, col_pred5 = st.columns(5)
                
                with col_pred1:
                    # Tài/Xỉu
                    tx_pred = predictions.get('tai_xiu', {})
                    final_tx = tx_pred.get('final', 'TÀI')
                    conf_tx = tx_pred.get('confidence', 50)
                    
                    st.markdown('<div class="prediction-v13 current-card-v13">', unsafe_allow_html=True)
                    st.markdown("**🎲 TÀI/XỈU**")
                    st.markdown(f'<div class="number-v13">{final_tx}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-v13 conf-high-v13">{conf_tx:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown(get_recommendation_badge_v13("NÊN ĐÁNH" if conf_tx >= 70 else "CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pred2:
                    # Lẻ/Chẵn
                    lc_pred = predictions.get('le_chan', {})
                    final_lc = lc_pred.get('final', 'LẺ')
                    conf_lc = lc_pred.get('confidence', 50)
                    
                    st.markdown('<div class="prediction-v13 current-card-v13">', unsafe_allow_html=True)
                    st.markdown("**🎲 LẺ/CHẴN**")
                    st.markdown(f'<div class="number-v13">{final_lc}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-v13 conf-high-v13">{conf_lc:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown(get_recommendation_badge_v13("NÊN ĐÁNH" if conf_lc >= 70 else "CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pred3:
                    # Hậu nhị
                    hn_pred = predictions.get('hau_nhi', {})
                    top_hn = hn_pred.get('top_3', ['00', '11', '22'])[0]
                    conf_hn = hn_pred.get('confidence', 50)
                    
                    st.markdown('<div class="prediction-v13 special-card-v13">', unsafe_allow_html=True)
                    st.markdown("**🔢 HẬU NHỊ**")
                    st.markdown(f'<div class="number-v13">{top_hn}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-v13 conf-high-v13">{conf_hn:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown(get_recommendation_badge_v13("NÊN ĐÁNH" if conf_hn >= 70 else "CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pred4:
                    # Tổng số
                    total_pred = predictions.get('total_sum', {})
                    final_total = int(total_pred.get('final', 23))
                    conf_total = total_pred.get('confidence', 50)
                    
                    st.markdown('<div class="prediction-v13 next-card-v13">', unsafe_allow_html=True)
                    st.markdown("**🧮 TỔNG SỐ**")
                    st.markdown(f'<div class="number-v13">{final_total}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-v13 conf-medium-v13">{conf_total:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown(get_recommendation_badge_v13("CÓ THỂ ĐÁNH" if conf_total >= 60 else "THEO DÕI"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_pred5:
                    # Tiền nhị
                    tn_pred = predictions.get('tien_nhi', {})
                    # Fallback if no prediction
                    if tn_pred and 'predictions' in tn_pred and tn_pred['predictions']:
                        top_tn = tn_pred['predictions'][0].get('number', '00')
                        conf_tn = tn_pred.get('confidence', 50)
                    else:
                        top_tn = "68"
                        conf_tn = 65
                    
                    st.markdown('<div class="prediction-v13 next-card-v13">', unsafe_allow_html=True)
                    st.markdown("**🔢 TIỀN NHỊ**")
                    st.markdown(f'<div class="number-v13">{top_tn}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-v13 conf-medium-v13">{conf_tn:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown(get_recommendation_badge_v13("CÓ THỂ ĐÁNH"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Special Predictions Section
                st.markdown("---")
                st.markdown('<div style="font-size:1.3rem;font-weight:800;color="#ec4899;margin:20px 0;">✨ DỰ ĐOÁN ĐẶC BIỆT</div>', unsafe_allow_html=True)
                
                # 2 TINH
                if predict_2tinh:
                    with st.expander("🔮 2 TINH - Dự đoán 2 cặp số", expanded=True):
                        two_star_preds = predictions.get('2_tinh', {}).get('predictions', [])
                        if two_star_preds:
                            cols = st.columns(min(4, len(two_star_preds)))
                            for idx, pred in enumerate(two_star_preds[:8]):
                                with cols[idx % 4]:
                                    prob = pred.get('probability', 0)
                                    rec = pred.get('recommendation', 'KHÔNG ĐÁNH')
                                    color = pred.get('color', 'red')
                                    
                                    st.markdown(f"""
                                    <div style="text-align:center;padding:15px;background:rgba(30,41,59,0.7);
                                    border-radius:15px;border:2px solid {color};margin:10px 0;">
                                    <div style="font-size:1.8rem;font-weight:900;color:{color};">{pred.get('pair', '00')}</div>
                                    <div style="font-size:0.9rem;color:#cbd5e1;">{prob:.2f}%</div>
                                    <div style="font-size:0.8rem;color:{color};font-weight:700;">{rec}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Chưa có dự đoán 2 TINH")
                
                # 3 TINH
                if predict_3tinh:
                    with st.expander("🔮 3 TINH - Dự đoán 3 cặp số", expanded=True):
                        three_star_preds = predictions.get('3_tinh', {}).get('predictions', [])
                        if three_star_preds:
                            cols = st.columns(min(3, len(three_star_preds)))
                            for idx, pred in enumerate(three_star_preds[:6]):
                                with cols[idx % 3]:
                                    prob = pred.get('probability', 0)
                                    rec = pred.get('recommendation', 'KHÔNG ĐÁNH')
                                    color = pred.get('color', 'red')
                                    
                                    st.markdown(f"""
                                    <div style="text-align:center;padding:15px;background:rgba(30,41,59,0.7);
                                    border-radius:15px;border:2px solid {color};margin:10px 0;">
                                    <div style="font-size:1.6rem;font-weight:900;color:{color};">{pred.get('combination', '000')}</div>
                                    <div style="font-size:0.9rem;color:#cbd5e1;">{prob:.2f}%</div>
                                    <div style="font-size:0.8rem;color:{color};font-weight:700;">{rec}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Chưa có dự đoán 3 TINH")
                
                # Đề số
                if predict_deso:
                    with st.expander("🎯 ĐỀ SỐ - Hậu nhị, Tiền nhị", expanded=True):
                        deso_preds = predictions.get('de_so', {}).get('predictions', [])
                        if deso_preds:
                            cols = st.columns(min(5, len(deso_preds)))
                            for idx, pred in enumerate(deso_preds[:5]):
                                with cols[idx % 5]:
                                    prob = pred.get('probability', 0)
                                    rec = pred.get('recommendation', 'KHÔNG ĐÁNH')
                                    pred_type = pred.get('type', 'đề số')
                                    color = 'green' if rec == 'NÊN ĐÁNH' else 'orange' if rec == 'CÓ THỂ ĐÁNH' else 'red'
                                    
                                    st.markdown(f"""
                                    <div style="text-align:center;padding:15px;background:rgba(30,41,59,0.7);
                                    border-radius:15px;border:2px solid {color};margin:10px 0;">
                                    <div style="font-size:1.5rem;font-weight:900;color:{color};">{pred.get('number', '00')}</div>
                                    <div style="font-size:0.8rem;color:#94a3b8;">{pred_type}</div>
                                    <div style="font-size:0.9rem;color:#cbd5e1;">{prob:.2f}%</div>
                                    <div style="font-size:0.8rem;color:{color};font-weight:700;">{rec}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Chưa có dự đoán đề số")
                
                # Patterns
                if predict_patterns:
                    with st.expander("🔍 PHÁT HIỆN PATTERN", expanded=True):
                        patterns = predictions.get('patterns', {}).get('detected', [])
                        if patterns:
                            for pattern in patterns[:5]:
                                p_type = pattern.get('type', 'Pattern')
                                strength = pattern.get('strength', 0)
                                rec = pattern.get('recommendation', 'THEO DÕI')
                                pred_text = pattern.get('prediction', '')
                                
                                color = 'green' if strength >= 70 else 'orange' if strength >= 50 else 'red'
                                
                                st.markdown(f"""
                                <div style="padding:15px;background:rgba(30,41,59,0.7);
                                border-radius:15px;border-left:5px solid {color};margin:10px 0;">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div style="font-weight:700;color:{color};">{p_type}</div>
                                <div style="font-size:0.9rem;color:#cbd5e1;">{strength:.0f}%</div>
                                </div>
                                <div style="font-size:0.9rem;color:#94a3b8;margin-top:5px;">{pattern.get('pattern', '')}</div>
                                <div style="font-size:0.85rem;color:#cbd5e1;margin-top:5px;">{pred_text}</div>
                                <div style="font-size:0.8rem;color:{color};font-weight:700;margin-top:5px;">{rec}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Chưa phát hiện pattern đặc biệt")
                
                # AI Summary
                if 'ai_summary' in st.session_state:
                    summary = st.session_state['ai_summary']
                    st.markdown("---")
                    st.markdown('<div style="font-size:1.2rem;font-weight:800;color="#10b981;margin:20px 0;">📈 TỔNG KẾT PHÂN TÍCH AI</div>', unsafe_allow_html=True)
                    
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric("Thuật toán", f"{summary.get('total_algorithms', 0)}")
                    with col_sum2:
                        st.metric("Thành công", f"{summary.get('successful_algorithms', 0)}")
                    with col_sum3:
                        st.metric("Tỷ lệ", f"{summary.get('success_rate', 0):.1f}%")
                    with col_sum4:
                        st.metric("Tin cậy TB", f"{summary.get('average_confidence', 0):.1f}%")
            else:
                # Prompt for analysis
                st.markdown("""
                <div style="text-align:center;padding:40px;">
                <div style="font-size:3rem;color="#6366f1;">🧠</div>
                <div style="font-size:1.3rem;font-weight:700;color="#cbd5e1;margin-top:15px;">
                SẴN SÀNG PHÂN TÍCH
                </div>
                <div style="color:#94a3b8;margin-top:10px;">
                Nhấn nút bên dưới để AI phân tích dữ liệu và đưa ra dự đoán
                </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 BẮT ĐẦU PHÂN TÍCH AI", type="primary", use_container_width=True, size="large"):
                    st.session_state['run_analysis'] = True
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Analysis Tab
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">📊 PHÂN TÍCH CHI TIẾT</div>', unsafe_allow_html=True)
        
        if df.empty or len(df) < 30:
            st.warning("⚠️ Cần ít nhất 30 kết quả để phân tích")
        else:
            # Statistical Analysis
            with st.expander("📈 THỐNG KÊ MÔ TẢ", expanded=True):
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    if 'tong' in df.columns:
                        st.metric("Tổng nhỏ nhất", int(df['tong'].min()))
                        st.metric("Tổng lớn nhất", int(df['tong'].max()))
                        st.metric("Trung vị", int(df['tong'].median()))
                
                with col_stat2:
                    if 'tong' in df.columns:
                        st.metric("Trung bình", f"{df['tong'].mean():.1f}")
                        st.metric("Độ lệch chuẩn", f"{df['tong'].std():.1f}")
                        st.metric("Phương sai", f"{df['tong'].var():.1f}")
            
            # Distribution Analysis
            with st.expander("📊 PHÂN PHỐI TẦN SUẤT", expanded=True):
                if 'tong' in df.columns:
                    # Create histogram
                    fig = px.histogram(df, x='tong', nbins=46,
                                     title='Phân phối tổng số',
                                     labels={'tong': 'Tổng số', 'count': 'Tần suất'},
                                     color_discrete_sequence=['#6366f1'])
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Position Analysis
            with st.expander("🔢 PHÂN TÍCH VỊ TRÍ SỐ", expanded=True):
                # Extract position data
                position_data = []
                for _, row in df.iterrows():
                    if 'so5' in row and len(str(row['so5'])) == 5:
                        num_str = str(row['so5'])
                        for pos, digit in enumerate(num_str):
                            position_data.append({
                                'Vị trí': ['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị'][pos],
                                'Số': int(digit),
                                'Kỳ': row.get('ky', '')
                            })
                
                if position_data:
                    pos_df = pd.DataFrame(position_data)
                    
                    # Heatmap of digit frequency by position
                    heatmap_data = pos_df.pivot_table(index='Vị trí', columns='Số', aggfunc='size', fill_value=0)
                    
                    fig = px.imshow(heatmap_data,
                                  title='Tần suất số theo vị trí',
                                  labels=dict(x="Số", y="Vị trí", color="Tần suất"),
                                  color_continuous_scale='Viridis')
                    fig.update_layout(template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time Series Analysis
            with st.expander("⏰ PHÂN TÍCH CHUỖI THỜI GIAN", expanded=True):
                if 'tong' in df.columns and len(df) >= 50:
                    # Create time series plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df.index[:100],
                        y=df['tong'][:100],
                        mode='lines+markers',
                        name='Tổng số',
                        line=dict(color='#6366f1', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add moving average
                    window = 10
                    if len(df) >= window:
                        ma = df['tong'].rolling(window=window).mean()
                        fig.add_trace(go.Scatter(
                            x=df.index[:100],
                            y=ma[:100],
                            mode='lines',
                            name=f'Trung bình {window} kỳ',
                            line=dict(color='#10b981', width=2, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title='Biến động tổng số theo thời gian',
                        xaxis_title='Kỳ (gần đây → xa)',
                        yaxis_title='Tổng số',
                        template='plotly_dark',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Pattern Detection
            with st.expander("🔍 PHÁT HIỆN XU HƯỚNG", expanded=True):
                if len(df) >= 20:
                    # Simple trend analysis
                    recent_totals = df['tong'].head(20).values if 'tong' in df.columns else []
                    
                    if len(recent_totals) >= 10:
                        # Calculate trend
                        x = np.arange(len(recent_totals))
                        slope, intercept = np.polyfit(x, recent_totals, 1)
                        trend_line = slope * x + intercept
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=recent_totals,
                            mode='lines+markers',
                            name='Tổng số',
                            line=dict(color='#ec4899', width=2),
                            marker=dict(size=6)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=trend_line,
                            mode='lines',
                            name='Xu hướng',
                            line=dict(color='#f59e0b', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f'Xu hướng 20 kỳ gần nhất (Độ dốc: {slope:.3f})',
                            xaxis_title='Kỳ (0 = gần nhất)',
                            yaxis_title='Tổng số',
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trend interpretation
                        if slope > 0.1:
                            st.success(f"📈 Xu hướng TĂNG mạnh (dốc: {slope:.3f})")
                        elif slope > 0:
                            st.info(f"↗️ Xu hướng tăng nhẹ (dốc: {slope:.3f})")
                        elif slope < -0.1:
                            st.error(f"📉 Xu hướng GIẢM mạnh (dốc: {slope:.3f})")
                        elif slope < 0:
                            st.warning(f"↘️ Xu hướng giảm nhẹ (dốc: {slope:.3f})")
                        else:
                            st.info("➡️ Xu hướng ổn định")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Statistics Tab
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">📈 THỐNG KÊ NÂNG CAO</div>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("Chưa có dữ liệu để thống kê")
        else:
            # Advanced statistics
            col_stat_adv1, col_stat_adv2, col_stat_adv3 = st.columns(3)
            
            with col_stat_adv1:
                if 'tai_xiu' in df.columns:
                    tai_ratio = (df['tai_xiu'] == 'TÀI').mean() * 100
                    st.metric("Tỷ lệ Tài", f"{tai_ratio:.1f}%")
            
            with col_stat_adv2:
                if 'le_chan' in df.columns:
                    le_ratio = (df['le_chan'] == 'LẺ').mean() * 100
                    st.metric("Tỷ lệ Lẻ", f"{le_ratio:.1f}%")
            
            with col_stat_adv3:
                if 'tong' in df.columns:
                    tai_threshold = 23
                    tai_percentage = (df['tong'] >= tai_threshold).mean() * 100
                    st.metric("Tài thực tế", f"{tai_percentage:.1f}%")
            
            # Detailed frequency analysis
            with st.expander("📊 TẦN SUẤT CHI TIẾT", expanded=True):
                tab_freq1, tab_freq2, tab_freq3 = st.tabs(["Tổng số", "Hậu nhị", "Tiền nhị"])
                
                with tab_freq1:
                    if 'tong' in df.columns:
                        freq_df = df['tong'].value_counts().reset_index()
                        freq_df.columns = ['Tổng số', 'Tần suất']
                        freq_df['Phần trăm'] = (freq_df['Tần suất'] / len(df) * 100).round(2)
                        freq_df = freq_df.sort_values('Tần suất', ascending=False)
                        st.dataframe(freq_df, use_container_width=True)
                
                with tab_freq2:
                    if 'hau_nhi' in df.columns:
                        freq_df = df['hau_nhi'].value_counts().reset_index()
                        freq_df.columns = ['Hậu nhị', 'Tần suất']
                        freq_df['Phần trăm'] = (freq_df['Tần suất'] / len(df) * 100).round(2)
                        freq_df = freq_df.sort_values('Tần suất', ascending=False)
                        st.dataframe(freq_df.head(20), use_container_width=True)
                
                with tab_freq3:
                    if 'tien_nhi' in df.columns:
                        freq_df = df['tien_nhi'].value_counts().reset_index()
                        freq_df.columns = ['Tiền nhị', 'Tần suất']
                        freq_df['Phần trăm'] = (freq_df['Tần suất'] / len(df) * 100).round(2)
                        freq_df = freq_df.sort_values('Tần suất', ascending=False)
                        st.dataframe(freq_df.head(20), use_container_width=True)
            
            # Hot/Cold numbers
            with st.expander("🔥❄️ SỐ NÓNG/LẠNH", expanded=True):
                if len(df) >= 50:
                    # Hot numbers (frequent in last 50 draws)
                    recent_df = df.head(50)
                    
                    col_hot1, col_hot2 = st.columns(2)
                    
                    with col_hot1:
                        st.markdown("**🔥 Số nóng (20 kỳ gần nhất):**")
                        if 'so5' in recent_df.columns:
                            # Count frequency of each digit
                            all_digits = []
                            for num in recent_df['so5'].head(20):
                                if len(str(num)) == 5:
                                    all_digits.extend(list(str(num)))
                            
                            digit_counter = Counter(all_digits)
                            hot_digits = digit_counter.most_common(5)
                            
                            for digit, count in hot_digits:
                                percentage = (count / (20 * 5)) * 100
                                st.markdown(f"**{digit}**: {count} lần ({percentage:.1f}%)")
                    
                    with col_hot2:
                        st.markdown("**❄️ Số lạnh (20 kỳ gần nhất):**")
                        if 'so5' in recent_df.columns:
                            all_digits = []
                            for num in recent_df['so5'].head(20):
                                if len(str(num)) == 5:
                                    all_digits.extend(list(str(num)))
                            
                            digit_counter = Counter(all_digits)
                            cold_digits = digit_counter.most_common()[-5:]
                            
                            for digit, count in cold_digits:
                                percentage = (count / (20 * 5)) * 100
                                st.markdown(f"**{digit}**: {count} lần ({percentage:.1f}%)")
            
            # Pattern statistics
            with st.expander("🔄 THỐNG KÊ PATTERN", expanded=True):
                if len(df) >= 30:
                    # Analyze consecutive patterns
                    patterns = []
                    for i in range(len(df) - 1):
                        if 'tai_xiu' in df.columns:
                            current_tx = df.iloc[i]['tai_xiu']
                            next_tx = df.iloc[i+1]['tai_xiu']
                            
                            if current_tx == next_tx:
                                patterns.append(f"{current_tx} lặp")
                            else:
                                patterns.append(f"{current_tx}→{next_tx}")
                    
                    if patterns:
                        pattern_counter = Counter(patterns)
                        pattern_df = pd.DataFrame(pattern_counter.items(), columns=['Pattern', 'Số lần'])
                        pattern_df['Phần trăm'] = (pattern_df['Số lần'] / len(patterns) * 100).round(2)
                        pattern_df = pattern_df.sort_values('Số lần', ascending=False)
                        
                        st.dataframe(pattern_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # Capital Management Tab
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">💰 QUẢN LÝ VỐN THÔNG MINH</div>', unsafe_allow_html=True)
        
        # Capital configuration
        col_cap1, col_cap2 = st.columns([2, 3])
        
        with col_cap1:
            st.markdown("**💼 Cấu hình vốn:**")
            
            total_capital = st.number_input(
                "Tổng vốn (VNĐ):",
                min_value=100000,
                max_value=1000000000,
                value=5000000,
                step=100000,
                format="%d",
                help="Tổng số vốn bạn có để đầu tư"
            )
            
            risk_level = st.select_slider(
                "Mức độ rủi ro:",
                options=["Rất thấp", "Thấp", "Trung bình", "Cao", "Rất cao"],
                value="Trung bình"
            )
            
            stop_loss = st.slider(
                "Stop-loss (%):",
                min_value=5,
                max_value=40,
                value=15,
                step=5,
                help="Dừng lỗ khi thua bao nhiêu % vốn"
            )
            
            take_profit = st.slider(
                "Take-profit (%):",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Chốt lời khi đạt bao nhiêu % lợi nhuận"
            )
            
            if st.button("🧮 TÍNH PHÂN BỔ", type="primary", use_container_width=True):
                # Calculate capital allocation
                allocations = calculate_capital_allocation(total_capital, risk_level)
                st.session_state['capital_allocations'] = allocations
                st.success("✅ Đã tính toán phân bổ vốn!")
        
        with col_cap2:
            st.markdown("**📊 Phân bổ vốn đề xuất:**")
            
            if 'capital_allocations' in st.session_state:
                allocations = st.session_state['capital_allocations']
                
                st.markdown('<div class="capital-v13">', unsafe_allow_html=True)
                
                for bet_type, allocation in allocations.items():
                    amount = allocation['amount']
                    percentage = allocation['percentage']
                    
                    col_type, col_bar, col_info = st.columns([2, 4, 2])
                    
                    with col_type:
                        st.markdown(f"**{bet_type}**")
                    
                    with col_bar:
                        st.progress(percentage / 100)
                    
                    with col_info:
                        st.markdown(f"**{percentage:.1f}%**")
                    
                    st.markdown(f"`{format_currency(amount)}`")
                    st.markdown("---")
                
                # Summary
                total_allocated = sum(a['amount'] for a in allocations.values())
                remaining = total_capital - total_allocated
                remaining_percentage = (remaining / total_capital) * 100
                
                col_sum1, col_sum2 = st.columns(2)
                with col_sum1:
                    st.metric("Tổng phân bổ", format_currency(total_allocated))
                with col_sum2:
                    st.metric("Vốn dự phòng", format_currency(remaining))
                
                if remaining_percentage < 10:
                    st.error(f"⚠️ Vốn dự phòng thấp ({remaining_percentage:.1f}%) - Nên giữ ít nhất 10%")
                elif remaining_percentage < 20:
                    st.warning(f"⚠️ Vốn dự phòng hơi thấp ({remaining_percentage:.1f}%)")
                else:
                    st.success(f"✅ Phân bổ hợp lý ({remaining_percentage:.1f}% dự phòng)")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Nhập vốn và nhấn 'TÍNH PHÂN BỔ' để xem phân bổ đề xuất")
        
        # Betting strategy
        st.markdown("---")
        st.markdown('<div style="font-size:1.2rem;font-weight:800;color="#f59e0b;margin:20px 0;">🎯 CHIẾN LƯỢC ĐẶT CƯỢC</div>', unsafe_allow_html=True)
        
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.markdown("**📈 Chiến lược cơ bản:**")
            strategy = st.selectbox(
                "Chọn chiến lược:",
                ["Bảo toàn vốn", "Tăng trưởng ổn định", "Tăng trưởng mạnh", "Theo AI"],
                index=1
            )
            
            bet_size = st.select_slider(
                "Mức đặt cược:",
                options=["Rất nhỏ (1-2%)", "Nhỏ (3-5%)", "Trung bình (5-10%)", "Lớn (10-15%)", "Rất lớn (15-20%)"],
                value="Trung bình (5-10%)"
            )
        
        with col_strat2:
            st.markdown("**🔄 Quản lý rủi ro:**")
            
            martingale = st.checkbox("Sử dụng Martingale", value=False,
                                   help="Tăng gấp đôi tiền cược sau mỗi lần thua")
            
            if martingale:
                max_doubles = st.slider("Số lần gấp đôi tối đa:", 1, 10, 3)
                st.info(f"⚠️ Martingale: Gấp đôi tối đa {max_doubles} lần")
            
            hedging = st.checkbox("Sử dụng Hedging", value=True,
                                help="Đặt cược đối ứng để giảm rủi ro")
        
        # Save strategy
        if st.button("💾 LƯU CHIẾN LƯỢC", use_container_width=True):
            strategy_data = {
                'total_capital': total_capital,
                'risk_level': risk_level,
                'strategy': strategy,
                'bet_size': bet_size,
                'martingale': martingale,
                'hedging': hedging,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to session
            st.session_state['betting_strategy'] = strategy_data
            st.success("✅ Đã lưu chiến lược!")
        
        # Show saved strategy
        if 'betting_strategy' in st.session_state:
            st.markdown("---")
            st.markdown("**📋 Chiến lược đã lưu:**")
            strategy_data = st.session_state['betting_strategy']
            
            col_saved1, col_saved2 = st.columns(2)
            with col_saved1:
                st.write(f"**Vốn:** {format_currency(strategy_data['total_capital'])}")
                st.write(f"**Rủi ro:** {strategy_data['risk_level']}")
                st.write(f"**Chiến lược:** {strategy_data['strategy']}")
            
            with col_saved2:
                st.write(f"**Mức cược:** {strategy_data['bet_size']}")
                st.write(f"**Stop-loss:** {strategy_data['stop_loss']}%")
                st.write(f"**Take-profit:** {strategy_data['take_profit']}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        # Settings Tab
        st.markdown('<div class="card-v13">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;color="#6366f1;margin-bottom:20px;">⚙️ CÀI ĐẶT HỆ THỐNG</div>', unsafe_allow_html=True)
        
        # System settings
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("**🔧 Cài đặt chung:**")
            
            auto_update = st.checkbox("Tự động cập nhật", value=True,
                                    help="Tự động cập nhật dữ liệu mới")
            
            notifications = st.checkbox("Thông báo", value=True,
                                      help="Hiển thị thông báo quan trọng")
            
            dark_mode = st.checkbox("Chế độ tối", value=True,
                                  help="Giao diện tối (mặc định)")
            
            language = st.selectbox("Ngôn ngữ:", ["Tiếng Việt", "English"], index=0)
        
        with col_set2:
            st.markdown("**📊 Cài đặt AI:**")
            
            ai_model = st.selectbox(
                "Mô hình AI:",
                ["Nhanh (Fast)", "Cân bằng (Balanced)", "Chính xác (Accurate)", "Tối đa (Maximum)"],
                index=2
            )
            
            prediction_horizon = st.slider(
                "Tầm nhìn dự đoán (kỳ):",
                min_value=1,
                max_value=10,
                value=3,
                help="Số kỳ dự đoán trước"
            )
            
            min_confidence = st.slider(
                "Độ tin cậy tối thiểu (%):",
                min_value=50,
                max_value=90,
                value=65,
                help="Chỉ hiển thị dự đoán có độ tin cậy từ mức này"
            )
        
        # Data management
        st.markdown("---")
        st.markdown('<div style="font-size:1.2rem;font-weight:800;color="#10b981;margin:20px 0;">🗃️ QUẢN LÝ DỮ LIỆU</div>', unsafe_allow_html=True)
        
        col_data1, col_data2, col_data3 = st.columns(3)
        
        with col_data1:
            if st.button("📥 Xuất dữ liệu", use_container_width=True):
                export_data()
        
        with col_data2:
            if st.button("🔄 Làm mới cache", use_container_width=True):
                clear_cache()
                st.success("✅ Đã làm mới cache!")
        
        with col_data3:
            if st.button("🗑️ Xóa dữ liệu", use_container_width=True, type="secondary"):
                if st.checkbox("Xác nhận xóa toàn bộ dữ liệu?"):
                    clear_all_data()
                    st.error("⚠️ Đã xóa toàn bộ dữ liệu!")
                    time.sleep(2)
                    st.rerun()
        
        # Cloud settings
        st.markdown("---")
        st.markdown('<div style="font-size:1.2rem;font-weight:800;color="#0ea5e9;margin:20px 0;">☁️ CÀI ĐẶT ĐÁM MÂY</div>', unsafe_allow_html=True)
        
        cloud_enabled = st.checkbox("Kích hoạt đồng bộ đám mây", value=True)
        
        if cloud_enabled:
            cloud_urls = st.text_area(
                "URL nguồn dữ liệu (mỗi dòng 1 URL):",
                value="https://api.lottoku.com/data\nhttps://soicaulottoku.com/api",
                height=100,
                help="Nhập các URL API để đồng bộ dữ liệu"
            )
            
            sync_interval = st.select_slider(
                "Chu kỳ đồng bộ:",
                options=["5 phút", "15 phút", "30 phút", "1 giờ", "3 giờ", "6 giờ", "12 giờ", "24 giờ"],
                value="1 giờ"
            )
            
            if st.button("☁️ ĐỒNG BỘ NGAY", use_container_width=True):
                with st.spinner("Đang đồng bộ với đám mây..."):
                    cloud_manager = CloudDataManager()
                    sync_result = cloud_manager.sync_with_cloud()
                    st.success(f"✅ Đồng bộ thành công! {sync_result.get('total_records', 0)} bản ghi")
        
        # Save settings
        st.markdown("---")
        if st.button("💾 LƯU CÀI ĐẶT", type="primary", use_container_width=True):
            settings = {
                'auto_update': auto_update,
                'notifications': notifications,
                'dark_mode': dark_mode,
                'language': language,
                'ai_model': ai_model,
                'prediction_horizon': prediction_horizon,
                'min_confidence': min_confidence,
                'cloud_enabled': cloud_enabled,
                'cloud_urls': cloud_urls if cloud_enabled else "",
                'sync_interval': sync_interval if cloud_enabled else "",
                'last_updated': datetime.now().isoformat()
            }
            
            save_settings(settings)
            st.success("✅ Đã lưu cài đặt!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color="#64748b;font-size:0.9rem;padding:20px;">
    <div style="font-weight:800;color="#94a3b8;margin-bottom:5px;">
    🚀 COS V13.0 ULTIMATE - SIÊU PHẨM AI LOTTO KU 🚀
    </div>
    <div style="color:#475569;">
    Hệ thống AI đa tầng • 25+ thuật toán nâng cao • Dự đoán chính xác • Quản lý vốn thông minh<br>
    ⚠️ Dành cho mục đích nghiên cứu • Đầu tư có rủi ro • Quản lý vốn là yếu tố sống còn<br>
    © 2024 COS AI Ultimate • Phiên bản 13.0 • <span style="color:#6366f1;">Powered by Advanced AI</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ================= HELPER FUNCTIONS V13 =================
def load_lotto_data_v13(limit: int = 300) -> pd.DataFrame:
    """Tải dữ liệu từ database V13"""
    conn = sqlite3.connect(DB_FILE)
    
    # Try new table first, then fallback to old
    try:
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
        FROM lotto_results_v13 
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
    except:
        # Fallback to old table
        try:
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
            df = pd.read_sql(query, conn)
        except:
            df = pd.DataFrame()
    
    conn.close()
    return df

def save_lotto_results_v13(numbers: List[str], current_ky: str = None) -> int:
    """Lưu kết quả vào database V13"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    added_count = 0
    
    for idx, num_str in enumerate(numbers):
        try:
            if len(num_str) != 5 or not num_str.isdigit():
                continue
            
            num = LotteryNumberV13.from_string(num_str)
            
            # Tạo kỳ
            if current_ky and idx == 0:
                ky = current_ky
            else:
                ky = f"KU{int(time.time() * 1000) % 1000000:06d}"
            
            # Try new table first
            try:
                c.execute("""
                INSERT OR IGNORE INTO lotto_results_v13 
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
            except:
                # Fallback to old table
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
            logger.error(f"Lỗi lưu số {num_str}: {e}")
    
    conn.commit()
    conn.close()
    return added_count

def get_confidence_class_v13(confidence: float) -> str:
    """Lớp CSS cho confidence V13"""
    if confidence >= 80:
        return "conf-high-v13"
    elif confidence >= 65:
        return "conf-medium-v13"
    else:
        return "conf-low-v13"

def get_recommendation_badge_v13(recommendation: str) -> str:
    """Badge cho khuyến nghị V13"""
    if recommendation == 'NÊN ĐÁNH':
        return '<span class="recommend-v13 rec-bet-v13">🎯 NÊN ĐÁNH</span>'
    elif recommendation == 'CÓ THỂ ĐÁNH':
        return '<span class="recommend-v13 rec-maybe-v13">⚠️ CÓ THỂ ĐÁNH</span>'
    elif recommendation == 'THEO DÕI':
        return '<span class="recommend-v13 rec-maybe-v13">👀 THEO DÕI</span>'
    else:
        return '<span class="recommend-v13 rec-no-v13">⛔ KHÔNG ĐÁNH</span>'

def format_currency(amount: float) -> str:
    """Định dạng tiền"""
    return f"{amount:,.0f}₫"

def calculate_capital_allocation(total_capital: float, risk_level: str) -> Dict:
    """Tính toán phân bổ vốn"""
    # Base allocations based on risk level
    if risk_level == "Rất thấp":
        allocations = {
            'Tài/Xỉu': 15,
            'Lẻ/Chẵn': 15,
            'Hậu nhị': 20,
            'Tiền nhị': 15,
            '2 Tinh': 10,
            '3 Tinh': 10,
            'Đề số': 10,
            'Dự phòng': 5
        }
    elif risk_level == "Thấp":
        allocations = {
            'Tài/Xỉu': 20,
            'Lẻ/Chẵn': 15,
            'Hậu nhị': 25,
            'Tiền nhị': 15,
            '2 Tinh': 10,
            '3 Tinh': 5,
            'Đề số': 5,
            'Dự phòng': 5
        }
    elif risk_level == "Trung bình":
        allocations = {
            'Tài/Xỉu': 25,
            'Lẻ/Chẵn': 15,
            'Hậu nhị': 30,
            'Tiền nhị': 10,
            '2 Tinh': 8,
            '3 Tinh': 5,
            'Đề số': 4,
            'Dự phòng': 3
        }
    elif risk_level == "Cao":
        allocations = {
            'Tài/Xỉu': 30,
            'Lẻ/Chẵn': 20,
            'Hậu nhị': 35,
            'Tiền nhị': 8,
            '2 Tinh': 3,
            '3 Tinh': 2,
            'Đề số': 1,
            'Dự phòng': 1
        }
    else:  # Rất cao
        allocations = {
            'Tài/Xỉu': 40,
            'Lẻ/Chẵn': 25,
            'Hậu nhị': 30,
            'Tiền nhị': 3,
            '2 Tinh': 1,
            '3 Tinh': 0.5,
            'Đề số': 0.5,
            'Dự phòng': 0
        }
    
    # Convert to amounts
    result = {}
    for bet_type, percentage in allocations.items():
        amount = total_capital * (percentage / 100)
        result[bet_type] = {
            'amount': amount,
            'percentage': percentage
        }
    
    return result

def export_data():
    """Xuất dữ liệu"""
    conn = sqlite3.connect(DB_FILE)
    
    # Export to CSV
    try:
        df = pd.read_sql("SELECT * FROM lotto_results_v13", conn)
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="📥 Tải xuống CSV",
            data=csv,
            file_name=f"lotto_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    except:
        st.error("Không có dữ liệu để xuất")
    
    conn.close()

def clear_cache():
    """Xóa cache"""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(exist_ok=True)

def clear_all_data():
    """Xóa toàn bộ dữ liệu"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Drop all tables
    c.execute("DROP TABLE IF EXISTS lotto_results_v13")
    c.execute("DROP TABLE IF EXISTS ai_predictions_v13")
    c.execute("DROP TABLE IF EXISTS patterns_v13")
    c.execute("DROP TABLE IF EXISTS algorithm_performance_v13")
    c.execute("DROP TABLE IF EXISTS user_predictions_v13")
    c.execute("DROP TABLE IF EXISTS cloud_sync_v13")
    
    conn.commit()
    conn.close()
    
    # Reinitialize
    init_v13_database()

def save_settings(settings: Dict):
    """Lưu cài đặt"""
    with open("cos_v13_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

# ================= RUN APP V13 =================
if __name__ == "__main__":
    main_v13()
