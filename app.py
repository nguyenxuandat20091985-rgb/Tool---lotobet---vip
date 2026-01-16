# ================= COS V13.1 LITE ULTIMATE - LOTTO KU/LOTOBET =================
# Complete Tool with All Betting Types

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
import warnings
import json
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict, deque, OrderedDict
import random
import math
from itertools import combinations, permutations
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging

warnings.filterwarnings('ignore')

# ================= AI LIBRARIES =================
try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        VotingClassifier, StackingClassifier, AdaBoostClassifier
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    
    AI_LIBS_AVAILABLE = True
except ImportError:
    AI_LIBS_AVAILABLE = False
    st.warning("⚠️ Thiếu thư viện scikit-learn. Một số tính năng AI có thể bị giới hạn.")

# ================= CONFIG =================
st.set_page_config(
    page_title="COS V13.1 LITE - LOTTO KU/LOTOBET",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/lottoku-ai',
        'Report a bug': "https://github.com/lottoku-ai/issues",
        'About': "# COS V13.1 LITE - Công cụ soi cầu KU/Lotobet đầy đủ"
    }
)

# ================= ENHANCED CSS =================
st.markdown("""
<style>
    /* Main Theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Ultimate Header */
    .header-ultimate {
        background: linear-gradient(90deg, 
            rgba(102, 126, 234, 0.9) 0%, 
            rgba(118, 75, 162, 0.9) 50%,
            rgba(237, 100, 166, 0.9) 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 25px;
        margin: 20px 0 30px 0;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-ultimate::before {
        content: '🎰 COS V13.1 LITE ULTIMATE 🎰';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 60px;
        opacity: 0.05;
        font-weight: 900;
        white-space: nowrap;
    }
    
    /* Prediction Grid */
    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #ed64a6);
    }
    
    /* Number Display */
    .number-display {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin: 20px 0;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    
    .confidence-high {
        background: linear-gradient(90deg, #10b981, #34d399);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #ef4444, #f87171);
        color: white;
    }
    
    /* Recommendation */
    .recommendation {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 1rem;
        margin: 15px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .recommend-bet {
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        border: 2px solid #065f46;
    }
    
    .recommend-maybe {
        background: linear-gradient(90deg, #d97706, #f59e0b);
        color: white;
        border: 2px solid #92400e;
    }
    
    .recommend-no {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        border: 2px solid #7f1d1d;
    }
    
    /* Real-time Counter */
    .counter-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .counter-time {
        font-size: 2.5rem;
        font-weight: 900;
        color: white;
        margin: 10px 0;
    }
    
    /* Pattern Badge */
    .pattern-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 5px;
        background: #f3f4f6;
        border: 1px solid #d1d5db;
    }
    
    .pattern-hot {
        background: linear-gradient(90deg, #fee2e2, #fecaca);
        border-color: #fca5a5;
        color: #dc2626;
    }
    
    .pattern-cold {
        background: linear-gradient(90deg, #dbeafe, #bfdbfe);
        border-color: #93c5fd;
        color: #1d4ed8;
    }
    
    .pattern-alive {
        background: linear-gradient(90deg, #dcfce7, #bbf7d0);
        border-color: #86efac;
        color: #15803d;
    }
    
    /* Table Styling */
    .data-table {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 15px;
        font-weight: 700;
        padding: 14px 28px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric Cards */
    .stMetric {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #ed64a6);
        border-radius: 10px;
    }
    
    /* Responsive */
    @media (max-width: 1200px) {
        .number-display { font-size: 2.8rem; }
        .header-ultimate { padding: 20px 30px; }
        .header-ultimate::before { font-size: 40px; }
    }
    
    @media (max-width: 768px) {
        .number-display { font-size: 2.2rem; }
        .header-ultimate { padding: 15px 20px; }
        .header-ultimate::before { font-size: 30px; }
        .prediction-card { padding: 20px; }
        .counter-time { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================= DATABASE =================
DB_FILE = "cos_v13_1_lite.db"

def init_database():
    """Khởi tạo database nâng cao"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Main results table
    c.execute("""
    CREATE TABLE IF NOT EXISTS lotto_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE NOT NULL,
        chuc_ngan INTEGER NOT NULL,
        ngan INTEGER NOT NULL,
        tram INTEGER NOT NULL,
        chuc INTEGER NOT NULL,
        don_vi INTEGER NOT NULL,
        full_number TEXT NOT NULL,
        tien_nhi TEXT NOT NULL,
        hau_nhi TEXT NOT NULL,
        tong INTEGER NOT NULL,
        tai_xiu TEXT NOT NULL,
        le_chan TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        verified INTEGER DEFAULT 0
    )
    """)
    
    # Predictions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS ai_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT NOT NULL,
        prediction_type TEXT NOT NULL,
        predicted_value TEXT NOT NULL,
        confidence REAL NOT NULL,
        recommendation TEXT NOT NULL,
        actual_result TEXT,
        is_correct INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Patterns table
    c.execute("""
    CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_data TEXT NOT NULL,
        start_ky TEXT,
        end_ky TEXT,
        strength REAL NOT NULL,
        confidence REAL NOT NULL,
        prediction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # User bets table
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        ky TEXT NOT NULL,
        bet_type TEXT NOT NULL,
        predicted_value TEXT NOT NULL,
        stake_amount REAL,
        actual_result TEXT,
        profit_loss REAL,
        status TEXT DEFAULT 'pending',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_ky ON lotto_results(ky)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON lotto_results(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_type ON ai_predictions(prediction_type)")
    
    conn.commit()
    conn.close()

init_database()

# ================= ENUMS & DATA CLASSES =================
class BetType(Enum):
    """Loại cược theo KU/Lotobet"""
    FIVE_STAR = "5_tinh"           # 5 tinh
    FOUR_STAR = "hau_tu"           # Hậu tứ
    THREE_STAR_FRONT = "tien_tam"  # Tiền tam
    THREE_STAR_MID = "trung_tam"   # Trung tam
    THREE_STAR_BACK = "hau_tam"    # Hậu tam
    TWO_STAR_FRONT = "tien_nhi"    # Tiền nhị
    TWO_STAR_BACK = "hau_nhi"      # Hậu nhị
    ONE_POSITION = "1_hang_so"     # 1 hàng số
    TAI_XIU = "tai_xiu"            # Tài/Xỉu
    LE_CHAN = "le_chan"            # Lẻ/Chẵn
    TWO_COMBINATION = "2_tinh"     # 2 tinh
    THREE_COMBINATION = "3_tinh"   # 3 tinh
    VARIABLE = "khong_co_dinh"     # Không cố định

class PatternType(Enum):
    """Loại pattern"""
    BAMBI = "cau_bam"              # Cầu bầm
    STREAK = "cau_bet"             # Cầu bệt
    ALIVE = "cau_song"             # Cầu sống
    DEAD = "cau_chet"              # Cầu chết
    REVERSE = "cau_dao"            # Cầu đảo
    GAP = "cau_gap"                # Cầu gấp
    TREND = "cau_trend"            # Xu hướng
    REPEAT = "cau_lap"             # Lặp lại

@dataclass
class LotteryNumber:
    """Biểu diễn số lotto 5 chữ số"""
    chuc_ngan: int  # Chục ngàn
    ngan: int       # Ngàn
    tram: int       # Trăm
    chuc: int       # Chục
    don_vi: int     # Đơn vị
    
    def __post_init__(self):
        for attr in ['chuc_ngan', 'ngan', 'tram', 'chuc', 'don_vi']:
            value = getattr(self, attr)
            if not 0 <= value <= 9:
                raise ValueError(f"{attr} phải từ 0-9")
    
    @classmethod
    def from_string(cls, num_str: str):
        if len(num_str) != 5 or not num_str.isdigit():
            raise ValueError("Chuỗi phải có đúng 5 chữ số")
        return cls(*[int(d) for d in num_str])
    
    def to_string(self) -> str:
        return f"{self.chuc_ngan}{self.ngan}{self.tram}{self.chuc}{self.don_vi}"
    
    def get_tien_nhi(self) -> str:
        return f"{self.chuc_ngan}{self.ngan}"
    
    def get_hau_nhi(self) -> str:
        return f"{self.chuc}{self.don_vi}"
    
    def get_tien_tam(self) -> str:
        return f"{self.chuc_ngan}{self.ngan}{self.tram}"
    
    def get_trung_tam(self) -> str:
        return f"{self.ngan}{self.tram}{self.chuc}"
    
    def get_hau_tam(self) -> str:
        return f"{self.tram}{self.chuc}{self.don_vi}"
    
    def get_hau_tu(self) -> str:
        return f"{self.ngan}{self.tram}{self.chuc}{self.don_vi}"
    
    def get_tong(self) -> int:
        return sum([self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi])
    
    def is_tai(self) -> bool:
        return 23 <= self.get_tong() <= 45
    
    def is_xiu(self) -> bool:
        return 0 <= self.get_tong() <= 22
    
    def is_chan(self) -> bool:
        return self.get_tong() % 2 == 0
    
    def is_le(self) -> bool:
        return self.get_tong() % 2 == 1
    
    def get_all_positions(self) -> List[int]:
        return [self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi]
    
    def get_2tinh_combinations(self) -> List[Tuple[int, int]]:
        """Tất cả cặp 2 số"""
        digits = self.get_all_positions()
        return list(combinations(digits, 2))
    
    def get_3tinh_combinations(self) -> List[Tuple[int, int, int]]:
        """Tất cả bộ 3 số"""
        digits = self.get_all_positions()
        return list(combinations(digits, 3))

@dataclass
class PredictionResult:
    """Kết quả dự đoán"""
    bet_type: BetType
    predicted_value: Any
    confidence: float  # 0-100%
    recommendation: str  # NÊN ĐÁNH, CÓ THỂ ĐÁNH, KHÔNG ĐÁNH
    reasoning: str
    probability_dist: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'bet_type': self.bet_type.value,
            'predicted_value': str(self.predicted_value),
            'confidence': self.confidence,
            'recommendation': self.recommendation,
            'reasoning': self.reasoning,
            'probability_dist': self.probability_dist,
            'timestamp': self.timestamp.isoformat()
        }

# ================= REAL-TIME MONITOR =================
class RealTimeMonitor:
    """Monitor thời gian thực cho KU/Lotobet"""
    
    def __init__(self):
        self.current_ky = None
        self.next_draw = None
        self.draw_interval = 150  # 2.5 phút (150 giây) cho Lotto A
        self.maintenance_start = None
        self.maintenance_end = None
        
    def sync_time(self) -> Dict:
        """Đồng bộ thời gian thực"""
        now = datetime.now()
        
        # Tính kỳ hiện tại (dựa trên thời gian)
        base_date = now.strftime("%y%m%d")
        
        # Tính số kỳ từ 05:00 (GMT+8)
        base_time = now.replace(hour=5, minute=0, second=0, microsecond=0)
        if now < base_time:
            base_time = base_time - timedelta(days=1)
        
        seconds_since_base = (now - base_time).total_seconds()
        kythu = int(seconds_since_base // self.draw_interval) + 1
        
        self.current_ky = f"KUA{base_date}{kythu:03d}"
        
        # Tính thời gian quay tiếp theo
        next_seconds = kythu * self.draw_interval
        self.next_draw = base_time + timedelta(seconds=next_seconds)
        
        # Kiểm tra bảo trì (05:00-05:30 GMT+8)
        maintenance_start = now.replace(hour=5, minute=0, second=0, microsecond=0)
        maintenance_end = now.replace(hour=5, minute=30, second=0, microsecond=0)
        
        in_maintenance = maintenance_start <= now < maintenance_end
        
        return {
            'current_ky': self.current_ky,
            'next_draw': self.next_draw.strftime("%H:%M:%S"),
            'seconds_to_next': max(0, int((self.next_draw - now).total_seconds())),
            'in_maintenance': in_maintenance,
            'current_time': now.strftime("%H:%M:%S"),
            'draw_interval': f"{self.draw_interval//60}:{self.draw_interval%60:02d} phút"
        }

# ================= ADVANCED AI ENGINE =================
class LottoAdvancedAI:
    """AI nâng cao cho KU/Lotobet với đầy đủ tính năng"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data.copy()
        self.numbers = self._extract_numbers()
        self.cache = {}
        self.patterns = []
        
    def _extract_numbers(self) -> List[LotteryNumber]:
        """Trích xuất số từ dataframe"""
        numbers = []
        for _, row in self.df.iterrows():
            try:
                if 'full_number' in row and len(str(row['full_number'])) == 5:
                    num = LotteryNumber.from_string(str(row['full_number']))
                    numbers.append(num)
            except:
                continue
        return numbers
    
    def analyze_all_predictions(self) -> Dict:
        """Phân tích tất cả loại dự đoán"""
        results = {}
        
        # 0. Hàng số 5 tinh
        results['5_tinh'] = self._analyze_five_star()
        
        # 1. 2 TINH (2 cặp số)
        results['2_tinh'] = self._analyze_two_star()
        
        # 2. 3 TINH (3 cặp số)
        results['3_tinh'] = self._analyze_three_star()
        
        # 3. Đề số hậu nhị, tiền nhị
        results['de_so'] = self._analyze_de_so()
        
        # 4. Tài xỉu - Chẵn lẻ
        results['tai_xiu'] = self._analyze_tai_xiu()
        results['le_chan'] = self._analyze_le_chan()
        
        # 5. Pattern detection
        results['patterns'] = self._detect_patterns()
        
        # 6. AI phân tích chi tiết
        results['ai_analysis'] = self._generate_ai_analysis(results)
        
        return results
    
    def _analyze_five_star(self) -> Dict:
        """Phân tích 5 tinh - Dự đoán từng vị trí"""
        if len(self.numbers) < 30:
            return {'confidence': 40, 'predictions': []}
        
        predictions = []
        
        # Phân tích từng vị trí
        positions = ['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']
        recent_nums = self.numbers[:50]
        
        for idx, pos_name in enumerate(positions):
            # Lấy số ở vị trí này từ recent numbers
            pos_values = []
            for num in recent_nums:
                if idx == 0:
                    pos_values.append(num.chuc_ngan)
                elif idx == 1:
                    pos_values.append(num.ngan)
                elif idx == 2:
                    pos_values.append(num.tram)
                elif idx == 3:
                    pos_values.append(num.chuc)
                else:
                    pos_values.append(num.don_vi)
            
            # Tính tần suất
            freq = Counter(pos_values)
            total = sum(freq.values())
            
            # Top 3 số có tần suất cao nhất
            top_numbers = []
            for digit, count in freq.most_common(5):
                percentage = (count / total) * 100
                
                if percentage >= 15:
                    recommendation = "NÊN ĐÁNH"
                    strength = "high"
                elif percentage >= 10:
                    recommendation = "CÓ THỂ ĐÁNH"
                    strength = "medium"
                else:
                    recommendation = "THEO DÕI"
                    strength = "low"
                
                top_numbers.append({
                    'digit': digit,
                    'percentage': round(percentage, 2),
                    'frequency': count,
                    'recommendation': recommendation,
                    'strength': strength
                })
            
            predictions.append({
                'position': pos_name,
                'top_predictions': top_numbers[:3],
                'confidence': min(85, (freq.most_common(1)[0][1] / total * 100 * 1.5) if total > 0 else 50)
            })
        
        overall_confidence = np.mean([p['confidence'] for p in predictions])
        
        return {
            'predictions': predictions,
            'overall_confidence': round(overall_confidence, 2),
            'data_points': len(recent_nums)
        }
    
    def _analyze_two_star(self) -> Dict:
        """Phân tích 2 TINH - Dự đoán 2 cặp số"""
        if len(self.numbers) < 40:
            return {'confidence': 45, 'predictions': []}
        
        # Thu thập tất cả cặp 2 số từ 50 kỳ gần nhất
        all_pairs = []
        for num in self.numbers[:50]:
            pairs = num.get_2tinh_combinations()
            all_pairs.extend([f"{a}{b}" for a, b in pairs])
        
        # Phân tích tần suất
        freq = Counter(all_pairs)
        total_pairs = sum(freq.values())
        
        predictions = []
        for pair, count in freq.most_common(15):
            percentage = (count / total_pairs) * 100
            
            # Xác định khuyến nghị
            if percentage >= 2.0:
                recommendation = "NÊN ĐÁNH"
                color = "#10b981"
                strength = "high"
            elif percentage >= 1.2:
                recommendation = "CÓ THỂ ĐÁNH"
                color = "#f59e0b"
                strength = "medium"
            else:
                recommendation = "THEO DÕI"
                color = "#6b7280"
                strength = "low"
            
            predictions.append({
                'pair': pair,
                'percentage': round(percentage, 2),
                'frequency': count,
                'recommendation': recommendation,
                'color': color,
                'strength': strength,
                'rank': len(predictions) + 1
            })
        
        # Tính độ tin cậy tổng
        if predictions:
            top_percentage = predictions[0]['percentage']
            confidence = min(90, top_percentage * 15 + 30)
        else:
            confidence = 50
        
        return {
            'predictions': predictions[:10],  # Top 10 cặp
            'total_pairs_analyzed': total_pairs,
            'unique_pairs': len(freq),
            'confidence': round(confidence, 2),
            'most_common': [p['pair'] for p in predictions[:3]]
        }
    
    def _analyze_three_star(self) -> Dict:
        """Phân tích 3 TINH - Dự đoán 3 cặp số"""
        if len(self.numbers) < 50:
            return {'confidence': 40, 'predictions': []}
        
        # Thu thập tất cả bộ 3 số từ 60 kỳ gần nhất
        all_trios = []
        for num in self.numbers[:60]:
            trios = num.get_3tinh_combinations()
            all_trios.extend([f"{a}{b}{c}" for a, b, c in trios])
        
        # Phân tích tần suất
        freq = Counter(all_trios)
        total_trios = sum(freq.values())
        
        predictions = []
        for trio, count in freq.most_common(12):
            percentage = (count / total_trios) * 100
            
            # Xác định khuyến nghị
            if percentage >= 0.8:
                recommendation = "NÊN ĐÁNH"
                color = "#10b981"
                strength = "high"
            elif percentage >= 0.5:
                recommendation = "CÓ THỂ ĐÁNH"
                color = "#f59e0b"
                strength = "medium"
            else:
                recommendation = "THEO DÕI"
                color = "#6b7280"
                strength = "low"
            
            predictions.append({
                'trio': trio,
                'percentage': round(percentage, 2),
                'frequency': count,
                'recommendation': recommendation,
                'color': color,
                'strength': strength,
                'rank': len(predictions) + 1
            })
        
        # Tính độ tin cậy tổng
        if predictions:
            top_percentage = predictions[0]['percentage']
            confidence = min(85, top_percentage * 25 + 25)
        else:
            confidence = 45
        
        return {
            'predictions': predictions[:8],  # Top 8 bộ
            'total_trios_analyzed': total_trios,
            'unique_trios': len(freq),
            'confidence': round(confidence, 2),
            'most_common': [p['trio'] for p in predictions[:3]]
        }
    
    def _analyze_de_so(self) -> Dict:
        """Phân tích đề số - Tiền nhị & Hậu nhị"""
        if len(self.numbers) < 35:
            return {'confidence': 42, 'predictions': []}
        
        recent_nums = self.numbers[:50]
        
        # Tiền nhị
        tien_nhi_list = [int(num.get_tien_nhi()) for num in recent_nums]
        tien_nhi_freq = Counter(tien_nhi_list)
        
        # Hậu nhị
        hau_nhi_list = [int(num.get_hau_nhi()) for num in recent_nums]
        hau_nhi_freq = Counter(hau_nhi_list)
        
        predictions = []
        
        # Top 5 tiền nhị
        for value, count in tien_nhi_freq.most_common(8):
            percentage = (count / len(recent_nums)) * 100
            
            if percentage >= 4.0:
                recommendation = "NÊN ĐÁNH"
                strength = "high"
            elif percentage >= 2.5:
                recommendation = "CÓ THỂ ĐÁNH"
                strength = "medium"
            else:
                recommendation = "THEO DÕI"
                strength = "low"
            
            predictions.append({
                'type': 'Tiền nhị',
                'number': f"{value:02d}",
                'percentage': round(percentage, 2),
                'frequency': count,
                'recommendation': recommendation,
                'strength': strength
            })
        
        # Top 5 hậu nhị
        for value, count in hau_nhi_freq.most_common(8):
            percentage = (count / len(recent_nums)) * 100
            
            if percentage >= 4.0:
                recommendation = "NÊN ĐÁNH"
                strength = "high"
            elif percentage >= 2.5:
                recommendation = "CÓ THỂ ĐÁNH"
                strength = "medium"
            else:
                recommendation = "THEO DÕI"
                strength = "low"
            
            predictions.append({
                'type': 'Hậu nhị',
                'number': f"{value:02d}",
                'percentage': round(percentage, 2),
                'frequency': count,
                'recommendation': recommendation,
                'strength': strength
            })
        
        # Sắp xếp theo phần trăm
        predictions.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Tính độ tin cậy
        if predictions:
            top_percentage = predictions[0]['percentage']
            confidence = min(88, top_percentage * 1.8 + 30)
        else:
            confidence = 50
        
        return {
            'predictions': predictions[:10],  # Top 10
            'confidence': round(confidence, 2),
            'total_analyzed': len(recent_nums)
        }
    
    def _analyze_tai_xiu(self) -> Dict:
        """Phân tích Tài/Xỉu"""
        if len(self.numbers) < 30:
            return {'confidence': 50, 'prediction': 'TÀI', 'recommendation': 'THEO DÕI'}
        
        recent_nums = self.numbers[:40]
        tai_count = sum(1 for num in recent_nums if num.is_tai())
        xiu_count = sum(1 for num in recent_nums if num.is_xiu())
        
        tai_percentage = (tai_count / len(recent_nums)) * 100
        xiu_percentage = (xiu_count / len(recent_nums)) * 100
        
        # Dự đoán dựa trên xu hướng
        if tai_percentage > 60:
            prediction = "TÀI"
            confidence = min(90, tai_percentage * 1.3)
            recommendation = "NÊN ĐÁNH"
            reasoning = f"Xu hướng Tài mạnh ({tai_percentage:.1f}% trong {len(recent_nums)} kỳ gần nhất)"
        elif xiu_percentage > 60:
            prediction = "XỈU"
            confidence = min(90, xiu_percentage * 1.3)
            recommendation = "NÊN ĐÁNH"
            reasoning = f"Xu hướng Xỉu mạnh ({xiu_percentage:.1f}% trong {len(recent_nums)} kỳ gần nhất)"
        elif tai_percentage > 55:
            prediction = "TÀI"
            confidence = tai_percentage * 1.2
            recommendation = "CÓ THỂ ĐÁNH"
            reasoning = f"Xu hướng Tài ({tai_percentage:.1f}%)"
        elif xiu_percentage > 55:
            prediction = "XỈU"
            confidence = xiu_percentage * 1.2
            recommendation = "CÓ THỂ ĐÁNH"
            reasoning = f"Xu hướng Xỉu ({xiu_percentage:.1f}%)"
        else:
            prediction = "TÀI" if tai_percentage > xiu_percentage else "XỈU"
            confidence = 50
            recommendation = "THEO DÕI"
            reasoning = f"Xu hướng chưa rõ ràng (Tài: {tai_percentage:.1f}%, Xỉu: {xiu_percentage:.1f}%)"
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'recommendation': recommendation,
            'reasoning': reasoning,
            'statistics': {
                'tai_percentage': round(tai_percentage, 2),
                'xiu_percentage': round(xiu_percentage, 2),
                'tai_count': tai_count,
                'xiu_count': xiu_count,
                'period_analyzed': len(recent_nums)
            }
        }
    
    def _analyze_le_chan(self) -> Dict:
        """Phân tích Lẻ/Chẵn"""
        if len(self.numbers) < 30:
            return {'confidence': 50, 'prediction': 'LẺ', 'recommendation': 'THEO DÕI'}
        
        recent_nums = self.numbers[:40]
        le_count = sum(1 for num in recent_nums if num.is_le())
        chan_count = sum(1 for num in recent_nums if num.is_chan())
        
        le_percentage = (le_count / len(recent_nums)) * 100
        chan_percentage = (chan_count / len(recent_nums)) * 100
        
        # Dự đoán dựa trên xu hướng
        if le_percentage > 60:
            prediction = "LẺ"
            confidence = min(90, le_percentage * 1.3)
            recommendation = "NÊN ĐÁNH"
            reasoning = f"Xu hướng Lẻ mạnh ({le_percentage:.1f}% trong {len(recent_nums)} kỳ gần nhất)"
        elif chan_percentage > 60:
            prediction = "CHẴN"
            confidence = min(90, chan_percentage * 1.3)
            recommendation = "NÊN ĐÁNH"
            reasoning = f"Xu hướng Chẵn mạnh ({chan_percentage:.1f}% trong {len(recent_nums)} kỳ gần nhất)"
        elif le_percentage > 55:
            prediction = "LẺ"
            confidence = le_percentage * 1.2
            recommendation = "CÓ THỂ ĐÁNH"
            reasoning = f"Xu hướng Lẻ ({le_percentage:.1f}%)"
        elif chan_percentage > 55:
            prediction = "CHẴN"
            confidence = chan_percentage * 1.2
            recommendation = "CÓ THỂ ĐÁNH"
            reasoning = f"Xu hướng Chẵn ({chan_percentage:.1f}%)"
        else:
            prediction = "LẺ" if le_percentage > chan_percentage else "CHẴN"
            confidence = 50
            recommendation = "THEO DÕI"
            reasoning = f"Xu hướng chưa rõ ràng (Lẻ: {le_percentage:.1f}%, Chẵn: {chan_percentage:.1f}%)"
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'recommendation': recommendation,
            'reasoning': reasoning,
            'statistics': {
                'le_percentage': round(le_percentage, 2),
                'chan_percentage': round(chan_percentage, 2),
                'le_count': le_count,
                'chan_count': chan_count,
                'period_analyzed': len(recent_nums)
            }
        }
    
    def _detect_patterns(self) -> Dict:
        """Phát hiện các pattern đặc biệt"""
        if len(self.numbers) < 50:
            return {'patterns': [], 'confidence': 40}
        
        patterns = []
        recent_nums_str = [num.to_string() for num in self.numbers[:50]]
        recent_totals = [num.get_tong() for num in self.numbers[:50]]
        
        # 1. Cầu bệt (số lặp lại)
        for i in range(len(recent_nums_str) - 1):
            if recent_nums_str[i] == recent_nums_str[i + 1]:
                patterns.append({
                    'type': 'Cầu bệt',
                    'description': f"Số {recent_nums_str[i]} lặp lại liên tiếp",
                    'strength': 85,
                    'recommendation': 'CẢNH BÁO',
                    'numbers': [recent_nums_str[i], recent_nums_str[i + 1]],
                    'position': i
                })
        
        # 2. Cầu sống (xu hướng tăng/giảm rõ ràng)
        if len(recent_totals) >= 10:
            # Kiểm tra xu hướng tăng
            increasing = all(recent_totals[i] < recent_totals[i + 1] for i in range(5))
            decreasing = all(recent_totals[i] > recent_totals[i + 1] for i in range(5))
            
            if increasing:
                patterns.append({
                    'type': 'Cầu sống (tăng)',
                    'description': "Tổng số liên tục tăng 5 kỳ liên tiếp",
                    'strength': 75,
                    'recommendation': 'THEO DÕI',
                    'trend': 'increasing',
                    'values': recent_totals[:6]
                })
            elif decreasing:
                patterns.append({
                    'type': 'Cầu sống (giảm)',
                    'description': "Tổng số liên tục giảm 5 kỳ liên tiếp",
                    'strength': 75,
                    'recommendation': 'THEO DÕI',
                    'trend': 'decreasing',
                    'values': recent_totals[:6]
                })
        
        # 3. Cầu chết (pattern kết thúc)
        for i in range(len(recent_nums_str) - 3):
            # Kiểm tra pattern lặp rồi dừng
            if (recent_nums_str[i] == recent_nums_str[i + 1] and 
                recent_nums_str[i + 2] != recent_nums_str[i]):
                patterns.append({
                    'type': 'Cầu chết',
                    'description': f"Số {recent_nums_str[i]} lặp 2 lần rồi dừng",
                    'strength': 70,
                    'recommendation': 'TRÁNH',
                    'numbers': recent_nums_str[i:i+3]
                })
        
        # 4. Cầu đảo (số đảo ngược)
        for i in range(len(recent_nums_str) - 1):
            if recent_nums_str[i] == recent_nums_str[i + 1][::-1]:
                patterns.append({
                    'type': 'Cầu đảo',
                    'description': f"Số {recent_nums_str[i]} đảo ngược thành {recent_nums_str[i + 1]}",
                    'strength': 65,
                    'recommendation': 'THEO DÕI',
                    'numbers': [recent_nums_str[i], recent_nums_str[i + 1]]
                })
        
        # Sắp xếp theo strength
        patterns.sort(key=lambda x: x['strength'], reverse=True)
        
        confidence = min(80, len(patterns) * 5 + 30) if patterns else 45
        
        return {
            'patterns': patterns[:10],  # Top 10 patterns
            'confidence': confidence,
            'total_patterns': len(patterns)
        }
    
    def _generate_ai_analysis(self, predictions: Dict) -> Dict:
        """Tạo phân tích AI tổng hợp"""
        analysis = {
            'summary': {},
            'recommendations': [],
            'risk_assessment': {},
            'next_predictions': {}
        }
        
        # Tổng hợp độ tin cậy
        confidences = []
        for key, result in predictions.items():
            if isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        analysis['summary']['avg_confidence'] = round(np.mean(confidences), 2) if confidences else 50
        analysis['summary']['high_confidence_predictions'] = sum(1 for c in confidences if c >= 70)
        analysis['summary']['total_predictions'] = len(predictions)
        
        # Tạo khuyến nghị tổng hợp
        strong_recommendations = []
        
        # Kiểm tra Tài/Xỉu
        if 'tai_xiu' in predictions and predictions['tai_xiu'].get('recommendation') == 'NÊN ĐÁNH':
            strong_recommendations.append({
                'type': 'Tài/Xỉu',
                'prediction': predictions['tai_xiu']['prediction'],
                'confidence': predictions['tai_xiu']['confidence']
            })
        
        # Kiểm tra Lẻ/Chẵn
        if 'le_chan' in predictions and predictions['le_chan'].get('recommendation') == 'NÊN ĐÁNH':
            strong_recommendations.append({
                'type': 'Lẻ/Chẵn',
                'prediction': predictions['le_chan']['prediction'],
                'confidence': predictions['le_chan']['confidence']
            })
        
        # Kiểm tra 2 TINH
        if '2_tinh' in predictions and predictions['2_tinh'].get('confidence', 0) >= 70:
            top_2tinh = predictions['2_tinh'].get('predictions', [])[:3]
            if top_2tinh:
                strong_recommendations.append({
                    'type': '2 TINH',
                    'predictions': [p['pair'] for p in top_2tinh if p.get('recommendation') == 'NÊN ĐÁNH'],
                    'confidence': predictions['2_tinh']['confidence']
                })
        
        analysis['recommendations'] = strong_recommendations
        
        # Đánh giá rủi ro
        if analysis['summary']['avg_confidence'] >= 75:
            analysis['risk_assessment']['level'] = 'THẤP'
            analysis['risk_assessment']['color'] = '#10b981'
            analysis['risk_assessment']['reason'] = 'Độ tin cậy cao, dự đoán rõ ràng'
        elif analysis['summary']['avg_confidence'] >= 60:
            analysis['risk_assessment']['level'] = 'TRUNG BÌNH'
            analysis['risk_assessment']['color'] = '#f59e0b'
            analysis['risk_assessment']['reason'] = 'Độ tin cậy khá, cần thận trọng'
        else:
            analysis['risk_assessment']['level'] = 'CAO'
            analysis['risk_assessment']['color'] = '#ef4444'
            analysis['risk_assessment']['reason'] = 'Độ tin cậy thấp, cần quan sát thêm'
        
        # Dự đoán cho kỳ tiếp theo
        analysis['next_predictions'] = {
            'best_bets': strong_recommendations[:3] if strong_recommendations else [],
            'time_to_next': "2.5 phút",
            'suggested_stake': "3-5% vốn" if analysis['summary']['avg_confidence'] >= 70 else "1-2% vốn"
        }
        
        return analysis

# ================= DATA MANAGEMENT =================
def save_lotto_results(numbers: List[str], ky: str = None) -> int:
    """Lưu kết quả vào database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    added = 0
    
    for idx, num_str in enumerate(numbers):
        try:
            if len(num_str) != 5 or not num_str.isdigit():
                continue
            
            num = LotteryNumber.from_string(num_str)
            
            # Tạo kỳ nếu không có
            if ky and idx == 0:
                current_ky = ky
            else:
                current_ky = f"KUA{int(time.time() * 1000) % 1000000:06d}"
            
            c.execute("""
            INSERT OR IGNORE INTO lotto_results 
            (ky, chuc_ngan, ngan, tram, chuc, don_vi, full_number, tien_nhi, hau_nhi, tong, tai_xiu, le_chan)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                current_ky,
                num.chuc_ngan, num.ngan, num.tram, num.chuc, num.don_vi,
                num.to_string(),
                num.get_tien_nhi(),
                num.get_hau_nhi(),
                num.get_tong(),
                "TÀI" if num.is_tai() else "XỈU",
                "LẺ" if num.is_le() else "CHẴN"
            ))
            
            if c.rowcount > 0:
                added += 1
                
        except Exception as e:
            print(f"Lỗi lưu số {num_str}: {e}")
    
    conn.commit()
    conn.close()
    return added

def load_lotto_data(limit: int = 500) -> pd.DataFrame:
    """Tải dữ liệu từ database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f"""
        SELECT 
            ky,
            full_number,
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

def clear_old_data(days: int = 30):
    """Xóa dữ liệu cũ"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    c.execute("DELETE FROM lotto_results WHERE date(timestamp) < ?", (cutoff_date,))
    
    conn.commit()
    conn.close()

# ================= MAIN APP =================
def main():
    # Header
    st.markdown("""
    <div class="header-ultimate">
    <h1 style="font-size:2.5rem;margin-bottom:10px;">🎰 COS V13.1 LITE ULTIMATE</h1>
    <h2 style="font-size:1.5rem;margin-top:0;opacity:0.9;">Công Cụ Soi Cầu KU/Lotobet Đầy Đủ</h2>
    <p style="font-size:1rem;opacity:0.7;">AI Nâng Cao • Dự Đoán Chính Xác • Quản Lý Vốn Thông Minh</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time Monitor
    monitor = RealTimeMonitor()
    time_info = monitor.sync_time()
    
    # Display real-time counter
    col_time1, col_time2, col_time3, col_time4 = st.columns(4)
    
    with col_time1:
        st.markdown(f"""
        <div class="counter-container">
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.8)">KỲ HIỆN TẠI</div>
        <div class="counter-time">{time_info['current_ky']}</div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.6)">{time_info['current_time']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_time2:
        minutes = time_info['seconds_to_next'] // 60
        seconds = time_info['seconds_to_next'] % 60
        st.markdown(f"""
        <div class="counter-container">
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.8)">QUAY TIẾP THEO</div>
        <div class="counter-time">{minutes:02d}:{seconds:02d}</div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.6)">{time_info['next_draw']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_time3:
        if time_info['in_maintenance']:
            status_text = "⛔ BẢO TRÌ"
            status_color = "#ef4444"
        else:
            status_text = "✅ ĐANG HOẠT ĐỘNG"
            status_color = "#10b981"
        
        st.markdown(f"""
        <div class="counter-container">
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.8)">TRẠNG THÁI</div>
        <div class="counter-time" style="color:{status_color};font-size:1.5rem">{status_text}</div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.6)">Lotto A: 2.5 phút/kỳ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_time4:
        # Load data stats
        df = load_lotto_data(10)
        total_records = len(df)
        
        st.markdown(f"""
        <div class="counter-container">
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.8)">DỮ LIỆU</div>
        <div class="counter-time">{total_records}</div>
        <div style="font-size:0.8rem;color:rgba(255,255,255,0.6)">kỳ đã phân tích</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ CẤU HÌNH HỆ THỐNG")
        
        # Data management
        st.markdown("#### 📥 QUẢN LÝ DỮ LIỆU")
        
        data_tab1, data_tab2 = st.tabs(["Nhập tay", "Từ file"])
        
        with data_tab1:
            raw_data = st.text_area(
                "Nhập số 5 chữ số:",
                height=150,
                placeholder="Mỗi dòng 1 số\nVD:\n12345\n67890\n54321",
                help="Nhập tối thiểu 20 số để có phân tích chính xác"
            )
            
            if st.button("💾 LƯU DỮ LIỆU", use_container_width=True):
                if raw_data:
                    lines = raw_data.strip().split('\n')
                    numbers = [line.strip() for line in lines if len(line.strip()) == 5 and line.strip().isdigit()]
                    
                    if numbers:
                        added = save_lotto_results(numbers, time_info['current_ky'])
                        if added > 0:
                            st.success(f"✅ Đã thêm {added} số mới!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("❌ Không có số hợp lệ để lưu")
        
        with data_tab2:
            uploaded_file = st.file_uploader(
                "Chọn file TXT/CSV",
                type=['txt', 'csv'],
                help="File chứa số 5 chữ số, mỗi dòng 1 số"
            )
            
            if uploaded_file is not None:
                content = uploaded_file.getvalue().decode('utf-8')
                numbers = [line.strip() for line in content.split('\n') if len(line.strip()) == 5 and line.strip().isdigit()]
                
                st.info(f"📄 Tìm thấy {len(numbers)} số hợp lệ")
                
                if st.button("📥 NHẬP TỪ FILE", use_container_width=True):
                    if numbers:
                        added = save_lotto_results(numbers)
                        if added > 0:
                            st.success(f"✅ Đã thêm {added} số từ file!")
                            time.sleep(1)
                            st.rerun()
        
        st.markdown("---")
        
        # Analysis settings
        st.markdown("#### 📊 THIẾT LẬP PHÂN TÍCH")
        
        analysis_depth = st.select_slider(
            "Độ sâu phân tích:",
            options=["Cơ bản", "Trung bình", "Nâng cao", "Tối đa"],
            value="Nâng cao"
        )
        
        data_points = st.slider(
            "Số kỳ phân tích:",
            min_value=30,
            max_value=500,
            value=200,
            step=10
        )
        
        auto_refresh = st.checkbox("Tự động làm mới", value=True)
        refresh_interval = st.slider("Chu kỳ làm mới (giây):", 30, 300, 60) if auto_refresh else 60
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("#### 🚀 HÀNH ĐỘNG NHANH")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🔄 LÀM MỚI", use_container_width=True):
                st.rerun()
        
        with col_act2:
            if st.button("🗑️ XÓA DỮ LIỆU CŨ", use_container_width=True):
                if st.checkbox("Xác nhận xóa dữ liệu cũ (trên 30 ngày)?"):
                    clear_old_data(30)
                    st.success("✅ Đã xóa dữ liệu cũ!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        
        # System info
        st.markdown("#### 📈 THÔNG TIN HỆ THỐNG")
        
        df = load_lotto_data(100)
        if not df.empty:
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Tổng kỳ", len(df))
            with col_info2:
                if 'tai_xiu' in df.columns:
                    tai_ratio = (df['tai_xiu'] == 'TÀI').mean() * 100
                    st.metric("Tỷ lệ Tài", f"{tai_ratio:.1f}%")
        
        if AI_LIBS_AVAILABLE:
            st.success("✅ AI SẴN SÀNG")
        else:
            st.warning("⚠️ AI GIỚI HẠN")
    
    # Main content tabs
    tab_main1, tab_main2, tab_main3, tab_main4, tab_main5 = st.tabs([
        "🎯 DỰ ĐOÁN CHÍNH",
        "📊 PHÂN TÍCH CHI TIẾT",
        "🔍 PHÁT HIỆN PATTERN",
        "💰 QUẢN LÝ VỐN",
        "📚 HƯỚNG DẪN"
    ])
    
    with tab_main1:
        # Main predictions tab
        st.markdown("### 🎯 DỰ ĐOÁN CHO KỲ TIẾP THEO")
        
        # Load data
        df = load_lotto_data(data_points)
        
        if df.empty or len(df) < 30:
            st.warning("""
            ⚠️ **CHƯA ĐỦ DỮ LIỆU ĐỂ PHÂN TÍCH**
            
            **Yêu cầu tối thiểu:** 30 kỳ quay
            **Hiện có:** {} kỳ
            
            **Vui lòng:**
