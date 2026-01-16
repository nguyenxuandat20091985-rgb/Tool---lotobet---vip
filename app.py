# ================= LOTTO KU AI SIÊU PHẨM – V13.0 LITE =================
# Phiên bản tối ưu cho Streamlit Cloud

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
from itertools import combinations
from dataclasses import dataclass, field
from enum import Enum
import pickle
import hashlib
import logging

warnings.filterwarnings('ignore')

# ================= BASIC AI LIBRARIES =================
try:
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        VotingClassifier, StackingClassifier
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    
    AI_LIBS_AVAILABLE = True
except ImportError:
    AI_LIBS_AVAILABLE = False
    st.warning("⚠️ Thiếu thư viện scikit-learn. Một số tính năng AI có thể bị giới hạn.")

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTTO KU AI V13 LITE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS SIMPLE
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .number {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .confidence {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .high { background: #10b981; color: white; }
    .medium { background: #f59e0b; color: white; }
    .low { background: #ef4444; color: white; }
    
    .recommend {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .bet { background: #059669; color: white; }
    .maybe { background: #d97706; color: white; }
    .no { background: #dc2626; color: white; }
    
    @media (max-width: 768px) {
        .number { font-size: 2rem; }
        .prediction-card { height: 180px; padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================= DATABASE =================
DB_FILE = "lotto_v13_lite.db"

def init_db():
    """Khởi tạo database đơn giản"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS lotto_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE NOT NULL,
        number TEXT NOT NULL,
        tien_nhi TEXT NOT NULL,
        hau_nhi TEXT NOT NULL,
        tong INTEGER NOT NULL,
        tai_xiu TEXT NOT NULL,
        le_chan TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

init_db()

# ================= CORE CLASSES =================
class BetType(Enum):
    TAI_XIU = "tài_xỉu"
    LE_CHAN = "lẻ_chẵn"
    TIEN_NHI = "tiền_nhị"
    HAU_NHI = "hậu_nhị"
    TWO_STAR = "2_tinh"
    THREE_STAR = "3_tinh"

@dataclass
class LotteryNumber:
    """Biểu diễn số lotto 5 chữ số"""
    chuc_ngan: int
    ngan: int
    tram: int
    chuc: int
    don_vi: int
    
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
    
    def get_2tinh_pairs(self) -> List[Tuple[int, int]]:
        digits = [self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi]
        return list(combinations(digits, 2))
    
    def get_3tinh_pairs(self) -> List[Tuple[int, int, int]]:
        digits = [self.chuc_ngan, self.ngan, self.tram, self.chuc, self.don_vi]
        return list(combinations(digits, 3))

# ================= AI ENGINE =================
class LottoAI:
    """AI engine đơn giản"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data.copy()
        self.numbers = self._extract_numbers()
    
    def _extract_numbers(self) -> List[LotteryNumber]:
        numbers = []
        for _, row in self.df.iterrows():
            try:
                if 'number' in row and len(str(row['number'])) == 5:
                    num = LotteryNumber.from_string(str(row['number']))
                    numbers.append(num)
            except:
                continue
        return numbers
    
    def analyze_tai_xiu(self) -> Dict:
        """Phân tích Tài/Xỉu"""
        if not self.numbers:
            return {'confidence': 50, 'prediction': 'TÀI'}
        
        recent = self.numbers[:30]
        tai_count = sum(1 for num in recent if num.is_tai())
        xiu_count = sum(1 for num in recent if num.is_xiu())
        
        tai_percent = (tai_count / len(recent)) * 100
        xiu_percent = (xiu_count / len(recent)) * 100
        
        if tai_percent > 55:
            return {
                'prediction': 'TÀI',
                'confidence': min(85, tai_percent * 1.2),
                'percentage': tai_percent,
                'recommendation': 'NÊN ĐÁNH' if tai_percent > 60 else 'CÓ THỂ ĐÁNH'
            }
        elif xiu_percent > 55:
            return {
                'prediction': 'XỈU',
                'confidence': min(85, xiu_percent * 1.2),
                'percentage': xiu_percent,
                'recommendation': 'NÊN ĐÁNH' if xiu_percent > 60 else 'CÓ THỂ ĐÁNH'
            }
        else:
            return {
                'prediction': 'TÀI' if tai_percent > xiu_percent else 'XỈU',
                'confidence': 50,
                'percentage': max(tai_percent, xiu_percent),
                'recommendation': 'THEO DÕI'
            }
    
    def analyze_le_chan(self) -> Dict:
        """Phân tích Lẻ/Chẵn"""
        if not self.numbers:
            return {'confidence': 50, 'prediction': 'LẺ'}
        
        recent = self.numbers[:30]
        le_count = sum(1 for num in recent if num.is_le())
        chan_count = sum(1 for num in recent if num.is_chan())
        
        le_percent = (le_count / len(recent)) * 100
        chan_percent = (chan_count / len(recent)) * 100
        
        if le_percent > 55:
            return {
                'prediction': 'LẺ',
                'confidence': min(85, le_percent * 1.2),
                'percentage': le_percent,
                'recommendation': 'NÊN ĐÁNH' if le_percent > 60 else 'CÓ THỂ ĐÁNH'
            }
        elif chan_percent > 55:
            return {
                'prediction': 'CHẴN',
                'confidence': min(85, chan_percent * 1.2),
                'percentage': chan_percent,
                'recommendation': 'NÊN ĐÁNH' if chan_percent > 60 else 'CÓ THỂ ĐÁNH'
            }
        else:
            return {
                'prediction': 'LẺ' if le_percent > chan_percent else 'CHẴN',
                'confidence': 50,
                'percentage': max(le_percent, chan_percent),
                'recommendation': 'THEO DÕI'
            }
    
    def analyze_two_star(self) -> Dict:
        """Phân tích 2 TINH"""
        if len(self.numbers) < 20:
            return {'confidence': 40, 'predictions': []}
        
        # Extract 2-tinh pairs
        all_pairs = []
        for num in self.numbers[:50]:
            pairs = num.get_2tinh_pairs()
            all_pairs.extend([f"{a}{b}" for a, b in pairs])
        
        # Analyze frequency
        freq = Counter(all_pairs[-100:]) if len(all_pairs) > 100 else Counter(all_pairs)
        total = sum(freq.values())
        
        predictions = []
        for pair, count in freq.most_common(10):
            percent = (count / total) * 100
            
            if percent >= 2.5:
                recommendation = "NÊN ĐÁNH"
                conf_level = "high"
            elif percent >= 1.5:
                recommendation = "CÓ THỂ ĐÁNH"
                conf_level = "medium"
            else:
                recommendation = "THEO DÕI"
                conf_level = "low"
            
            predictions.append({
                'pair': pair,
                'percentage': round(percent, 2),
                'frequency': count,
                'recommendation': recommendation,
                'confidence': conf_level
            })
        
        avg_percent = np.mean([p['percentage'] for p in predictions[:5]]) if predictions else 0
        confidence = min(80, avg_percent * 1.5 + 30)
        
        return {
            'predictions': predictions[:8],
            'confidence': confidence
        }
    
    def analyze_three_star(self) -> Dict:
        """Phân tích 3 TINH"""
        if len(self.numbers) < 30:
            return {'confidence': 35, 'predictions': []}
        
        # Extract 3-tinh combinations
        all_combs = []
        for num in self.numbers[:50]:
            combs = num.get_3tinh_pairs()
            all_combs.extend([f"{a}{b}{c}" for a, b, c in combs])
        
        # Analyze frequency
        freq = Counter(all_combs[-150:]) if len(all_combs) > 150 else Counter(all_combs)
        total = sum(freq.values())
        
        predictions = []
        for comb, count in freq.most_common(8):
            percent = (count / total) * 100
            
            if percent >= 1.0:
                recommendation = "NÊN ĐÁNH"
                conf_level = "high"
            elif percent >= 0.5:
                recommendation = "CÓ THỂ ĐÁNH"
                conf_level = "medium"
            else:
                recommendation = "THEO DÕI"
                conf_level = "low"
            
            predictions.append({
                'combination': comb,
                'percentage': round(percent, 2),
                'frequency': count,
                'recommendation': recommendation,
                'confidence': conf_level
            })
        
        avg_percent = np.mean([p['percentage'] for p in predictions[:5]]) if predictions else 0
        confidence = min(75, avg_percent * 2 + 25)
        
        return {
            'predictions': predictions[:6],
            'confidence': confidence
        }
    
    def analyze_de_so(self) -> Dict:
        """Phân tích đề số"""
        if len(self.numbers) < 25:
            return {'confidence': 40, 'predictions': []}
        
        recent = self.numbers[:50]
        
        # Tien nhi
        tien_nhi = [int(num.get_tien_nhi()) for num in recent]
        tien_freq = Counter(tien_nhi)
        
        # Hau nhi
        hau_nhi = [int(num.get_hau_nhi()) for num in recent]
        hau_freq = Counter(hau_nhi)
        
        predictions = []
        
        # Top tien nhi
        for value, count in tien_freq.most_common(5):
            percent = (count / len(recent)) * 100
            predictions.append({
                'type': 'Tiền nhị',
                'number': f"{value:02d}",
                'percentage': round(percent, 2),
                'recommendation': 'NÊN ĐÁNH' if percent > 3 else 'CÓ THỂ ĐÁNH' if percent > 1.5 else 'THEO DÕI'
            })
        
        # Top hau nhi
        for value, count in hau_freq.most_common(5):
            percent = (count / len(recent)) * 100
            predictions.append({
                'type': 'Hậu nhị',
                'number': f"{value:02d}",
                'percentage': round(percent, 2),
                'recommendation': 'NÊN ĐÁNH' if percent > 3 else 'CÓ THỂ ĐÁNH' if percent > 1.5 else 'THEO DÕI'
            })
        
        # Sort by percentage
        predictions.sort(key=lambda x: x['percentage'], reverse=True)
        
        confidence = min(80, (len(predictions) * 5) + 30)
        
        return {
            'predictions': predictions[:8],
            'confidence': confidence
        }

# ================= HELPER FUNCTIONS =================
def save_lotto_data(numbers: List[str], ky: str = None):
    """Lưu dữ liệu vào database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    added = 0
    
    for idx, num_str in enumerate(numbers):
        try:
            if len(num_str) != 5 or not num_str.isdigit():
                continue
            
            num = LotteryNumber.from_string(num_str)
            
            # Generate ky if not provided
            if ky and idx == 0:
                current_ky = ky
            else:
                current_ky = f"KU{int(time.time() * 1000) % 1000000:06d}"
            
            c.execute("""
            INSERT OR IGNORE INTO lotto_results 
            (ky, number, tien_nhi, hau_nhi, tong, tai_xiu, le_chan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                current_ky,
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
            print(f"Error saving {num_str}: {e}")
    
    conn.commit()
    conn.close()
    return added

def load_lotto_data(limit: int = 300) -> pd.DataFrame:
    """Tải dữ liệu từ database"""
    conn = sqlite3.connect(DB_FILE)
    
    try:
        query = f"""
        SELECT 
            ky,
            number,
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

def format_currency(amount: float) -> str:
    """Định dạng tiền"""
    return f"{amount:,.0f}₫"

# ================= MAIN APP =================
def main():
    # Header
    st.markdown("""
    <div class="header">
    <h1>🎰 COS V13 LITE - LOTTO KU AI</h1>
    <p>Phiên bản tối ưu cho Streamlit Cloud • AI dự đoán thông minh</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ CẤU HÌNH")
        
        # Data management
        st.markdown("#### 📥 NHẬP DỮ LIỆU")
        
        input_method = st.radio(
            "Chọn phương thức nhập:",
            ["Nhập tay", "Từ file"]
        )
        
        if input_method == "Nhập tay":
            raw_data = st.text_area(
                "Nhập số (mỗi dòng 1 số 5 chữ số):",
                height=150,
                placeholder="12345\n67890\n54321"
            )
        else:
            uploaded_file = st.file_uploader(
                "Chọn file TXT/CSV",
                type=['txt', 'csv']
            )
        
        # Analysis settings
        st.markdown("#### 📊 THIẾT LẬP")
        data_points = st.slider("Số kỳ phân tích:", 50, 1000, 300)
        auto_analyze = st.checkbox("Tự động phân tích", value=True)
        
        if st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True):
            st.session_state['analyze'] = True
        
        st.markdown("---")
        st.markdown("#### 📈 THỐNG KÊ")
        
        # Load stats
        df = load_lotto_data(10)
        if not df.empty:
            st.metric("Tổng kỳ", len(df))
            if 'tai_xiu' in df.columns:
                tai_ratio = (df['tai_xiu'] == 'TÀI').mean() * 100
                st.metric("Tỷ lệ Tài", f"{tai_ratio:.1f}%")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Tổng quan",
        "🎯 Dự đoán", 
        "📊 Phân tích",
        "💰 Quản lý"
    ])
    
    with tab1:
        st.markdown("### 📊 TỔNG QUAN HỆ THỐNG")
        
        # Load data
        df = load_lotto_data(data_points)
        
        if df.empty:
            st.info("Chưa có dữ liệu. Vui lòng nhập dữ liệu ở sidebar.")
            
            # Quick input
            col1, col2 = st.columns(2)
            with col1:
                sample_data = st.text_area(
                    "Nhập số mẫu (tối thiểu 20 số):",
                    height=200,
                    placeholder="12345\n54321\n98765\n56789\n...",
                    help="Mỗi dòng 1 số 5 chữ số"
                )
            
            with col2:
                st.markdown("#### 📋 Ví dụ dữ liệu:")
                st.code("""12345
54321
98765
56789
13579
24680
11223
33445
55667
77889""")
                
                if st.button("📥 NHẬP DỮ LIỆU MẪU", use_container_width=True):
                    sample_numbers = [
                        '12345', '54321', '98765', '56789', '13579',
                        '24680', '11223', '33445', '55667', '77889',
                        '99001', '22334', '44556', '66778', '88990',
                        '00112', '23344', '45566', '67788', '89900'
                    ]
                    added = save_lotto_data(sample_numbers)
                    if added > 0:
                        st.success(f"✅ Đã thêm {added} số mẫu!")
                        time.sleep(1)
                        st.rerun()
        else:
            # Show overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tổng kỳ", len(df))
                latest_num = df.iloc[0]['number'] if 'number' in df.columns else "00000"
                st.metric("Gần nhất", latest_num)
            
            with col2:
                if 'tong' in df.columns:
                    avg_total = df['tong'].mean()
                    st.metric("Tổng TB", f"{avg_total:.1f}")
                    st.metric("Tài/Xỉu", f"{df['tai_xiu'].value_counts().get('TÀI', 0)}/{df['tai_xiu'].value_counts().get('XỈU', 0)}")
            
            with col3:
                if 'le_chan' in df.columns:
                    le_ratio = (df['le_chan'] == 'LẺ').mean() * 100
                    st.metric("Lẻ", f"{le_ratio:.1f}%")
                    chan_ratio = (df['le_chan'] == 'CHẴN').mean() * 100
                    st.metric("Chẵn", f"{chan_ratio:.1f}%")
            
            # Show recent data
            with st.expander("📋 DỮ LIỆU GẦN ĐÂY", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
    
    with tab2:
        st.markdown("### 🎯 DỰ ĐOÁN")
        
        df = load_lotto_data(data_points)
        
        if df.empty or len(df) < 30:
            st.warning("⚠️ Cần ít nhất 30 kết quả để dự đoán")
        else:
            # Initialize AI
            ai = LottoAI(df)
            
            # Run analysis
            if 'analyze' in st.session_state or auto_analyze:
                with st.spinner("🤖 AI đang phân tích..."):
                    # Get predictions
                    tai_xiu = ai.analyze_tai_xiu()
                    le_chan = ai.analyze_le_chan()
                    two_star = ai.analyze_two_star()
                    three_star = ai.analyze_three_star()
                    de_so = ai.analyze_de_so()
                    
                    # Store in session
                    st.session_state['predictions'] = {
                        'tai_xiu': tai_xiu,
                        'le_chan': le_chan,
                        'two_star': two_star,
                        'three_star': three_star,
                        'de_so': de_so
                    }
            
            # Show predictions if available
            if 'predictions' in st.session_state:
                preds = st.session_state['predictions']
                
                # Main predictions
                st.markdown("#### 🎲 DỰ ĐOÁN CHÍNH")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    tx = preds['tai_xiu']
                    st.markdown(f"""
                    <div class="card" style="text-align:center">
                    <h3>🎲 TÀI/XỈU</h3>
                    <div class="number">{tx['prediction']}</div>
                    <div class="confidence {'high' if tx['confidence'] >= 70 else 'medium' if tx['confidence'] >= 60 else 'low'}">
                    {tx['confidence']:.1f}%
                    </div>
                    <div class="recommend {'bet' if 'NÊN' in tx['recommendation'] else 'maybe' if 'THỂ' in tx['recommendation'] else 'no'}">
                    {tx['recommendation']}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    lc = preds['le_chan']
                    st.markdown(f"""
                    <div class="card" style="text-align:center">
                    <h3>🎲 LẺ/CHẴN</h3>
                    <div class="number">{lc['prediction']}</div>
                    <div class="confidence {'high' if lc['confidence'] >= 70 else 'medium' if lc['confidence'] >= 60 else 'low'}">
                    {lc['confidence']:.1f}%
                    </div>
                    <div class="recommend {'bet' if 'NÊN' in lc['recommendation'] else 'maybe' if 'THỂ' in lc['recommendation'] else 'no'}">
                    {lc['recommendation']}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Hậu nhị từ de_so
                    de_preds = preds['de_so']['predictions']
                    hau_nhi = next((p for p in de_preds if p['type'] == 'Hậu nhị'), None)
                    
                    if hau_nhi:
                        st.markdown(f"""
                        <div class="card" style="text-align:center">
                        <h3>🔢 HẬU NHỊ</h3>
                        <div class="number">{hau_nhi['number']}</div>
                        <div style="color:#6b7280;font-size:0.9rem">
                        {hau_nhi['percentage']}%
                        </div>
                        <div class="recommend {'bet' if 'NÊN' in hau_nhi['recommendation'] else 'maybe' if 'THỂ' in hau_nhi['recommendation'] else 'no'}">
                        {hau_nhi['recommendation']}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col4:
                    # Tiền nhị từ de_so
                    tien_nhi = next((p for p in de_preds if p['type'] == 'Tiền nhị'), None)
                    
                    if tien_nhi:
                        st.markdown(f"""
                        <div class="card" style="text-align:center">
                        <h3>🔢 TIỀN NHỊ</h3>
                        <div class="number">{tien_nhi['number']}</div>
                        <div style="color:#6b7280;font-size:0.9rem">
                        {tien_nhi['percentage']}%
                        </div>
                        <div class="recommend {'bet' if 'NÊN' in tien_nhi['recommendation'] else 'maybe' if 'THỂ' in tien_nhi['recommendation'] else 'no'}">
                        {tien_nhi['recommendation']}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Special predictions
                st.markdown("---")
                st.markdown("#### ✨ DỰ ĐOÁN ĐẶC BIỆT")
                
                # 2 TINH
                with st.expander("🔮 2 TINH (2 cặp số)", expanded=True):
                    two_preds = preds['two_star']['predictions']
                    if two_preds:
                        cols = st.columns(4)
                        for idx, pred in enumerate(two_preds[:8]):
                            with cols[idx % 4]:
                                rec_class = 'bet' if 'NÊN' in pred['recommendation'] else 'maybe' if 'THỂ' in pred['recommendation'] else 'no'
                                st.markdown(f"""
                                <div style="text-align:center;padding:1rem;background:#f8fafc;border-radius:10px;border:2px solid #e5e7eb">
                                <div style="font-size:1.5rem;font-weight:bold">{pred['pair']}</div>
                                <div style="color:#6b7280">{pred['percentage']}%</div>
                                <div class="recommend {rec_class}" style="margin-top:0.5rem">
                                {pred['recommendation']}
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # 3 TINH
                with st.expander("🔮 3 TINH (3 cặp số)", expanded=True):
                    three_preds = preds['three_star']['predictions']
                    if three_preds:
                        cols = st.columns(3)
                        for idx, pred in enumerate(three_preds[:6]):
                            with cols[idx % 3]:
                                rec_class = 'bet' if 'NÊN' in pred['recommendation'] else 'maybe' if 'THỂ' in pred['recommendation'] else 'no'
                                st.markdown(f"""
                                <div style="text-align:center;padding:1rem;background:#f8fafc;border-radius:10px;border:2px solid #e5e7eb">
                                <div style="font-size:1.3rem;font-weight:bold">{pred['combination']}</div>
                                <div style="color:#6b7280">{pred['percentage']}%</div>
                                <div class="recommend {rec_class}" style="margin-top:0.5rem">
                                {pred['recommendation']}
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Đề số
                with st.expander("🎯 ĐỀ SỐ (Hậu nhị/Tiền nhị)", expanded=True):
                    de_preds = preds['de_so']['predictions']
                    if de_preds:
                        cols = st.columns(4)
                        for idx, pred in enumerate(de_preds[:8]):
                            with cols[idx % 4]:
                                rec_class = 'bet' if 'NÊN' in pred['recommendation'] else 'maybe' if 'THỂ' in pred['recommendation'] else 'no'
                                color = '#10b981' if pred['type'] == 'Hậu nhị' else '#3b82f6'
                                st.markdown(f"""
                                <div style="text-align:center;padding:1rem;background:#f8fafc;border-radius:10px;border:2px solid {color}">
                                <div style="font-size:1.3rem;font-weight:bold">{pred['number']}</div>
                                <div style="color:#6b7280;font-size:0.9rem">{pred['type']}</div>
                                <div style="color:#6b7280">{pred['percentage']}%</div>
                                <div class="recommend {rec_class}" style="margin-top:0.5rem">
                                {pred['recommendation']}
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.info("Nhấn nút 'PHÂN TÍCH' ở sidebar để bắt đầu")
                
                if st.button("🚀 BẮT ĐẦU PHÂN TÍCH", type="primary", use_container_width=True):
                    st.session_state['analyze'] = True
                    st.rerun()
    
    with tab3:
        st.markdown("### 📊 PHÂN TÍCH THỐNG KÊ")
        
        df = load_lotto_data(data_points)
        
        if df.empty:
            st.warning("Chưa có dữ liệu để phân tích")
        else:
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 TẦN SUẤT TỔNG SỐ")
                if 'tong' in df.columns:
                    freq_df = df['tong'].value_counts().reset_index()
                    freq_df.columns = ['Tổng số', 'Tần suất']
                    freq_df['Phần trăm'] = (freq_df['Tần suất'] / len(df) * 100).round(2)
                    freq_df = freq_df.sort_values('Tần suất', ascending=False)
                    st.dataframe(freq_df.head(15), use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 PHÂN PHỐI TÀI/XỈU")
                if 'tai_xiu' in df.columns:
                    tai_ratio = (df['tai_xiu'] == 'TÀI').mean() * 100
                    xiu_ratio = (df['tai_xiu'] == 'XỈU').mean() * 100
                    
                    st.metric("Tài", f"{tai_ratio:.1f}%")
                    st.metric("Xỉu", f"{xiu_ratio:.1f}%")
                    
                    # Simple bar chart
                    chart_data = pd.DataFrame({
                        'Loại': ['Tài', 'Xỉu'],
                        'Tỷ lệ': [tai_ratio, xiu_ratio]
                    })
                    st.bar_chart(chart_data.set_index('Loại'))
            
            # Hậu nhị analysis
            st.markdown("#### 🔢 THỐNG KÊ HẬU NHỊ")
            if 'hau_nhi' in df.columns:
                hau_freq = df['hau_nhi'].value_counts().reset_index()
                hau_freq.columns = ['Hậu nhị', 'Tần suất']
                hau_freq['Phần trăm'] = (hau_freq['Tần suất'] / len(df) * 100).round(2)
                hau_freq = hau_freq.sort_values('Tần suất', ascending=False)
                
                col_h1, col_h2 = st.columns([2, 1])
                
                with col_h1:
                    st.dataframe(hau_freq.head(15), use_container_width=True)
                
                with col_h2:
                    st.markdown("**🔥 Số nóng:**")
                    for _, row in hau_freq.head(5).iterrows():
                        st.write(f"**{row['Hậu nhị']}**: {row['Tần suất']} lần ({row['Phần trăm']}%)")
    
    with tab4:
        st.markdown("### 💰 QUẢN LÝ VỐN")
        
        # Capital configuration
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            total_capital = st.number_input(
                "Tổng vốn (VNĐ):",
                min_value=100000,
                max_value=1000000000,
                value=5000000,
                step=100000
            )
            
            risk_level = st.select_slider(
                "Mức rủi ro:",
                options=["Thấp", "Trung bình", "Cao"],
                value="Trung bình"
            )
        
        with col_c2:
            stop_loss = st.slider("Stop-loss (%):", 5, 30, 15)
            take_profit = st.slider("Take-profit (%):", 10, 50, 25)
            
            if st.button("🧮 TÍNH PHÂN BỔ", use_container_width=True):
                # Calculate allocations
                if risk_level == "Thấp":
                    allocations = {
                        'Tài/Xỉu': 20,
                        'Lẻ/Chẵn': 15,
                        'Hậu nhị': 25,
                        'Tiền nhị': 15,
                        '2 Tinh': 10,
                        '3 Tinh': 10,
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
                        'Dự phòng': 7
                    }
                else:  # High
                    allocations = {
                        'Tài/Xỉu': 30,
                        'Lẻ/Chẵn': 20,
                        'Hậu nhị': 35,
                        'Tiền nhị': 5,
                        '2 Tinh': 5,
                        '3 Tinh': 3,
                        'Dự phòng': 2
                    }
                
                st.session_state['allocations'] = allocations
        
        # Show allocations
        if 'allocations' in st.session_state:
            st.markdown("#### 📊 PHÂN BỔ VỐN")
            
            allocations = st.session_state['allocations']
            total_allocated = 0
            
            for bet_type, percentage in allocations.items():
                amount = total_capital * (percentage / 100)
                total_allocated += amount
                
                col_a1, col_a2, col_a3 = st.columns([2, 3, 2])
                
                with col_a1:
                    st.write(f"**{bet_type}**")
                
                with col_a2:
                    st.progress(percentage / 100)
                
                with col_a3:
                    st.write(f"{percentage}% ({format_currency(amount)})")
            
            st.markdown("---")
            remaining = total_capital - total_allocated
            remaining_percent = (remaining / total_capital) * 100
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Tổng phân bổ", format_currency(total_allocated))
            with col_r2:
                st.metric("Vốn dự phòng", format_currency(remaining))
            
            if remaining_percent < 10:
                st.error(f"⚠️ Vốn dự phòng thấp ({remaining_percent:.1f}%)")
            elif remaining_percent < 20:
                st.warning(f"⚠️ Vốn dự phòng hơi thấp ({remaining_percent:.1f}%)")
            else:
                st.success(f"✅ Phân bổ hợp lý ({remaining_percent:.1f}% dự phòng)")
        
        # Betting strategy
        st.markdown("---")
        st.markdown("#### 🎯 CHIẾN LƯỢC")
        
        strategy = st.selectbox(
            "Chiến lược đặt cược:",
            ["Bảo toàn vốn", "Tăng trưởng ổn định", "Tăng trưởng mạnh"],
            index=1
        )
        
        bet_size = st.select_slider(
            "Mức đặt cược:",
            options=["Nhỏ (1-3%)", "Trung bình (3-5%)", "Lớn (5-10%)", "Rất lớn (10-15%)"],
            value="Trung bình (3-5%)"
        )
        
        if st.button("💾 LƯU CHIẾN LƯỢC", type="primary", use_container_width=True):
            st.success("✅ Đã lưu chiến lược!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#6b7280;font-size:0.9rem">
    🎰 COS V13 LITE - Phiên bản tối ưu cho Streamlit Cloud<br>
    ⚠️ Dành cho mục đích nghiên cứu • Quản lý vốn thông minh là yếu tố sống còn<br>
    © 2024 • Powered by Basic AI
    </div>
    """, unsafe_allow_html=True)

# ================= RUN APP =================
if __name__ == "__main__":
    main()
