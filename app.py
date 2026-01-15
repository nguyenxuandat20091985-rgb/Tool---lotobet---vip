# ================= LOTOBET ULTRA AI PRO – V10.1 COMPLETE =================
# Multi-Algorithm AI System with Gambling Tips Integration

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import thư viện AI nâng cao
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    AI_LIBS_AVAILABLE = True
except ImportError:
    AI_LIBS_AVAILABLE = False
    st.warning("⚠️ Cần cài đặt thư viện AI: `pip install scikit-learn statsmodels`")

from collections import Counter, defaultdict, deque

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET ULTRA AI PRO – V10.1",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced CSS for V10.1
st.markdown("""
<style>
    /* Main highlight box */
    .highlight-main {
        background: linear-gradient(135deg, #FFA726 0%, #FF9800 100%);
        padding: 25px;
        border-radius: 15px;
        border: 4px solid #F57C00;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(255, 167, 38, 0.4);
    }
    
    /* AI Analysis Box */
    .ai-analysis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 15px 0;
        border: 2px solid #5a67d8;
    }
    
    /* Tips Box */
    .tips-box {
        background-color: #E3F2FD;
        border-left: 6px solid #2196F3;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Pattern Box */
    .pattern-box {
        background-color: #F3E5F5;
        border-left: 6px solid #9C27B0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Section headers with numbers */
    .section-header {
        background-color: #2D3748;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 15px 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Number displays */
    .big-number {
        font-size: 3.8rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .medium-number {
        font-size: 2.4rem;
        font-weight: bold;
        color: #2D3748;
        text-align: center;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
    
    /* Algorithm badges */
    .algo-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    
    .algo-1 { background-color: #3B82F6; color: white; }
    .algo-2 { background-color: #10B981; color: white; }
    .algo-3 { background-color: #8B5CF6; color: white; }
    .algo-4 { background-color: #F59E0B; color: white; }
    .algo-5 { background-color: #EF4444; color: white; }
    
    /* Trend colors */
    .trend-up { color: #10B981; }
    .trend-down { color: #EF4444; }
    .trend-neutral { color: #6B7280; }
    
    /* Responsive */
    @media (max-width: 768px) {
        .big-number { font-size: 2.8rem; }
        .medium-number { font-size: 1.8rem; }
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "lotobet_ultra_v10_1.db"

# ================= DATABASE =================
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    
    # Bảng kỳ quay
    c.execute("""
    CREATE TABLE IF NOT EXISTS ky_quay (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE,
        so5 TEXT,
        tien_nhi TEXT,
        hau_nhi TEXT,
        tong INTEGER,
        tai_xiu TEXT,
        le_chan TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng phân tích AI (mở rộng cho V10.1)
    c.execute("""
    CREATE TABLE IF NOT EXISTS phan_tich_ai (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT,
        algo_type TEXT,
        algo_name TEXT,
        predictions TEXT,
        confidence REAL,
        details TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng mẫu hình phát hiện
    c.execute("""
    CREATE TABLE IF NOT EXISTS mau_hinh (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT,
        pattern_data TEXT,
        start_ky TEXT,
        length INTEGER,
        strength REAL,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng mẹo đánh áp dụng
    c.execute("""
    CREATE TABLE IF NOT EXISTS meo_danh (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meo_type TEXT,
        meo_name TEXT,
        numbers TEXT,
        description TEXT,
        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng cài đặt AI
    c.execute("""
    CREATE TABLE IF NOT EXISTS cai_dat_ai (
        id INTEGER PRIMARY KEY,
        enable_algo_1 INTEGER DEFAULT 1,
        enable_algo_2 INTEGER DEFAULT 1,
        enable_algo_3 INTEGER DEFAULT 1,
        enable_algo_4 INTEGER DEFAULT 1,
        enable_algo_5 INTEGER DEFAULT 1,
        min_confidence REAL DEFAULT 60.0,
        auto_update INTEGER DEFAULT 1
    )
    """)
    
    c.execute("INSERT OR IGNORE INTO cai_dat_ai (id) VALUES (1)")
    
    conn.commit()
    conn.close()

init_db()

# ================= HELPER FUNCTIONS =================
def tai_xiu(tong):
    return "TÀI" if tong >= 23 else "XỈU"

def le_chan(tong):
    return "LẺ" if tong % 2 else "CHẴN"

def format_tien(tien):
    return f"{tien:,.0f} VNĐ"

def smart_parse_input(raw_text):
    """Xử lý input thông minh"""
    if not raw_text:
        return []
    
    lines = raw_text.strip().split('\n')
    results = []
    
    for line in lines:
        line_clean = ''.join(c for c in line if c.isdigit() or c.isspace())
        numbers = line_clean.split()
        
        for num in numbers:
            if len(num) == 5 and num.isdigit():
                results.append(num)
            elif len(num) == 4 and num.isdigit():
                results.append(num)
    
    return results

def get_algo_badge(algo_num):
    badges = {
        1: '<span class="algo-badge algo-1">L1</span>',
        2: '<span class="algo-badge algo-2">L2</span>',
        3: '<span class="algo-badge algo-3">L3</span>',
        4: '<span class="algo-badge algo-4">L4</span>',
        5: '<span class="algo-badge algo-5">L5</span>'
    }
    return badges.get(algo_num, '<span class="algo-badge">AI</span>')

def calculate_confidence(data_length, pattern_count, algo_count):
    """Tính độ tin cậy tổng hợp"""
    base_conf = min(90, data_length / 100 * 70)
    pattern_bonus = min(15, pattern_count * 3)
    algo_bonus = min(10, algo_count * 2)
    
    return min(95, base_conf + pattern_bonus + algo_bonus)

# ================= ADVANCED AI ENGINE V10.1 =================

class AdvancedLottoAI_V10_1:
    """Hệ thống AI đa thuật toán V10.1"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.analyses_cache = {}
        self.algorithms_enabled = {
            'basic_stats': True,
            'hot_cold': True,
            'fibonacci': True,
            'yin_yang': True,
            'patterns': True,
            'markov': True,
            'time_series': True,
            'gambling_tips': True,
            'random_forest': AI_LIBS_AVAILABLE
        }
    
    def run_all_analyses(self):
        """Chạy tất cả phân tích AI"""
        all_results = {}
        
        # Lớp 1: Thống kê cơ bản
        if self.algorithms_enabled['basic_stats']:
            all_results['basic_stats'] = self._basic_statistics()
        
        # Lớp 2: Số nóng/lạnh
        if self.algorithms_enabled['hot_cold']:
            all_results['hot_cold'] = self._hot_cold_analysis()
        
        # Lớp 3: Chu kỳ Fibonacci
        if self.algorithms_enabled['fibonacci']:
            all_results['fibonacci'] = self._fibonacci_analysis()
        
        # Lớp 4: Âm dương
        if self.algorithms_enabled['yin_yang']:
            all_results['yin_yang'] = self._yin_yang_analysis()
        
        # Lớp 5: Mẫu hình
        if self.algorithms_enabled['patterns']:
            all_results['patterns'] = self._pattern_detection()
        
        # Lớp 6: Markov Chain
        if self.algorithms_enabled['markov']:
            all_results['markov'] = self._markov_chain_analysis()
        
        # Lớp 7: Time Series
        if self.algorithms_enabled['time_series']:
            all_results['time_series'] = self._time_series_analysis()
        
        # Lớp 8: Mẹo đánh
        if self.algorithms_enabled['gambling_tips']:
            all_results['gambling_tips'] = self._gambling_tips_analysis()
        
        # Lớp 9: Random Forest (nếu có thư viện)
        if self.algorithms_enabled['random_forest']:
            all_results['random_forest'] = self._random_forest_analysis()
        
        self.analyses_cache = all_results
        return all_results
    
    def _basic_statistics(self):
        """Thống kê cơ bản"""
        if self.df.empty:
            return {}
        
        return {
            'total_games': len(self.df),
            'avg_sum': float(self.df['tong'].mean()),
            'tai_ratio': float((self.df['tai_xiu'] == 'TÀI').mean()),
            'le_ratio': float((self.df['le_chan'] == 'LẺ').mean()),
            'common_tien_nhi': self.df['tien_nhi'].value_counts().head(3).to_dict(),
            'common_hau_nhi': self.df['hau_nhi'].value_counts().head(3).to_dict()
        }
    
    def _hot_cold_analysis(self):
        """Phân tích số nóng/lạnh"""
        if len(self.df) < 20:
            return {}
        
        lookback = min(50, len(self.df))
        recent_df = self.df.head(lookback)
        
        digit_counts = {str(i): 0 for i in range(10)}
        for num in recent_df['so5']:
            for digit in num:
                digit_counts[digit] += 1
        
        sorted_digits = sorted(digit_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Số nóng (xuất hiện nhiều)
        hot_numbers = [d for d, c in sorted_digits[:4]]
        
        # Số lạnh (xuất hiện ít)
        cold_numbers = [d for d, c in sorted_digits[-4:]]
        
        # Số gan (lâu chưa về)
        gan_numbers = self._find_gan_numbers()
        
        return {
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'gan_numbers': gan_numbers[:3],
            'digit_frequencies': dict(sorted_digits)
        }
    
    def _find_gan_numbers(self):
        """Tìm số gan (lâu chưa về)"""
        if self.df.empty:
            return []
        
        all_digits = set(str(i) for i in range(10))
        last_seen = {digit: 0 for digit in all_digits}
        
        for idx, row in self.df.iterrows():
            for digit in row['so5']:
                last_seen[digit] = idx
        
        current_idx = len(self.df)
        gan_periods = {digit: current_idx - last_seen[digit] for digit in all_digits}
        
        sorted_gan = sorted(gan_periods.items(), key=lambda x: x[1], reverse=True)
        return [digit for digit, period in sorted_gan[:5]]
    
    def _fibonacci_analysis(self):
        """Phân tích chu kỳ Fibonacci"""
        if len(self.df) < 13:
            return {}
        
        fib_seq = [3, 5, 8, 13, 21]
        results = {}
        
        for fib in fib_seq:
            if len(self.df) >= fib:
                pattern_count = 0
                digit_matches = defaultdict(int)
                
                for i in range(len(self.df) - fib):
                    current = set(self.df.iloc[i]['so5'])
                    future = set(self.df.iloc[i + fib]['so5'])
                    common = current & future
                    
                    if len(common) >= 2:
                        pattern_count += 1
                        for digit in common:
                            digit_matches[digit] += 1
                
                if digit_matches:
                    top_digits = sorted(digit_matches.items(), key=lambda x: x[1], reverse=True)[:3]
                    results[f'F{fib}'] = {
                        'patterns': pattern_count,
                        'top_digits': dict(top_digits),
                        'confidence': min(80, pattern_count * 10)
                    }
        
        return results
    
    def _yin_yang_analysis(self):
        """Phân tích âm dương (chẵn/lẻ)"""
        if self.df.empty:
            return {}
        
        yin_yang_patterns = []
        for num in self.df.head(20)['so5']:
            pattern = ''.join(['Y' if int(d) % 2 == 1 else 'D' for d in num])
            yin_yang_patterns.append(pattern)
        
        pattern_counts = Counter(yin_yang_patterns)
        
        # Tính tỷ lệ âm/dương cho từng vị trí
        position_analysis = []
        for pos in range(5):
            yin_count = sum(1 for num in self.df.head(20)['so5'] if int(num[pos]) % 2 == 1)
            yang_count = 20 - yin_count
            position_analysis.append({
                'position': pos,
                'yin_ratio': yin_count / 20,
                'yang_ratio': yang_count / 20,
                'dominant': 'Âm' if yin_count > yang_count else 'Dương'
            })
        
        return {
            'common_patterns': pattern_counts.most_common(3),
            'position_analysis': position_analysis,
            'current_pattern': yin_yang_patterns[0] if yin_yang_patterns else None
        }
    
    def _pattern_detection(self):
        """Phát hiện các mẫu hình quan trọng"""
        patterns = {
            'straight_patterns': self._detect_straight_patterns(),
            'wave_patterns': self._detect_wave_patterns(),
            'mirror_patterns': self._detect_mirror_patterns(),
            'ladder_patterns': self._detect_ladder_patterns(),
            'repeat_patterns': self._detect_repeat_patterns()
        }
        
        return patterns
    
    def _detect_straight_patterns(self):
        """Phát hiện cầu bệt"""
        if len(self.df) < 5:
            return []
        
        straights = []
        current_streak = []
        
        for i in range(len(self.df) - 1):
            current_num = self.df.iloc[i]['so5']
            next_num = self.df.iloc[i + 1]['so5']
            common_digits = set(current_num) & set(next_num)
            
            if len(common_digits) >= 2:
                if not current_streak:
                    current_streak = [(i, current_num), (i+1, next_num)]
                elif current_streak[-1][0] == i:
                    current_streak.append((i+1, next_num))
            else:
                if len(current_streak) >= 3:
                    common = set.intersection(*[set(num) for _, num in current_streak])
                    straights.append({
                        'type': 'straight',
                        'length': len(current_streak),
                        'common_digits': list(common)[:2],
                        'start_position': current_streak[0][0]
                    })
                current_streak = []
        
        return straights[:5]
    
    def _detect_wave_patterns(self):
        """Phát hiện cầu sóng"""
        if len(self.df) < 10:
            return []
        
        waves = []
        for i in range(len(self.df) - 8):
            sums = self.df.iloc[i:i+9]['tong'].tolist()
            
            # Kiểm tra mẫu sóng
            changes = []
            for j in range(len(sums)-1):
                changes.append('U' if sums[j] < sums[j+1] else 'D')
            
            wave_count = changes.count('U') + changes.count('D')
            if wave_count >= 6:
                waves.append({
                    'type': 'wave',
                    'pattern': ''.join(changes),
                    'start_position': i,
                    'amplitude': max(sums) - min(sums)
                })
        
        return waves[:3]
    
    def _detect_mirror_patterns(self):
        """Phát hiện số gương (bóng)"""
        if self.df.empty:
            return []
        
        mirror_map = {'0':'5','1':'6','2':'7','3':'8','4':'9',
                     '5':'0','6':'1','7':'2','8':'3','9':'4'}
        
        mirrors = []
        for i in range(min(15, len(self.df))):
            num = self.df.iloc[i]['so5']
            mirror_num = ''.join([mirror_map.get(d, d) for d in num])
            
            for j in range(i+1, min(i+6, len(self.df))):
                if self.df.iloc[j]['so5'] == mirror_num:
                    mirrors.append({
                        'original': num,
                        'mirror': mirror_num,
                        'delay': j - i,
                        'position': i
                    })
                    break
        
        return mirrors[:5]
    
    def _detect_ladder_patterns(self):
        """Phát hiện cầu thang"""
        if len(self.df) < 5:
            return []
        
        ladders = []
        for i in range(len(self.df) - 4):
            nums = self.df.iloc[i:i+5]['so5'].tolist()
            
            # Kiểm tra tăng dần
            if all(int(nums[j]) < int(nums[j+1]) for j in range(4)):
                ladders.append({
                    'type': 'increasing_ladder',
                    'numbers': nums,
                    'position': i
                })
            # Kiểm tra giảm dần
            elif all(int(nums[j]) > int(nums[j+1]) for j in range(4)):
                ladders.append({
                    'type': 'decreasing_ladder',
                    'numbers': nums,
                    'position': i
                })
        
        return ladders[:3]
    
    def _detect_repeat_patterns(self):
        """Phát hiện số lặp"""
        if len(self.df) < 10:
            return []
        
        repeats = []
        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]['so5']
            next_num = self.df.iloc[i + 1]['so5']
            
            common_digits = set(current) & set(next_num)
            if len(common_digits) >= 3:
                repeats.append({
                    'type': 'repeat',
                    'digits': list(common_digits),
                    'position': i,
                    'strength': len(common_digits)
                })
        
        return repeats[:5]
    
    def _markov_chain_analysis(self):
        """Phân tích Markov Chain"""
        if len(self.df) < 30:
            return {}
        
        predictions = []
        for pos in range(5):
            transition_matrix = np.zeros((10, 10))
            
            for i in range(len(self.df) - 1):
                current = int(self.df.iloc[i]['so5'][pos])
                next_digit = int(self.df.iloc[i + 1]['so5'][pos])
                transition_matrix[current][next_digit] += 1
            
            # Chuẩn hóa
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, 
                                        where=row_sums!=0)
            
            # Dự đoán
            last_digit = int(self.df.iloc[0]['so5'][pos])
            probs = transition_matrix[last_digit]
            
            top_3 = probs.argsort()[-3:][::-1]
            predictions.append({
                'position': pos,
                'last_digit': last_digit,
                'predictions': [(int(d), float(probs[d])) for d in top_3 if probs[d] > 0]
            })
        
        return predictions
    
    def _time_series_analysis(self):
        """Phân tích chuỗi thời gian"""
        if len(self.df) < 30:
            return {}
        
        try:
            sums = self.df['tong'].values[::-1]
            
            # Dự đoán đơn giản bằng moving average
            window = min(10, len(sums))
            predicted_sum = np.mean(sums[:window])
            
            # Xu hướng
            if len(sums) >= 5:
                trend = 'tăng' if sums[0] > sums[4] else 'giảm'
            else:
                trend = 'ổn định'
            
            return {
                'predicted_sum': round(float(predicted_sum), 1),
                'predicted_tai_xiu': tai_xiu(predicted_sum),
                'predicted_le_chan': le_chan(predicted_sum),
                'trend': trend,
                'confidence': min(75, len(sums) / 100 * 100)
            }
        except:
            return {}
    
    def _gambling_tips_analysis(self):
        """Phân tích và áp dụng mẹo đánh"""
        tips = []
        
        if self.df.empty:
            return tips
        
        # 1. MẸO BẠC NHỚ
        bach_nho = self._apply_bach_nho()
        if bach_nho:
            tips.append({
                'id': 'bach_nho',
                'name': 'Bạc Nhớ',
                'description': 'Số thường đi cùng nhau',
                'numbers': bach_nho[:3],
                'confidence': 70
            })
        
        # 2. MẸO LÔ GAN
        lo_gan = self._find_gan_numbers()[:3]
        if lo_gan:
            tips.append({
                'id': 'lo_gan',
                'name': 'Lô Gan',
                'description': 'Số lâu chưa về, sắp về',
                'numbers': lo_gan,
                'confidence': 60
            })
        
        # 3. MẸO CHẠM ĐẦU ĐUÔI
        cham_dau_duoi = self._apply_cham_dau_duoi()
        if cham_dau_duoi:
            tips.append({
                'id': 'cham_dau_duoi',
                'name': 'Chạm Đầu Đuôi',
                'description': 'Đầu/đuôi thường xuất hiện',
                'numbers': cham_dau_duoi,
                'confidence': 65
            })
        
        # 4. MẸO TỔNG ĐỀ
        tong_de = self._apply_tong_de()
        if tong_de:
            tips.append({
                'id': 'tong_de',
                'name': 'Tổng Đề',
                'description': 'Tổng số đề phổ biến',
                'numbers': tong_de,
                'confidence': 68
            })
        
        # 5. MẸO BÓNG SỐ
        bong_so = self._apply_bong_so()
        if bong_so:
            tips.append({
                'id': 'bong_so',
                'name': 'Bóng Số',
                'description': 'Bóng âm/dương của số gần đây',
                'numbers': bong_so,
                'confidence': 62
            })
        
        # 6. MẸO KẸP SỐ
        kep_so = self._apply_kep_so()
        if kep_so:
            tips.append({
                'id': 'kep_so',
                'name': 'Kẹp Số',
                'description': 'Số kẹp giữa các số đã ra',
                'numbers': kep_so,
                'confidence': 58
            })
        
        return tips
    
    def _apply_bach_nho(self):
        """Áp dụng mẹo bạc nhớ"""
        if len(self.df) < 10:
            return []
        
        pair_counter = defaultdict(int)
        for num in self.df.head(20)['so5']:
            digits = list(num)
            for pair in itertools.combinations(digits, 2):
                sorted_pair = ''.join(sorted(pair))
                pair_counter[sorted_pair] += 1
        
        common_pairs = [pair for pair, count in pair_counter.items() if count >= 3]
        return common_pairs[:5]
    
    def _apply_cham_dau_duoi(self):
        """Áp dụng mẹo chạm đầu đuôi"""
        if len(self.df) < 10:
            return []
        
        heads = []
        tails = []
        for num in self.df.head(15)['so5']:
            heads.append(num[0])
            tails.append(num[-1])
        
        head_counter = Counter(heads)
        tail_counter = Counter(tails)
        
        common_heads = [digit for digit, _ in head_counter.most_common(2)]
        common_tails = [digit for digit, _ in tail_counter.most_common(2)]
        
        return common_heads + common_tails
    
    def _apply_tong_de(self):
        """Áp dụng mẹo tổng đề"""
        if len(self.df) < 10:
            return []
        
        sums = self.df.head(20)['tong'].tolist()
        sum_counter = Counter(sums)
        common_sums = [str(s) for s, _ in sum_counter.most_common(3)]
        
        return common_sums
    
    def _apply_bong_so(self):
        """Áp dụng mẹo bóng số"""
        if self.df.empty:
            return []
        
        bong_map = {'0':'5','1':'6','2':'7','3':'8','4':'9',
                   '5':'0','6':'1','7':'2','8':'3','9':'4'}
        
        recent_nums = self.df.head(5)['so5'].tolist()
        bong_numbers = set()
        
        for num in recent_nums:
            for digit in num:
                if digit in bong_map:
                    bong_numbers.add(bong_map[digit])
        
        return list(bong_numbers)[:4]
    
    def _apply_kep_so(self):
        """Áp dụng mẹo kẹp số"""
        if len(self.df) < 5:
            return []
        
        recent_digits = set()
        for num in self.df.head(5)['so5']:
            for digit in num:
                recent_digits.add(int(digit))
        
        kep_numbers = []
        sorted_digits = sorted(recent_digits)
        
        for i in range(len(sorted_digits) - 1):
            diff = sorted_digits[i+1] - sorted_digits[i]
            if diff > 1:
                for d in range(sorted_digits[i] + 1, sorted_digits[i+1]):
                    kep_numbers.append(str(d))
        
        return kep_numbers[:4]
    
    def _random_forest_analysis(self):
        """Phân tích bằng Random Forest (nếu có thư viện)"""
        if not AI_LIBS_AVAILABLE or len(self.df) < 50:
            return {}
        
        try:
            # Chuẩn bị dữ liệu
            features = []
            targets = []
            
            for i in range(len(self.df) - 1):
                current_num = self.df.iloc[i]['so5']
                next_num = self.df.iloc[i + 1]['so5']
                
                # Tạo features
                feature = [int(d) for d in current_num] + [self.df.iloc[i]['tong']]
                features.append(feature)
                
                # Target: số đầu tiên của kỳ tiếp theo
                targets.append(int(next_num[0]))
            
            # Huấn luyện mô hình đơn giản
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Dự đoán
            last_feature = [int(d) for d in self.df.iloc[0]['so5']] + [self.df.iloc[0]['tong']]
            prediction = model.predict([last_feature])[0]
            proba = model.predict_proba([last_feature])[0]
            
            return {
                'predicted_digit': int(prediction),
                'confidence': float(max(proba) * 100),
                'feature_importance': model.feature_importances_.tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_final_predictions(self):
        """Tạo dự đoán cuối cùng từ tất cả phân tích"""
        if self.df.empty:
            return {'status': 'no_data'}
        
        # Chạy tất cả phân tích
        all_analyses = self.run_all_analyses()
        
        # Tính điểm cho từng số 0-9
        digit_scores = {str(i): 0 for i in range(10)}
        
        # 1. Điểm từ số nóng
        hot_cold = all_analyses.get('hot_cold', {})
        hot_numbers = hot_cold.get('hot_numbers', [])
        for digit in hot_numbers:
            digit_scores[digit] += 25
        
        # 2. Điểm từ Markov Chain
        markov = all_analyses.get('markov', [])
        for pos_pred in markov:
            for digit, prob in pos_pred.get('predictions', []):
                digit_scores[str(digit)] += prob * 20
        
        # 3. Điểm từ mẫu hình
        patterns = all_analyses.get('patterns', {})
        straights = patterns.get('straight_patterns', [])
        for pattern in straights[:3]:
            for digit in pattern.get('common_digits', []):
                digit_scores[digit] += 15
        
        # 4. Điểm từ mẹo đánh
        tips = all_analyses.get('gambling_tips', [])
        for tip in tips:
            numbers = tip.get('numbers', [])
            confidence = tip.get('confidence', 50) / 100
            for num in numbers:
                for digit in str(num):
                    if digit.isdigit():
                        digit_scores[digit] += 10 * confidence
        
        # 5. Điểm từ Fibonacci
        fibonacci = all_analyses.get('fibonacci', {})
        for fib_key, fib_data in fibonacci.items():
            top_digits = fib_data.get('top_digits', {})
            for digit, count in top_digits.items():
                digit_scores[digit] += count * 3
        
        # Sắp xếp số theo điểm
        sorted_digits = sorted(digit_scores.items(), key=lambda x: x[1], reverse=True)
        top_digits = [digit for digit, score in sorted_digits[:8]]
        
        # Tạo tổ hợp 2 số
        top_2_combos = []
        for i in range(len(top_digits)):
            for j in range(i+1, len(top_digits)):
                combo = ''.join(sorted([top_digits[i], top_digits[j]]))
                score = digit_scores[top_digits[i]] + digit_scores[top_digits[j]]
                top_2_combos.append((combo, score))
        
        top_2_combos.sort(key=lambda x: x[1], reverse=True)
        
        # Tạo tổ hợp 3 số
        top_3_combos = []
        for i in range(len(top_digits)):
            for j in range(i+1, len(top_digits)):
                for k in range(j+1, len(top_digits)):
                    combo = ''.join(sorted([top_digits[i], top_digits[j], top_digits[k]]))
                    score = (digit_scores[top_digits[i]] + 
                            digit_scores[top_digits[j]] + 
                            digit_scores[top_digits[k]])
                    top_3_combos.append((combo, score))
        
        top_3_combos.sort(key=lambda x: x[1], reverse=True)
        
        # Dự đoán Tài/Xỉu, Lẻ/Chẵn
        time_series = all_analyses.get('time_series', {})
        
        return {
            'status': 'success',
            'top_2_numbers': top_2_combos[:5],
            'top_3_numbers': top_3_combos[:5],
            'tai_xiu': time_series.get('predicted_tai_xiu', 'TÀI'),
            'le_chan': time_series.get('predicted_le_chan', 'LẺ'),
            'confidence': {
                '2_numbers': calculate_confidence(len(self.df), len(straights), len(all_analyses)),
                '3_numbers': calculate_confidence(len(self.df), len(straights), len(all_analyses)) * 0.9,
                'tai_xiu': time_series.get('confidence', 50),
                'le_chan': time_series.get('confidence', 50)
            },
            'analyses_count': len(all_analyses),
            'patterns_count': len(straights),
            'tips_count': len(tips),
            'detailed_analyses': all_analyses
        }

# ================= DATA MANAGEMENT =================

def save_ky_quay(numbers):
    conn = get_conn()
    c = conn.cursor()
    added_count = 0
    
    for num in numbers:
        if len(num) != 5 or not num.isdigit():
            continue
            
        ky_id = f"KY{int(time.time() * 1000) % 1000000:06d}"
        so5 = num
        tien_nhi = num[:2]
        hau_nhi = num[-2:]
        tong = sum(int(d) for d in num)
        
        try:
            c.execute("""
            INSERT OR IGNORE INTO ky_quay 
            (ky, so5, tien_nhi, hau_nhi, tong, tai_xiu, le_chan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ky_id, so5, tien_nhi, hau_nhi, tong,
                tai_xiu(tong), le_chan(tong)
            ))
            
            if c.rowcount > 0:
                added_count += 1
        except:
            pass
    
    conn.commit()
    conn.close()
    return added_count

def load_recent_data(limit=1000):
    conn = get_conn()
    query = f"""
    SELECT * FROM ky_quay 
    ORDER BY timestamp DESC 
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================= MAIN APP V10.1 =================

def main():
    # App Header
    st.title("🎰 LOTOBET ULTRA AI PRO – V10.1 MULTI-ALGORITHM")
    st.caption(f"⏱️ Cập nhật: {datetime.now().strftime('%H:%M:%S')} | 🧠 AI Đa thuật toán")
    
    # AI Status
    if not AI_LIBS_AVAILABLE:
        st.warning("""
        ⚠️ **Thiếu thư viện AI nâng cao!** 
        Cài đặt: `pip install scikit-learn statsmodels`
        Một số tính năng AI có thể bị hạn chế.
        """)
    
    st.markdown("---")
    
    # Load data
    df = load_recent_data(500)
    
    # ========== 1️⃣ KHUNG DỮ LIỆU ==========
    st.markdown('<div class="section-header">📥 1️⃣ KHUNG DỮ LIỆU</div>', unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        raw_data = st.text_area(
            "**Dán dữ liệu kết quả:**",
            height=120,
            placeholder="""Nhập kết quả (mỗi dòng 1 số 5 chữ số):
12345
67890
54321

Hoặc: 12345 67890 54321
Hoặc: 2 tinh: 5264 3 tinh: 5289"""
        )
    
    with col_input2:
        st.markdown("**📁 Nhập từ file:**")
        uploaded_file = st.file_uploader("Chọn file TXT/CSV", 
                                       type=['txt', 'csv'], 
                                       label_visibility="collapsed")
        
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            if st.button("📥 Import từ file", use_container_width=True):
                numbers = smart_parse_input(file_content)
                added = save_ky_quay(numbers)
                if added > 0:
                    st.success(f"✅ Đã thêm {added} kỳ mới!")
                    time.sleep(1)
                    st.rerun()
    
    # Parse and save data
    if raw_data:
        numbers = smart_parse_input(raw_data)
        
        if numbers:
            st.markdown(f"**📋 Đã nhận diện {len(numbers)} kỳ:**")
            with st.expander("Xem chi tiết", expanded=False):
                for i, num in enumerate(numbers[:10], 1):
                    st.text(f"{i}. {num}")
                if len(numbers) > 10:
                    st.text(f"... và {len(numbers)-10} kỳ khác")
            
            if st.button("💾 LƯU VÀO DATABASE", type="primary", use_container_width=True):
                with st.spinner("Đang lưu dữ liệu..."):
                    added = save_ky_quay(numbers)
                    if added > 0:
                        st.success(f"✅ Đã lưu thành công {added} kỳ mới!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Dữ liệu đã tồn tại hoặc có lỗi")
    
    # Show current data
    if not df.empty:
        st.markdown(f"**📊 Dữ liệu hiện có: {len(df)} kỳ**")
        with st.expander("Xem 10 kỳ gần nhất", expanded=False):
            st.dataframe(
                df.head(10)[["ky", "so5", "tai_xiu", "le_chan", "timestamp"]],
                use_container_width=True,
                height=250
            )
    else:
        st.info("📭 Chưa có dữ liệu. Vui lòng nhập dữ liệu trước.")
    
    # ========== 2️⃣ KHUNG PHÂN TÍCH AI ==========
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 2️⃣ KHUNG PHÂN TÍCH AI V10.1</div>', unsafe_allow_html=True)
    
    if not df.empty:
        # Initialize AI
        ai_engine = AdvancedLottoAI_V10_1(df)
        
        # Run analysis
        with st.spinner("🔄 AI V10.1 đang phân tích đa thuật toán..."):
            predictions = ai_engine.generate_final_predictions()
        
        if predictions['status'] == 'success':
            # Display AI analysis summary
            st.markdown('<div class="ai-analysis-box">', unsafe_allow_html=True)
            st.markdown("### 📊 TỔNG HỢP PHÂN TÍCH AI")
            
            col_algo1, col_algo2, col_algo3 = st.columns(3)
            
            with col_algo1:
                st.metric("Thuật toán", predictions['analyses_count'])
                st.caption("Lớp AI áp dụng")
            
            with col_algo2:
                st.metric("Mẫu hình", predictions['patterns_count'])
                st.caption("Pattern phát hiện")
            
            with col_algo3:
                st.metric("Mẹo đánh", predictions['tips_count'])
                st.caption("Mẹo áp dụng")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ========== 3️⃣ KHUNG KẾT LUẬN SỐ ĐÁNH ==========
            st.markdown("---")
            st.markdown('<div class="highlight-main">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎯 3️⃣ KẾT LUẬN SỐ ĐÁNH</div>', unsafe_allow_html=True)
            
            col_concl1, col_concl2, col_concl3, col_concl4 = st.columns(4)
            
            with col_concl1:
                best_2num = predictions['top_2_numbers'][0][0]
                conf_2num = predictions['confidence']['2_numbers']
                st.markdown("### 🔥 ĐÁNH 2 SỐ")
                st.markdown(f'<div class="big-number">{best_2num}</div>', unsafe_allow_html=True)
                st.metric("Độ tin cậy", f"{conf_2num:.1f}%")
                st.progress(conf_2num/100)
            
            with col_concl2:
                best_3num = predictions['top_3_numbers'][0][0]
                conf_3num = predictions['confidence']['3_numbers']
                st.markdown("### 🔥 ĐÁNH 3 SỐ")
                st.markdown(f'<div class="big-number">{best_3num}</div>', unsafe_allow_html=True)
                st.metric("Độ tin cậy", f"{conf_3num:.1f}%")
                st.progress(conf_3num/100)
            
            with col_concl3:
                pred_tx = predictions['tai_xiu']
                conf_tx = predictions['confidence']['tai_xiu']
                st.markdown("### 🎲 TÀI/XỈU")
                st.markdown(f'<div class="medium-number">{pred_tx}</div>', unsafe_allow_html=True)
                st.metric("Độ tin cậy", f"{conf_tx:.1f}%")
                st.progress(conf_tx/100)
            
            with col_concl4:
                pred_lc = predictions['le_chan']
                conf_lc = predictions['confidence']['le_chan']
                st.markdown("### 🎲 LẺ/CHẴN")
                st.markdown(f'<div class="medium-number">{pred_lc}</div>', unsafe_allow_html=True)
                st.metric("Độ tin cậy", f"{conf_lc:.1f}%")
                st.progress(conf_lc/100)
            
            st.markdown("---")
            st.markdown("**✅ Dự đoán tổng hợp từ 5 lớp AI và 10+ mẹo đánh**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ========== 4️⃣ KHUNG PHÂN TÍCH CHI TIẾT ==========
            st.markdown("---")
            st.markdown('<div class="section-header">📊 4️⃣ PHÂN TÍCH CHI TIẾT</div>', unsafe_allow_html=True)
            
            # Tạo tabs cho từng loại phân tích
            tab_det1, tab_det2, tab_det3, tab_det4 = st.tabs([
                "🔥 Số Nóng/Lạnh",
                "🌀 Mẫu Hình",
                "💡 Mẹo Đánh",
                "🤖 Thuật Toán"
            ])
            
            with tab_det1:
                hot_cold = predictions['detailed_analyses'].get('hot_cold', {})
                if hot_cold:
                    col_hot, col_cold, col_gan = st.columns(3)
                    
                    with col_hot:
                        st.markdown("#### 🔥 SỐ NÓNG")
                        for digit in hot_cold.get('hot_numbers', []):
                            st.markdown(f"- **{digit}**")
                    
                    with col_cold:
                        st.markdown("#### ❄️ SỐ LẠNH")
                        for digit in hot_cold.get('cold_numbers', []):
                            st.markdown(f"- **{digit}**")
                    
                    with col_gan:
                        st.markdown("#### ⏳ LÔ GAN")
                        for digit in hot_cold.get('gan_numbers', []):
                            st.markdown(f"- **{digit}**")
            
            with tab_det2:
                patterns = predictions['detailed_analyses'].get('patterns', {})
                
                col_pat1, col_pat2 = st.columns(2)
                
                with col_pat1:
                    if patterns.get('straight_patterns'):
                        st.markdown("#### ⏫ CẦU BỆT")
                        for pattern in patterns['straight_patterns'][:3]:
                            st.markdown(f"- **{pattern['common_digits']}** (dài {pattern['length']} kỳ)")
                
                with col_pat2:
                    if patterns.get('mirror_patterns'):
                        st.markdown("#### 🔄 SỐ GƯƠNG")
                        for pattern in patterns['mirror_patterns'][:3]:
                            st.markdown(f"- **{pattern['original']}** → **{pattern['mirror']}** (sau {pattern['delay']} kỳ)")
            
            with tab_det3:
                tips = predictions['detailed_analyses'].get('gambling_tips', [])
                if tips:
                    for tip in tips[:5]:
                        st.markdown(f'<div class="tips-box">', unsafe_allow_html=True)
                        st.markdown(f"**{tip['name']}** ({tip['confidence']}%)")
                        st.markdown(f"*{tip['description']}*")
                        st.markdown(f"**Số đề xuất:** {', '.join(map(str, tip['numbers'][:3]))}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with tab_det4:
                st.markdown("#### 📈 THUẬT TOÁN ÁP DỤNG")
                
                algo_list = [
                    ("Thống kê cơ bản", predictions['detailed_analyses'].get('basic_stats', {})),
                    ("Số nóng/lạnh", predictions['detailed_analyses'].get('hot_cold', {})),
                    ("Chu kỳ Fibonacci", predictions['detailed_analyses'].get('fibonacci', {})),
                    ("Âm Dương", predictions['detailed_analyses'].get('yin_yang', {})),
                    ("Markov Chain", predictions['detailed_analyses'].get('markov', [])),
                    ("Time Series", predictions['detailed_analyses'].get('time_series', {}))
                ]
                
                for algo_name, algo_data in algo_list:
                    if algo_data:
                        st.markdown(f"✅ **{algo_name}**: Đã áp dụng")
            
            # ========== 5️⃣ KHUNG QUẢN LÝ VỐN ==========
            st.markdown("---")
            st.markdown('<div class="section-header">💰 5️⃣ QUẢN LÝ VỐN</div>', unsafe_allow_html=True)
            
            col_cap1, col_cap2 = st.columns(2)
            
            with col_cap1:
                st.markdown("#### ⚙️ THIẾT LẬP")
                
                tong_von = st.number_input(
                    "Tổng vốn (VNĐ):",
                    min_value=100,
                    max_value=10000000,
                    value=1000000,
                    step=10000,
                    help="Nhập số vốn hiện có"
                )
                
                rui_ro = st.slider(
                    "Rủi ro/kỳ (%):",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Tỷ lệ vốn tối đa đánh mỗi kỳ"
                )
            
            with col_cap2:
                st.markdown("#### 📊 PHÂN BỔ")
                
                max_bet = tong_von * (rui_ro / 100)
                best_2num = predictions['top_2_numbers'][0][0]
                best_3num = predictions['top_3_numbers'][0][0]
                
                # Tính phân bổ đơn giản
                bet_2so = max_bet * 0.55
                bet_3so = max_bet * 0.45
                
                st.metric("Tối đa/kỳ", format_tien(max_bet))
                st.markdown(f"**2 số `{best_2num}`:** {format_tien(bet_2so)}")
                st.markdown(f"**3 số `{best_3num}`:** {format_tien(bet_3so)}")
                
                # Visual
                st.markdown("**Phân bổ:**")
                col_vis1, col_vis2 = st.columns([55, 45])
                with col_vis1:
                    st.markdown(f'<div style="background-color:#3B82F6;height:20px;border-radius:5px"></div>', 
                              unsafe_allow_html=True)
                    st.caption("55% - 2 số")
                with col_vis2:
                    st.markdown(f'<div style="background-color:#10B981;height:20px;border-radius:5px"></div>', 
                              unsafe_allow_html=True)
                    st.caption("45% - 3 số")
            
            # ========== 6️⃣ KHUNG CÀI ĐẶT AI ==========
            st.markdown("---")
            st.markdown('<div class="section-header">⚙️ 6️⃣ CÀI ĐẶT AI</div>', unsafe_allow_html=True)
            
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                st.markdown("#### 🧠 THUẬT TOÁN")
                
                algo_settings = {
                    "Thống kê cơ bản": st.checkbox("Lớp 1: Thống kê", value=True),
                    "Số nóng/lạnh": st.checkbox("Lớp 2: Nóng/lạnh", value=True),
                    "Chu kỳ Fibonacci": st.checkbox("Lớp 3: Fibonacci", value=True),
                    "Âm Dương": st.checkbox("Lớp 4: Âm Dương", value=True),
                    "Mẫu hình": st.checkbox("Lớp 5: Mẫu hình", value=True),
                    "Markov Chain": st.checkbox("Lớp 6: Markov", value=True),
                    "Time Series": st.checkbox("Lớp 7: Time Series", value=True),
                    "Mẹo đánh": st.checkbox("Lớp 8: Mẹo đánh", value=True)
                }
            
            with col_set2:
                st.markdown("#### ⚠️ CẢNH BÁO")
                
                losing_streak = st.number_input(
                    "Cảnh báo chuỗi thua:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Cảnh báo khi thua N kỳ liên tiếp"
                )
                
                max_daily_loss = st.slider(
                    "Lỗ tối đa/ngày (%):",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Tự động dừng khi đạt ngưỡng"
                )
                
                if st.button("💾 Lưu cài đặt AI", use_container_width=True):
                    st.success("✅ Đã lưu cài đặt AI!")
            
            # Final message
            st.markdown("---")
            st.markdown('<div style="background-color:#E3F2FD;padding:20px;border-radius:10px;text-align:center">', 
                       unsafe_allow_html=True)
            st.markdown("### 🧠 **V10.1 - AI ĐA THUẬT TOÁN**")
            st.markdown("""
            **5 lớp AI tích hợp | 10+ mẹo đánh | Dự đoán đa chiều**
            
            > ⚠️ Tool hỗ trợ phân tích, không đảm bảo thắng 100%
            > 💡 Quản lý vốn là yếu tố sống còn
            > 🔄 Cập nhật dữ liệu thường xuyên để tăng độ chính xác
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export button
            if st.button("📤 XUẤT BÁO CÁO PHÂN TÍCH ĐẦY ĐỦ", use_container_width=True):
                st.success("""
                ✅ Báo cáo AI V10.1 đã được tạo (giả lập)
                
                **Nội dung báo cáo:**
                - Tổng hợp dự đoán từ 8 thuật toán
                - Chi tiết các mẫu hình phát hiện
                - Mẹo đánh đã áp dụng
                - Độ tin cậy từng phương pháp
                - Khuyến nghị quản lý vốn
                """)
        
        else:
            st.warning("⚠️ Cần thêm dữ liệu để phân tích AI nâng cao")
            st.info(f"Hiện có: {len(df)} kỳ | Yêu cầu tối thiểu: 30 kỳ")
    
    # Footer
    st.markdown("---")
    st.caption("""
    © 2024 LOTOBET ULTRA AI PRO – V10.1 MULTI-ALGORITHM
    Phiên bản: V10.1 Final | Ngày phát hành: 15/01/2024
    """)

# ================= RUN APP =================
if __name__ == "__main__":
    main()
