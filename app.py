import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timedelta
import os
import random
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE AI MASTER - 2 SỐ 5 TÍNH",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "numcore_data.csv"
AI_CONFIG_FILE = "ai_config.json"
PAIR_HISTORY_FILE = "pair_history.json"

# ================= ENHANCED AI FOR 2-NUMBER PAIRS =================
class EnhancedTwoNumberAI:
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.history_window = 30
        self.pair_frequency = Counter()
        self.position_pairs = defaultdict(Counter)
        self.consecutive_pairs = Counter()
        self.repeated_numbers = Counter()
        self.load_config()
        self.load_pair_history()
    
    def load_config(self):
        """Load AI configuration with error handling"""
        default_config = {
            "algorithm_weights": {
                "frequency_based": 0.20,
                "gap_analysis": 0.18,
                "hot_cold_mix": 0.18,
                "pattern_based": 0.15,
                "position_based": 0.15,
                "trend_based": 0.14
            },
            "recent_weight": 0.65,
            "avoid_recent_pairs": 5,
            "min_confidence": 65,
            "avoid_same_digits": True,
            "max_consecutive_gap": 3,
            "prefer_complementary": True
        }
        
        try:
            if os.path.exists(AI_CONFIG_FILE):
                with open(AI_CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    self.config = {**default_config, **loaded_config}
            else:
                self.config = default_config
        except:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save AI configuration"""
        try:
            with open(AI_CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def load_pair_history(self):
        """Load historical pair data with error handling"""
        try:
            if os.path.exists(PAIR_HISTORY_FILE):
                with open(PAIR_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    
                    self.pair_frequency = Counter(data.get('pair_frequency', {}))
                    self.consecutive_pairs = Counter(data.get('consecutive_pairs', {}))
                    self.repeated_numbers = Counter(data.get('repeated_numbers', {}))
                    
                    pos_pairs_data = data.get('position_pairs', {})
                    self.position_pairs = defaultdict(Counter)
                    for key_str, counter_data in pos_pairs_data.items():
                        try:
                            key = eval(key_str)
                            self.position_pairs[key] = Counter(counter_data)
                        except:
                            continue
            else:
                self.reset_statistics()
        except:
            self.reset_statistics()
    
    def save_pair_history(self):
        """Save pair history data with safe serialization"""
        try:
            data = {
                'pair_frequency': dict(self.pair_frequency),
                'consecutive_pairs': dict(self.consecutive_pairs),
                'repeated_numbers': dict(self.repeated_numbers),
                'position_pairs': {str(k): dict(v) for k, v in self.position_pairs.items()}
            }
            
            with open(PAIR_HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.pair_frequency = Counter()
        self.consecutive_pairs = Counter()
        self.repeated_numbers = Counter()
        self.position_pairs = defaultdict(Counter)
    
    def extract_pairs_from_history(self, numbers_history):
        """Extract all valid 2-number pairs from history (excluding same digits)"""
        all_pairs = []
        
        for numbers in numbers_history:
            if len(numbers) == 5:
                unique_numbers = set(numbers)
                
                has_repeated = any(numbers.count(num) > 1 for num in unique_numbers)
                if has_repeated:
                    valid_numbers = [num for num in unique_numbers]
                    for pair in combinations(sorted(valid_numbers), 2):
                        if pair[0] != pair[1]:
                            all_pairs.append(pair)
                else:
                    for pair in combinations(sorted(unique_numbers), 2):
                        if pair[0] != pair[1]:
                            all_pairs.append(pair)
        
        return all_pairs
    
    def analyze_number_patterns(self, numbers_history):
        """Analyze patterns in numbers including consecutive numbers and repeats"""
        if not numbers_history:
            return {}
        
        patterns = {
            'consecutive_pairs': Counter(),
            'digit_gaps': [],
            'repeated_numbers': Counter(),
            'position_analysis': {i: Counter() for i in range(5)}
        }
        
        for numbers in numbers_history[-20:]:
            if len(numbers) == 5:
                for i in range(4):
                    if abs(numbers[i] - numbers[i+1]) == 1:
                        pair = tuple(sorted([numbers[i], numbers[i+1]]))
                        patterns['consecutive_pairs'][pair] += 1
                
                num_counter = Counter(numbers)
                for num, count in num_counter.items():
                    if count > 1:
                        patterns['repeated_numbers'][num] += 1
                
                for pos, num in enumerate(numbers):
                    patterns['position_analysis'][pos][num] += 1
        
        return patterns
    
    def update_pair_statistics(self, numbers_history):
        """Update pair frequency statistics"""
        if len(numbers_history) == 0:
            return
        
        recent_history = numbers_history[-15:] if len(numbers_history) >= 15 else numbers_history
        
        new_pairs = self.extract_pairs_from_history(recent_history)
        for pair in new_pairs:
            self.pair_frequency[pair] += 1
        
        patterns = self.analyze_number_patterns(recent_history)
        self.consecutive_pairs.update(patterns['consecutive_pairs'])
        self.repeated_numbers.update(patterns['repeated_numbers'])
        
        for numbers in recent_history:
            if len(numbers) == 5:
                for i in range(5):
                    for j in range(i+1, 5):
                        pos_pair = (i, j, numbers[i], numbers[j])
                        key = (numbers[i], numbers[j])
                        self.position_pairs[key][pos_pair] += 1
        
        self.save_pair_history()
    
    # ============= ENHANCED ALGORITHMS =============
    
    def algorithm_frequency_based(self, numbers_history, hot_numbers):
        """Algorithm 1: Enhanced frequency-based prediction"""
        if len(numbers_history) < 10:
            return []
        
        all_pairs = self.extract_pairs_from_history(numbers_history)
        if not all_pairs:
            return []
        
        pair_counter = Counter(all_pairs)
        frequent_pairs = pair_counter.most_common(20)
        
        pair_scores = {}
        recent_pairs = self.extract_pairs_from_history(numbers_history[-5:])
        
        for pair, freq in frequent_pairs:
            score = freq * 0.4
            
            recent_count = recent_pairs.count(pair)
            score *= (1 + recent_count * 0.3)
            
            hot_bonus = 0
            for num in pair:
                if num in hot_numbers[:3]:
                    hot_bonus += 0.2
                elif num in hot_numbers:
                    hot_bonus += 0.1
            score *= (1 + hot_bonus)
            
            if abs(pair[0] - pair[1]) == 1:
                score *= 0.8
            
            if abs(pair[0] - pair[1]) > 5:
                score *= 0.7
            
            pair_scores[pair] = score
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def algorithm_gap_analysis(self, numbers_history):
        """Algorithm 2: Gap analysis between numbers"""
        if len(numbers_history) < 15:
            return []
        
        gap_distribution = Counter()
        successful_pairs = []
        
        for numbers in numbers_history[-15:]:
            if len(numbers) == 5:
                unique_numbers = sorted(set(numbers))
                for i in range(len(unique_numbers)):
                    for j in range(i+1, len(unique_numbers)):
                        gap = abs(unique_numbers[i] - unique_numbers[j])
                        gap_distribution[gap] += 1
                        successful_pairs.append((unique_numbers[i], unique_numbers[j]))
        
        optimal_gaps = []
        for gap, count in gap_distribution.most_common(5):
            if 1 <= gap <= 4:
                optimal_gaps.append(gap)
        
        pair_scores = {}
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                gap = abs(num1 - num2)
                if gap in optimal_gaps:
                    score = gap_distribution.get(gap, 1) * 0.3
                    
                    pair = (num1, num2)
                    if pair in successful_pairs:
                        score *= 1.5
                    
                    if gap == 1:
                        score *= 0.6
                    
                    pair_scores[pair] = score
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def algorithm_hot_cold_mix(self, numbers_history, hot_numbers, cold_numbers):
        """Algorithm 3: Mix hot and cold numbers"""
        if len(numbers_history) < 12:
            return []
        
        pair_scores = {}
        
        for hot_num in hot_numbers[:3]:
            for cold_num in cold_numbers[:3]:
                if hot_num == cold_num:
                    continue
                
                pair = tuple(sorted([hot_num, cold_num]))
                
                together_count = sum(1 for nums in numbers_history[-10:] 
                                   if hot_num in nums and cold_num in nums)
                
                score = 0.5
                
                if together_count > 0:
                    score *= (1 + together_count * 0.2)
                else:
                    score *= 1.3
                
                gap = abs(hot_num - cold_num)
                if 2 <= gap <= 4:
                    score *= 1.2
                elif gap == 1:
                    score *= 0.7
                
                pair_scores[pair] = score
        
        for i in range(len(cold_numbers[:3])):
            for j in range(i+1, len(cold_numbers[:3])):
                cold1, cold2 = cold_numbers[i], cold_numbers[j]
                pair = tuple(sorted([cold1, cold2]))
                
                score = 0.4
                
                last_appearance1 = self.get_last_appearance(cold1, numbers_history)
                last_appearance2 = self.get_last_appearance(cold2, numbers_history)
                
                if last_appearance1 > 5 or last_appearance2 > 5:
                    score *= 1.4
                
                pair_scores[pair] = score
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def get_last_appearance(self, number, numbers_history):
        """Get how many draws since last appearance"""
        for i, numbers in enumerate(reversed(numbers_history)):
            if number in numbers:
                return i
        return len(numbers_history)
    
    def algorithm_pattern_based_enhanced(self, numbers_history):
        """Algorithm 4: Enhanced pattern recognition"""
        if len(numbers_history) < 20:
            return []
        
        patterns = self.analyze_number_patterns(numbers_history)
        
        pair_scores = {}
        
        for pair, count in patterns['consecutive_pairs'].items():
            if count >= 2:
                score = count * 0.4
                score *= 0.7
                pair_scores[pair] = score
        
        sum_distribution = Counter()
        for numbers in numbers_history[-15:]:
            if len(numbers) == 5:
                unique_numbers = sorted(set(numbers))
                for i in range(len(unique_numbers)):
                    for j in range(i+1, len(unique_numbers)):
                        pair_sum = unique_numbers[i] + unique_numbers[j]
                        sum_distribution[pair_sum] += 1
        
        optimal_sums = [sum_val for sum_val, count in sum_distribution.most_common(3)]
        
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                pair = (num1, num2)
                pair_sum = num1 + num2
                
                if pair_sum in optimal_sums:
                    if pair not in pair_scores:
                        pair_scores[pair] = 0
                    pair_scores[pair] += sum_distribution.get(pair_sum, 1) * 0.3
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def algorithm_position_based_enhanced(self, numbers_history):
        """Algorithm 5: Enhanced position-based prediction"""
        if len(numbers_history) < 15:
            return []
        
        position_stats = {i: Counter() for i in range(5)}
        
        for numbers in numbers_history[-15:]:
            if len(numbers) == 5:
                for pos, num in enumerate(numbers):
                    position_stats[pos][num] += 1
        
        position_combinations = defaultdict(Counter)
        
        for numbers in numbers_history[-10:]:
            if len(numbers) == 5:
                for i in range(5):
                    for j in range(i+1, 5):
                        key = (i, j)
                        pair = tuple(sorted([numbers[i], numbers[j]]))
                        position_combinations[key][pair] += 1
        
        pair_scores = {}
        
        for (pos1, pos2), pair_counter in position_combinations.items():
            for pair, count in pair_counter.most_common(5):
                if count >= 2:
                    total_pos1 = max(1, sum(position_stats[pos1].values()))
                    total_pos2 = max(1, sum(position_stats[pos2].values()))
                    
                    pos1_strength = position_stats[pos1][pair[0]] / total_pos1
                    pos2_strength = position_stats[pos2][pair[1]] / total_pos2
                    
                    score = count * (pos1_strength + pos2_strength) * 0.4
                    
                    if pair not in pair_scores or score > pair_scores[pair]:
                        pair_scores[pair] = score
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def algorithm_trend_based_enhanced(self, numbers_history):
        """Algorithm 6: Enhanced trend analysis"""
        if len(numbers_history) < 20:
            return []
        
        momentum_scores = {}
        
        for num in range(10):
            recent = numbers_history[-5:] if len(numbers_history) >= 10 else numbers_history[-len(numbers_history)//2:]
            older = numbers_history[-10:-5] if len(numbers_history) >= 10 else numbers_history[:len(numbers_history)//2]
            
            recent_count = sum(1 for nums in recent for n in nums if n == num)
            older_count = sum(1 for nums in older for n in nums if n == num)
            
            total = recent_count + older_count
            if total > 0:
                momentum = (recent_count - older_count) / total
                momentum_scores[num] = momentum
        
        pair_scores = {}
        
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                if num1 in momentum_scores and num2 in momentum_scores:
                    momentum_diff = abs(momentum_scores[num1] - momentum_scores[num2])
                    
                    both_rising = momentum_scores[num1] > 0.3 and momentum_scores[num2] > 0.3
                    
                    if momentum_diff > 0.4 or both_rising:
                        score = momentum_diff * 0.5 if not both_rising else 0.6
                        
                        recent_together = sum(1 for nums in numbers_history[-8:] 
                                           if num1 in nums and num2 in nums)
                        
                        if recent_together > 0:
                            score *= (1 + recent_together * 0.2)
                        
                        pair_scores[(num1, num2)] = score
        
        if pair_scores:
            return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return []
    
    def combine_algorithms(self, numbers_history, hot_numbers, cold_numbers):
        """Combine results from all enhanced algorithms"""
        if len(numbers_history) < 10:
            return []
        
        algo_results = {}
        
        try:
            algo_results['frequency'] = self.algorithm_frequency_based(numbers_history, hot_numbers)
        except:
            algo_results['frequency'] = []
        
        try:
            algo_results['gap'] = self.algorithm_gap_analysis(numbers_history)
        except:
            algo_results['gap'] = []
        
        try:
            algo_results['hot_cold'] = self.algorithm_hot_cold_mix(numbers_history, hot_numbers, cold_numbers)
        except:
            algo_results['hot_cold'] = []
        
        try:
            algo_results['pattern'] = self.algorithm_pattern_based_enhanced(numbers_history)
        except:
            algo_results['pattern'] = []
        
        try:
            algo_results['position'] = self.algorithm_position_based_enhanced(numbers_history)
        except:
            algo_results['position'] = []
        
        try:
            algo_results['trend'] = self.algorithm_trend_based_enhanced(numbers_history)
        except:
            algo_results['trend'] = []
        
        combined_scores = defaultdict(float)
        algo_weights = self.config['algorithm_weights']
        
        for algo_name, results in algo_results.items():
            weight = algo_weights.get(algo_name, 0.1)
            
            if results:
                for i, (pair, score) in enumerate(results):
                    rank_score = (len(results) - i) / len(results)
                    combined_score = score * rank_score * weight
                    combined_scores[pair] += combined_score
        
        filtered_scores = {}
        for pair, score in combined_scores.items():
            filtered_score = score
            
            if self.config.get('avoid_same_digits', True) and pair[0] == pair[1]:
                continue
            
            if abs(pair[0] - pair[1]) == 1:
                filtered_score *= 0.6
            
            if abs(pair[0] - pair[1]) > self.config.get('max_consecutive_gap', 3):
                filtered_score *= 0.8
            
            if pair[0] in self.repeated_numbers or pair[1] in self.repeated_numbers:
                repeated_penalty = min(0.9, 1.0 - (self.repeated_numbers.get(pair[0], 0) + 
                                                   self.repeated_numbers.get(pair[1], 0)) * 0.1)
                filtered_score *= repeated_penalty
            
            filtered_scores[pair] = filtered_score
        
        try:
            recent_pairs = self.extract_pairs_from_history(numbers_history[-self.config['avoid_recent_pairs']:])
            for pair in recent_pairs:
                if pair in filtered_scores:
                    filtered_scores[pair] *= 0.7
        except:
            pass
        
        if filtered_scores:
            sorted_pairs = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_pairs
        return []
    
    def predict_top_pairs(self, numbers_history, hot_numbers, cold_numbers, num_pairs=8):
        """Predict top N pairs with highest probability"""
        if len(numbers_history) < 10:
            return [], {}
        
        try:
            self.update_pair_statistics(numbers_history)
        except:
            pass
        
        all_predictions = self.combine_algorithms(numbers_history, hot_numbers, cold_numbers)
        
        if not all_predictions:
            return [], {}
        
        predictions = []
        confidence_details = {}
        
        for i, (pair, score) in enumerate(all_predictions[:num_pairs]):
            predictions.append(pair)
            
            max_score = all_predictions[0][1] if all_predictions else 1
            normalized_score = score / max_score if max_score > 0 else 0
            
            data_factor = min(1.0, len(numbers_history) / 30)
            score_factor = normalized_score
            pattern_factor = self.calculate_pattern_strength(pair, numbers_history)
            
            base_confidence = 50 + (score_factor * 30) + (data_factor * 10) + (pattern_factor * 10)
            confidence = min(95, base_confidence)
            
            gap = abs(pair[0] - pair[1])
            if gap == 1:
                confidence *= 0.8
            elif 2 <= gap <= 4:
                confidence *= 1.1
            
            confidence_details[pair] = {
                'confidence': int(confidence),
                'score': round(score, 3),
                'gap': gap,
                'recent_appearances': self.count_recent_appearances(pair, numbers_history),
                'historical_frequency': self.pair_frequency.get(pair, 0),
                'is_consecutive': gap == 1,
                'is_optimal_gap': 2 <= gap <= 4
            }
        
        return predictions, confidence_details
    
    def calculate_pattern_strength(self, pair, numbers_history):
        """Calculate pattern strength for a pair"""
        strength = 0
        
        for numbers in numbers_history[-10:]:
            if pair[0] in numbers and pair[1] in numbers:
                pos1 = numbers.index(pair[0])
                pos2 = numbers.index(pair[1])
                pos_diff = abs(pos1 - pos2)
                
                if pos_diff >= 2:
                    strength += 0.1
        
        return min(1.0, strength)
    
    def count_recent_appearances(self, pair, numbers_history, window=10):
        """Count how many times pair appeared recently"""
        count = 0
        try:
            recent_history = numbers_history[-window:] if len(numbers_history) > window else numbers_history
            
            for numbers in recent_history:
                if pair[0] in numbers and pair[1] in numbers:
                    count += 1
        except:
            count = 0
        
        return count
    
    def generate_strategy_recommendations(self, top_pairs, confidence_details):
        """Generate strategic recommendations based on predictions"""
        strategies = []
        
        for pair in top_pairs[:4]:
            details = confidence_details.get(pair, {})
            confidence = details.get('confidence', 0)
            gap = details.get('gap', 0)
            
            strategy_info = {
                'pair': pair,
                'confidence': confidence,
                'gap': gap,
                'recent_appearances': details.get('recent_appearances', 0)
            }
            
            if confidence >= 75:
                strategy_info.update({
                    'strategy': "ĐẶT CƯỢC MẠNH",
                    'reason': f"Độ tin cậy rất cao ({confidence}%)",
                    'color': 'success'
                })
            elif confidence >= 65:
                strategy_info.update({
                    'strategy': "ĐẶT CƯỢC VỪA",
                    'reason': f"Độ tin cậy tốt ({confidence}%)",
                    'color': 'info'
                })
            elif confidence >= 55:
                strategy_info.update({
                    'strategy': "ĐẶT CƯỢC NHẸ",
                    'reason': f"Tiềm năng khá ({confidence}%)",
                    'color': 'warning'
                })
            else:
                strategy_info.update({
                    'strategy': "THEO DÕI",
                    'reason': f"Cần quan sát thêm ({confidence}%)",
                    'color': 'secondary'
                })
            
            if gap == 1:
                strategy_info['gap_analysis'] = "⚠️ Số liền kề (tỉ lệ trúng thấp hơn)"
            elif 2 <= gap <= 4:
                strategy_info['gap_analysis'] = "✅ Khoảng cách tối ưu"
            else:
                strategy_info['gap_analysis'] = "📊 Khoảng cách xa"
            
            strategies.append(strategy_info)
        
        return strategies

# ================= DATA FUNCTIONS =================
def load_data():
    """Load historical data with error handling"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers", "source"])
    
    try:
        df = pd.read_csv(DATA_FILE)
        
        required_cols = ["time", "numbers", "source"]
        for col in required_cols:
            if col not in df.columns:
                if col == "numbers" and len(df.columns) > 0:
                    df["numbers"] = df.iloc[:, -1].astype(str)
                elif col == "time":
                    df["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elif col == "source":
                    df["source"] = "manual"
        
        df["numbers"] = df["numbers"].astype(str).str.strip()
        return df[["time", "numbers", "source"]]
    
    except Exception as e:
        return pd.DataFrame(columns=["time", "numbers", "source"])

def save_data(values, source="manual"):
    """Save multiple entries with source tracking"""
    try:
        df = load_data()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows = []
        for v in values:
            v_str = str(v).strip()
            if v_str.isdigit() and len(v_str) == 5:
                rows.append({
                    "time": now, 
                    "numbers": v_str,
                    "source": source
                })
        
        if rows:
            new_df = pd.DataFrame(rows)
            df = pd.concat([df, new_df], ignore_index=True)
            
            df = df.drop_duplicates(subset=['numbers'], keep='first')
            
            try:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time', ascending=True)
            except:
                pass
            
            df.to_csv(DATA_FILE, index=False)
        
        return len(rows)
    except:
        return 0

def parse_numbers(v):
    """Parse string to list of integers"""
    try:
        return [int(x) for x in str(v) if x.isdigit()][:5]
    except:
        return []

def get_statistics(df):
    """Calculate comprehensive statistics"""
    if df.empty:
        return {}
    
    try:
        all_numbers = []
        number_sequences = []
        
        for nums_str in df['numbers']:
            nums = parse_numbers(nums_str)
            if len(nums) == 5:
                all_numbers.extend(nums)
                number_sequences.append(nums)
        
        if not all_numbers:
            return {}
        
        counter = Counter(all_numbers)
        total = len(all_numbers)
        
        stats = {
            'total_draws': len(df),
            'total_digits': total,
            'frequency': dict(counter),
            'percentage': {k: f"{(v/total*100):.1f}%" for k, v in counter.items()},
            'most_common': counter.most_common(10),
            'least_common': counter.most_common()[:-11:-1],
            'hot_numbers': [n for n, c in counter.most_common(5)],
            'warm_numbers': [n for n, c in counter.most_common(10)[5:8]],
            'cold_numbers': [n for n, c in counter.most_common()[:-6:-1]],
            'number_sequences': number_sequences,
            'data_quality': len(number_sequences) / len(df) if len(df) > 0 else 0
        }
        
        return stats
    except:
        return {}

# ================= MAIN APP =================
def main():
    st.title("🎯 NUMCORE AI MASTER - 2 SỐ 5 TÍNH")
    st.caption("AI NÂNG CAO - Loại bỏ số chập - Dự đoán chính xác - Chiến lược thông minh")
    
    try:
        ai = EnhancedTwoNumberAI()
    except Exception as e:
        st.error("⚠️ Đang khởi tạo AI nâng cao...")
        ai = EnhancedTwoNumberAI()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Nhập liệu",
        "🎯 Dự đoán AI",
        "📊 Phân tích số",
        "🤖 Thuật toán",
        "⚙️ Cấu hình"
    ])
    
    # ============ TAB 1: DATA INPUT ============
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📥 Nhập dữ liệu giải thưởng")
            
            raw = st.text_area(
                "Nhập nhiều kỳ (mỗi dòng 5 số)",
                height=200,
                placeholder="Ví dụ:\n12345\n67890\n54321\n...",
                help="Mỗi dòng là một giải thưởng gồm 5 chữ số. Số chập (11, 22, 66...) sẽ được AI tự động xử lý.",
                key="data_input"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("💾 Lưu dữ liệu", type="primary", use_container_width=True):
                    if raw.strip():
                        lines = [x.strip() for x in raw.splitlines() if x.strip()]
                        saved = save_data(lines)
                        
                        if saved > 0:
                            st.success(f"✅ Đã lưu {saved} kỳ hợp lệ")
                            st.rerun()
                        else:
                            st.error("❌ Không có dữ liệu hợp lệ")
                    else:
                        st.warning("⚠️ Vui lòng nhập dữ liệu")
            
            with col_btn2:
                if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
                    st.rerun()
        
        with col2:
            st.subheader("📁 Thông tin dữ liệu")
            df = load_data()
            
            if not df.empty:
                st.metric("Tổng số kỳ", len(df))
                
                try:
                    df['time'] = pd.to_datetime(df['time'])
                    latest = df['time'].max().strftime("%d/%m/%Y")
                    st.metric("Dữ liệu mới nhất", latest)
                except:
                    st.metric("Dữ liệu mới nhất", "N/A")
                
                stats = get_statistics(df)
                if stats:
                    hot_num = stats['hot_numbers'][0] if stats['hot_numbers'] else "--"
                    st.metric("Số nóng nhất", hot_num)
                
                with st.expander("📋 5 kỳ gần nhất"):
                    display_df = df.tail(5).copy()
                    if 'time' in display_df.columns:
                        try:
                            display_df['time'] = pd.to_datetime(display_df['time']).dt.strftime('%H:%M %d/%m')
                        except:
                            pass
                    st.dataframe(
                        display_df[['time', 'numbers']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "time": "Thời gian",
                            "numbers": "Số"
                        }
                    )
            else:
                st.info("📭 Chưa có dữ liệu")
                st.caption("Nhập dữ liệu để bắt đầu phân tích")
    
    # ============ TAB 2: AI PREDICTION ============
    with tab2:
        df = load_data()
        
        if df.empty:
            st.warning("⏳ Vui lòng nhập dữ liệu trước khi phân tích")
            st.info("Chuyển sang tab '📥 Nhập liệu' để thêm dữ liệu")
        else:
            stats = get_statistics(df)
            
            if 'number_sequences' not in stats:
                st.error("Dữ liệu không đúng định dạng 5 số")
                return
            
            numbers_history = stats['number_sequences']
            hot_numbers = stats.get('hot_numbers', [])
            cold_numbers = stats.get('cold_numbers', [])
            
            if len(numbers_history) < 10:
                st.warning(f"⚠️ Cần ít nhất 10 kỳ để phân tích (hiện có: {len(numbers_history)})")
                
                if hot_numbers:
                    st.subheader("🔥 Số nóng hiện tại")
                    hot_cols = st.columns(5)
                    for idx, num in enumerate(hot_numbers[:5]):
                        with hot_cols[idx]:
                            percent = stats['percentage'].get(num, "0%")
                            st.metric(f"Số {num}", f"{percent}")
                return
            
            st.subheader("🎯 AI DỰ ĐOÁN CẶP SỐ")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Số kỳ phân tích", len(numbers_history))
            with col2:
                data_quality = stats.get('data_quality', 0)
                st.metric("Chất lượng dữ liệu", f"{data_quality*100:.0f}%")
            with col3:
                hot_num = hot_numbers[0] if hot_numbers else "--"
                st.metric("Số nóng nhất", hot_num)
            with col4:
                cold_num = cold_numbers[0] if cold_numbers else "--"
                st.metric("Số lạnh nhất", cold_num)
            
            try:
                top_pairs, confidence_details = ai.predict_top_pairs(
                    numbers_history, hot_numbers, cold_numbers, num_pairs=8
                )
            except Exception as e:
                st.error("⚠️ Có lỗi khi phân tích dữ liệu. Vui lòng thử lại.")
                top_pairs = []
                confidence_details = {}
            
            if not top_pairs:
                st.info("🔍 Đang phân tích dữ liệu... Hãy nhập thêm dữ liệu")
                return
            
            st.subheader("🏆 TOP CẶP SỐ DỰ ĐOÁN")
            
            top_cols = st.columns(4)
            for idx, (pair, col) in enumerate(zip(top_pairs[:4], top_cols)):
                with col:
                    details = confidence_details.get(pair, {})
                    confidence = details.get('confidence', 0)
                    gap = details.get('gap', 0)
                    
                    if confidence >= 75:
                        badge = "🔥"
                        color = "green"
                    elif confidence >= 65:
                        badge = "⭐"
                        color = "blue"
                    elif confidence >= 55:
                        badge = "📈"
                        color = "orange"
                    else:
                        badge = "📊"
                        color = "gray"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; border-radius: 10px; 
                                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                border: 2px solid {color}; margin: 5px;">
                        <h3 style="color: {color}; margin: 0;">{badge} Cặp {idx+1}</h3>
                        <h1 style="font-size: 2.5em; margin: 10px 0; color: #2c3e50;">{pair[0]}{pair[1]}</h1>
                        <div style="font-size: 1.2em; color: {color}; font-weight: bold;">
                            {confidence}% tin cậy
                        </div>
                        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                            Khoảng cách: {gap} | Gần đây: {details.get('recent_appearances', 0)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            st.subheader("🎯 CHIẾN LƯỢC ĐẶT CƯỢC THÔNG MINH")
            
            try:
                strategies = ai.generate_strategy_recommendations(top_pairs, confidence_details)
            except:
                strategies = []
            
            if strategies:
                strategy_cols = st.columns(min(4, len(strategies)))
                for idx, (strategy, col) in enumerate(zip(strategies[:4], strategy_cols)):
                    with col:
                        pair_str = f"{strategy['pair'][0]}{strategy['pair'][1]}"
                        
                        if strategy['color'] == 'success':
                            st.success(f"**{pair_str}** - {strategy['strategy']}")
                        elif strategy['color'] == 'info':
                            st.info(f"**{pair_str}** - {strategy['strategy']}")
                        elif strategy['color'] == 'warning':
                            st.warning(f"**{pair_str}** - {strategy['strategy']}")
                        else:
                            st.markdown(f"**{pair_str}** - {strategy['strategy']}")
                        
                        st.caption(f"{strategy['reason']}")
                        st.caption(f"📊 {strategy['gap_analysis']}")
                        
                        with st.expander("📈 Thống kê nhanh"):
                            st.write(f"**Độ tin cậy:** {strategy['confidence']}%")
                            st.write(f"**Khoảng cách:** {strategy['gap']}")
                            st.write(f"**Xuất hiện gần đây:** {strategy['recent_appearances']} lần")
            
            st.divider()
            
            st.subheader("🔍 PHÂN TÍCH CHI TIẾT CẶP SỐ")
            
            if top_pairs:
                options = []
                for pair in top_pairs:
                    details = confidence_details.get(pair, {})
                    confidence = details.get('confidence', 0)
                    gap = details.get('gap', 0)
                    options.append(f"{pair[0]}{pair[1]} (Độ tin cậy: {confidence}%, Khoảng cách: {gap})")
                
                selected_option = st.selectbox(
                    "Chọn cặp số để phân tích chi tiết:",
                    options=options,
                    index=0
                )
                
                if selected_option:
                    pair_str = selected_option.split(" ")[0]
                    if len(pair_str) == 2:
                        pair_tuple = (int(pair_str[0]), int(pair_str[1]))
                        details = confidence_details.get(pair_tuple, {})
                        
                        if details:
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.markdown("### 📊 Thống kê cơ bản")
                                
                                confidence = details.get('confidence', 0)
                                st.progress(confidence/100, text=f"Độ tin cậy: {confidence}%")
                                
                                metrics_data = {
                                    "Điểm số AI": f"{details.get('score', 0):.3f}",
                                    "Khoảng cách số": str(details.get('gap', 0)),
                                    "Xuất hiện gần đây": f"{details.get('recent_appearances', 0)} lần",
                                    "Tổng lần xuất hiện": f"{details.get('historical_frequency', 0)} lần"
                                }
                                
                                for key, value in metrics_data.items():
                                    st.write(f"**{key}:** {value}")
                                
                                st.markdown("### 🔥 Trạng thái số")
                                for num in pair_tuple:
                                    if num in hot_numbers[:3]:
                                        status = f"**Số {num}:** 🔥 Số nóng hàng đầu"
                                    elif num in hot_numbers:
                                        status = f"**Số {num}:** ⭐ Số nóng"
                                    elif num in cold_numbers:
                                        status = f"**Số {num}:** ❄️ Số lạnh (tiềm năng)"
                                    else:
                                        status = f"**Số {num}:** 📊 Số trung bình"
                                    st.write(status)
                            
                            with detail_col2:
                                st.markdown("### 🎯 Đánh giá chiến lược")
                                
                                gap = details.get('gap', 0)
                                if gap == 1:
                                    st.warning("**Khoảng cách liền kề:** Cặp số liền nhau thường có tỉ lệ trúng thấp hơn trong trò chơi 2 số 5 tính.")
                                elif 2 <= gap <= 4:
                                    st.success("**Khoảng cách tối ưu:** Khoảng cách từ 2-4 là lý tưởng cho cặp số.")
                                else:
                                    st.info("**Khoảng cách xa:** Có thể tạo ra sự bất ngờ.")
                                
                                if details.get('is_consecutive'):
                                    st.warning("⚠️ **Cảnh báo:** Số liền kề thường ít xuất hiện cùng nhau.")
                                
                                recent = details.get('recent_appearances', 0)
                                if recent >= 2:
                                    st.success(f"✅ **Xu hướng tốt:** Đã xuất hiện {recent} lần gần đây.")
                                elif recent == 1:
                                    st.info(f"📈 **Đang nổi:** Vừa xuất hiện trong kỳ gần nhất.")
                                else:
                                    st.info(f"🔍 **Tiềm năng:** Chưa xuất hiện gần đây, có thể là cơ hội.")
            
            st.divider()
            
            st.subheader("📋 TẤT CẢ CẶP SỐ DỰ ĐOÁN")
            
            pairs_data = []
            for pair in top_pairs:
                details = confidence_details.get(pair, {})
                
                confidence = details.get('confidence', 0)
                if confidence >= 75:
                    status = "🔥 Rất tốt"
                elif confidence >= 65:
                    status = "⭐ Tốt"
                elif confidence >= 55:
                    status = "📈 Khá"
                else:
                    status = "📊 Theo dõi"
                
                pairs_data.append({
                    'Cặp số': f"{pair[0]}{pair[1]}",
                    'Độ tin cậy': f"{confidence}%",
                    'Trạng thái': status,
                    'Khoảng cách': details.get('gap', 0),
                    'Gần đây': details.get('recent_appearances', 0),
                    'Lịch sử': details.get('historical_frequency', 0),
                    'Điểm số': round(details.get('score', 0), 3)
                })
            
            if pairs_data:
                pairs_df = pd.DataFrame(pairs_data)
                
                def color_status(val):
                    if "🔥" in val:
                        return 'background-color: #ffcccc'
                    elif "⭐" in val:
                        return 'background-color: #ccffcc'
                    elif "📈" in val:
                        return 'background-color: #ffffcc'
                    else:
                        return 'background-color: #e6e6e6'
                
                styled_df = pairs_df.style.applymap(color_status, subset=['Trạng thái'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Cặp số": st.column_config.TextColumn("Cặp số", width="small"),
                        "Độ tin cậy": st.column_config.ProgressColumn(
                            "Độ tin cậy",
                            min_value=0,
                            max_value=100,
                            format="%d%%"
                        ),
                        "Trạng thái": st.column_config.TextColumn("Trạng thái", width="medium"),
                        "Khoảng cách": st.column_config.NumberColumn("K.cách", width="small"),
                        "Gần đây": st.column_config.NumberColumn("Gần đây", width="small"),
                        "Lịch sử": st.column_config.NumberColumn("Lịch sử", width="small"),
                        "Điểm số": st.column_config.NumberColumn("Điểm", format="%.3f", width="small")
                    }
                )
    
    # ============ TAB 3: NUMBER ANALYSIS ============
    with tab3:
        st.subheader("📊 PHÂN TÍCH SỐ HỌC NÂNG CAO")
        
        df = load_data()
        
        if df.empty:
            st.info("📭 Chưa có dữ liệu để phân tích")
        else:
            stats = get_statistics(df)
            
            if not stats:
                return
            
            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("Tổng số kỳ", stats['total_draws'])
            with overview_cols[1]:
                st.metric("Tổng lượt số", stats['total_digits'])
            with overview_cols[2]:
                coverage = len(stats['frequency']) / 10 * 100
                st.metric("Độ phủ số", f"{coverage:.1f}%")
            with overview_cols[3]:
                data_quality = stats.get('data_quality', 0) * 100
                st.metric("Chất lượng DL", f"{data_quality:.0f}%")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 TOP SỐ NÓNG")
                
                if stats['hot_numbers']:
                    hot_data = []
                    for num in stats['hot_numbers'][:5]:
                        count = stats['frequency'].get(num, 0)
                        percent = stats['percentage'].get(num, "0%")
                        hot_data.append({
                            'Số': num,
                            'Lần xuất hiện': count,
                            'Tỉ lệ': percent,
                            'Trạng thái': 'Rất nóng' if num == stats['hot_numbers'][0] else 'Nóng'
                        })
                    
                    hot_df = pd.DataFrame(hot_data)
                    st.dataframe(
                        hot_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("**📈 Biểu đồ số nóng:**")
                    hot_counts = {num: stats['frequency'].get(num, 0) for num in stats['hot_numbers'][:5]}
                    if hot_counts:
                        hot_series = pd.Series(hot_counts)
                        st.bar_chart(hot_series)
            
            with col2:
                st.subheader("❄️ TOP SỐ LẠNH")
                
                if stats['cold_numbers']:
                    cold_data = []
                    for num in stats['cold_numbers'][:5]:
                        count = stats['frequency'].get(num, 0)
                        percent = stats['percentage'].get(num, "0%")
                        cold_data.append({
                            'Số': num,
                            'Lần xuất hiện': count,
                            'Tỉ lệ': percent,
                            'Trạng thái': 'Rất lạnh' if num == stats['cold_numbers'][0] else 'Lạnh'
                        })
                    
                    cold_df = pd.DataFrame(cold_data)
                    st.dataframe(
                        cold_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("💡 **Gợi ý:** Số lạnh có thể sẽ xuất hiện trở lại theo chu kỳ. Kết hợp số lạnh với số nóng thường tạo ra cặp số tiềm năng.")
            
            st.divider()
            
            st.subheader("🔍 PHÁT HIỆN MẪU SỐ")
            
            if 'number_sequences' in stats and len(stats['number_sequences']) >= 10:
                numbers_history = stats['number_sequences']
                
                pattern_cols = st.columns(3)
                
                with pattern_cols[0]:
                    st.write("**🎯 Số thường đi cùng:**")
                    pair_counter = Counter()
                    for numbers in numbers_history[-15:]:
                        if len(numbers) == 5:
                            unique_nums = sorted(set(numbers))
                            for i in range(len(unique_nums)):
                                for j in range(i+1, len(unique_nums)):
                                    pair = (unique_nums[i], unique_nums[j])
                                    pair_counter[pair] += 1
                    
                    for pair, count in pair_counter.most_common(3):
                        st.write(f"{pair[0]}{pair[1]}: {count} lần")
                
                with pattern_cols[1]:
                    st.write("**📊 Phân bố chẵn/lẻ:**")
                    even_odd_counts = {'Chẵn': 0, 'Lẻ': 0}
                    for numbers in numbers_history[-10:]:
                        for num in numbers:
                            if num % 2 == 0:
                                even_odd_counts['Chẵn'] += 1
                            else:
                                even_odd_counts['Lẻ'] += 1
                    
                    total = sum(even_odd_counts.values())
                    if total > 0:
                        st.write(f"Chẵn: {even_odd_counts['Chẵn']} ({even_odd_counts['Chẵn']/total*100:.1f}%)")
                        st.write(f"Lẻ: {even_odd_counts['Lẻ']} ({even_odd_counts['Lẻ']/total*100:.1f}%)")
                
                with pattern_cols[2]:
                    st.write("**🔢 Phân bố lớn/nhỏ:**")
                    size_counts = {'Nhỏ (0-4)': 0, 'Lớn (5-9)': 0}
                    for numbers in numbers_history[-10:]:
                        for num in numbers:
                            if num <= 4:
                                size_counts['Nhỏ (0-4)'] += 1
                            else:
                                size_counts['Lớn (5-9)'] += 1
                    
                    total = sum(size_counts.values())
                    if total > 0:
                        st.write(f"Nhỏ: {size_counts['Nhỏ (0-4)']} ({size_counts['Nhỏ (0-4)']/total*100:.1f}%)")
                        st.write(f"Lớn: {size_counts['Lớn (5-9)']} ({size_counts['Lớn (5-9)']/total*100:.1f}%)")
            
            st.divider()
            st.subheader("📅 DỮ LIỆU GẦN ĐÂY")
            
            if not df.empty:
                recent_df = df.tail(8).copy()
                if 'time' in recent_df.columns:
                    try:
                        recent_df['time'] = pd.to_datetime(recent_df['time']).dt.strftime('%H:%M %d/%m')
                    except:
                        pass
                
                st.dataframe(
                    recent_df[['time', 'numbers']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "time": "Thời gian",
                        "numbers": "Số"
                    }
                )
    
    # ============ TAB 4: ALGORITHM ANALYSIS ============
    with tab4:
        st.subheader("🤖 PHÂN TÍCH THUẬT TOÁN AI")
        
        st.markdown("""
        ### 🎯 6 THUẬT TOÁN NÂNG CAO
        
        Phiên bản AI Master sử dụng 6 thuật toán chuyên biệt cho trò chơi **2 số 5 tính**:
        
        1. **🔢 Tần suất nâng cao** (20%) - Phân tích tần suất xuất hiện của cặp số, loại bỏ số chập
        2. **📏 Phân tích khoảng cách** (18%) - Tối ưu khoảng cách giữa 2 số (tránh số liền kề)
        3. **🔥❄️ Kết hợp nóng-lạnh** (18%) - Kết hợp số nóng với số lạnh tiềm năng
        4. **🎯 Nhận diện mẫu** (15%) - Phát hiện pattern xuất hiện của cặp số
        5. **📍 Phân tích vị trí** (15%) - Vị trí xuất hiện trong giải 5 số
        6. **📈 Phân tích xu hướng** (14%) - Xu hướng tăng/giảm của từng số
        
        **⚡ Ưu điểm:** Tự động loại bỏ số chập (11, 22, 33...), tối ưu khoảng cách số, kết hợp thông minh giữa số nóng và số lạnh.
        """)
        
        df = load_data()
        
        if not df.empty:
            stats = get_statistics(df)
            
            if 'number_sequences' in stats and len(stats['number_sequences']) >= 10:
                numbers_history = stats['number_sequences']
                hot_numbers = stats.get('hot_numbers', [])
                cold_numbers = stats.get('cold_numbers', [])
                
                st.divider()
                st.subheader("📊 HIỆU SUẤT THUẬT TOÁN")
                
                algorithms = [
                    ("Tần suất", ai.algorithm_frequency_based),
                    ("Khoảng cách", ai.algorithm_gap_analysis),
                    ("Nóng-Lạnh", ai.algorithm_hot_cold_mix),
                    ("Mẫu số", ai.algorithm_pattern_based_enhanced),
                    ("Vị trí", ai.algorithm_position_based_enhanced),
                    ("Xu hướng", ai.algorithm_trend_based_enhanced)
                ]
                
                algo_cols = st.columns(3)
                for idx, (algo_name, algo_func) in enumerate(algorithms):
                    with algo_cols[idx % 3]:
                        try:
                            if algo_name == "Nóng-Lạnh":
                                results = algo_func(numbers_history, hot_numbers, cold_numbers)
                            elif algo_name == "Tần suất":
                                results = algo_func(numbers_history, hot_numbers)
                            else:
                                results = algo_func(numbers_history)
                            
                            st.write(f"**{algo_name}:**")
                            
                            if results:
                                top_2 = results[:2]
                                for i, (pair, score) in enumerate(top_2):
                                    if pair[0] != pair[1]:
                                        st.write(f"{i+1}. **{pair[0]}{pair[1]}** ({score:.3f})")
                                    else:
                                        st.write(f"{i+1}. ❌ {pair[0]}{pair[1]} (số chập)")
                            else:
                                st.write("⏳ Đang tính toán...")
                        except:
                            st.write(f"**{algo_name}:** 🔄 Đang xử lý...")
                
                st.divider()
                st.subheader("⚖️ CƠ CHẾ KẾT HỢP THUẬT TOÁN")
                
                st.info("""
                **AI Master sử dụng cơ chế kết hợp thông minh:**
                
                - Mỗi thuật toán có **trọng số** riêng dựa trên hiệu quả
                - **Tự động loại bỏ** cặp số chập (11, 22, 33...)
                - **Ưu tiên** khoảng cách số tối ưu (2-4)
                - **Giảm điểm** số liền kề (0-1, 1-2...)
                - **Kết hợp** số nóng với số lạnh tiềm năng
                - **Phân tích** pattern lịch sử xuất hiện
                """)
    
    # ============ TAB 5: CONFIGURATION ============
    with tab5:
        st.subheader("⚙️ CẤU HÌNH AI NÂNG CAO")
        
        try:
            ai.load_config()
            current_config = ai.config
        except:
            current_config = ai.config
        
        st.markdown("### ⚖️ ĐIỀU CHỈNH TRỌNG SỐ THUẬT TOÁN")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_weight = st.slider(
                "Thuật toán Tần suất",
                0.05, 0.35, float(current_config['algorithm_weights'].get('frequency_based', 0.20)), 0.05
            )
            
            gap_weight = st.slider(
                "Thuật toán Khoảng cách",
                0.05, 0.35, float(current_config['algorithm_weights'].get('gap_analysis', 0.18)), 0.05
            )
            
            hotcold_weight = st.slider(
                "Thuật toán Nóng-Lạnh",
                0.05, 0.35, float(current_config['algorithm_weights'].get('hot_cold_mix', 0.18)), 0.05
            )
        
        with col2:
            pattern_weight = st.slider(
                "Thuật toán Mẫu số",
                0.05, 0.30, float(current_config['algorithm_weights'].get('pattern_based', 0.15)), 0.05
            )
            
            position_weight = st.slider(
                "Thuật toán Vị trí",
                0.05, 0.30, float(current_config['algorithm_weights'].get('position_based', 0.15)), 0.05
            )
            
            trend_weight = st.slider(
                "Thuật toán Xu hướng",
                0.05, 0.30, float(current_config['algorithm_weights'].get('trend_based', 0.14)), 0.05
            )
        
        st.divider()
        
        st.markdown("### 🎯 THIẾT LẬP NÂNG CAO")
        
        col3, col4 = st.columns(2)
        
        with col3:
            avoid_recent = st.slider(
                "Tránh cặp trùng (số kỳ)",
                1, 15, current_config.get('avoid_recent_pairs', 5), 1
            )
            
            min_confidence = st.slider(
                "Độ tin cậy tối thiểu (%)",
                45, 85, current_config.get('min_confidence', 65), 5
            )
            
            max_gap = st.slider(
                "Khoảng cách tối đa ưu tiên",
                2, 8, current_config.get('max_consecutive_gap', 3), 1
            )
        
        with col4:
            avoid_same = st.checkbox(
                "Tự động loại bỏ số chập",
                value=current_config.get('avoid_same_digits', True)
            )
            
            prefer_complementary = st.checkbox(
                "Ưu tiên số bổ trợ",
                value=current_config.get('prefer_complementary', True)
            )
            
            recent_weight = st.slider(
                "Trọng số dữ liệu gần",
                0.4, 0.9, current_config.get('recent_weight', 0.65), 0.05
            )
        
        if st.button("💾 Lưu cấu hình AI", type="primary", use_container_width=True):
            ai.config['algorithm_weights'] = {
                'frequency_based': freq_weight,
                'gap_analysis': gap_weight,
                'hot_cold_mix': hotcold_weight,
                'pattern_based': pattern_weight,
                'position_based': position_weight,
                'trend_based': trend_weight
            }
            
            ai.config['avoid_recent_pairs'] = avoid_recent
            ai.config['min_confidence'] = min_confidence
            ai.config['max_consecutive_gap'] = max_gap
            ai.config['avoid_same_digits'] = avoid_same
            ai.config['prefer_complementary'] = prefer_complementary
            ai.config['recent_weight'] = recent_weight
            
            try:
                ai.save_config()
                st.success("✅ Đã lưu cấu hình AI thành công!")
                st.rerun()
            except:
                st.error("❌ Lỗi khi lưu cấu hình")
        
        st.divider()
        
        st.subheader("🔄 QUẢN LÝ DỮ LIỆU")
        
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("🗑️ Xóa dữ liệu AI", use_container_width=True, type="secondary"):
                for file in [AI_CONFIG_FILE, PAIR_HISTORY_FILE]:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
                ai.reset_statistics()
                st.success("✅ Đã xóa dữ liệu AI học tập")
                st.rerun()
        
        with col6:
            if st.button("📥 Xuất dữ liệu", use_container_width=True):
                df = load_data()
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Tải file CSV",
                        data=csv,
                        file_name=f"numcore_master_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("Không có dữ liệu để xuất")
        
        st.divider()
        st.markdown("### ℹ️ THÔNG TIN ỨNG DỤNG")
        
        info_cols = st.columns(3)
        with info_cols[0]:
            df_count = len(load_data())
            st.metric("Dữ liệu hiện có", f"{df_count} kỳ")
        
        with info_cols[1]:
            st.metric("Phiên bản AI", "Master v10.0")
        
        with info_cols[2]:
            st.metric("Trạng thái", "✅ Hoạt động")
        
        st.caption("NUMCORE AI MASTER - Tối ưu cho trò chơi 2 số 5 tính | Tự động loại bỏ số chập | Dự đoán chính xác cao")

# ================= RUN APP =================
if __name__ == "__main__":
    main()
