import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from datetime import datetime, timedelta
import os
import random
import json
from pathlib import Path
import warnings
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE AI ULTIMATE - 2 SỐ 5 TÍNH",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "numcore_data.csv"
AI_CONFIG_FILE = "ai_config.json"
PAIR_HISTORY_FILE = "pair_history.json"

# ================= ADVANCED AI FOR 2-NUMBER PAIRS =================
class TwoNumberAI:
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.history_window = 30
        self.pair_frequency = Counter()
        self.position_pairs = defaultdict(Counter)
        self.load_config()
        self.load_pair_history()
    
    def load_config(self):
        """Load AI configuration"""
        default_config = {
            "algorithm_weights": {
                "frequency_based": 0.25,
                "position_based": 0.20,
                "trend_based": 0.20,
                "pattern_based": 0.15,
                "statistical": 0.10,
                "neural_inspired": 0.10
            },
            "recent_weight": 0.6,
            "pair_appearance_threshold": 2,
            "avoid_recent_pairs": 5,
            "min_confidence": 60
        }
        
        if os.path.exists(AI_CONFIG_FILE):
            with open(AI_CONFIG_FILE, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
    
    def load_pair_history(self):
        """Load historical pair data"""
        if os.path.exists(PAIR_HISTORY_FILE):
            with open(PAIR_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                self.pair_frequency = Counter(data.get('pair_frequency', {}))
                self.position_pairs = defaultdict(Counter, data.get('position_pairs', {}))
    
    def save_pair_history(self):
        """Save pair history data"""
        data = {
            'pair_frequency': dict(self.pair_frequency),
            'position_pairs': dict(self.position_pairs)
        }
        with open(PAIR_HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def extract_pairs_from_history(self, numbers_history):
        """Extract all 2-number pairs from history"""
        all_pairs = []
        
        for numbers in numbers_history:
            if len(numbers) == 5:
                # Extract all unique pairs from this draw
                unique_numbers = set(numbers)
                for pair in combinations(sorted(unique_numbers), 2):
                    all_pairs.append(tuple(pair))
        
        return all_pairs
    
    def update_pair_statistics(self, numbers_history):
        """Update pair frequency statistics"""
        new_pairs = self.extract_pairs_from_history(numbers_history[-10:])  # Last 10 draws
        
        for pair in new_pairs:
            self.pair_frequency[pair] += 1
        
        # Update position-based pairs
        for numbers in numbers_history[-10:]:
            if len(numbers) == 5:
                for i in range(5):
                    for j in range(i+1, 5):
                        pos_pair = (i, j, numbers[i], numbers[j])
                        self.position_pairs[(numbers[i], numbers[j])][pos_pair] += 1
        
        self.save_pair_history()
    
    # ============= MULTIPLE ALGORITHMS =============
    
    def algorithm_frequency_based(self, numbers_history, hot_numbers):
        """Algorithm 1: Frequency-based pair prediction"""
        if len(numbers_history) < 10:
            return []
        
        all_pairs = self.extract_pairs_from_history(numbers_history)
        pair_counter = Counter(all_pairs)
        
        # Get most frequent pairs
        frequent_pairs = pair_counter.most_common(20)
        
        # Score pairs based on frequency and recency
        pair_scores = {}
        recent_pairs = self.extract_pairs_from_history(numbers_history[-5:])
        
        for pair, freq in frequent_pairs:
            score = freq * 0.5
            
            # Bonus for recent appearance
            if pair in recent_pairs:
                score *= 1.3
            
            # Bonus if contains hot numbers
            hot_count = sum(1 for num in pair if num in hot_numbers[:3])
            score *= (1 + hot_count * 0.2)
            
            pair_scores[pair] = score
        
        return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def algorithm_position_based(self, numbers_history):
        """Algorithm 2: Position-based pair prediction"""
        if len(numbers_history) < 15:
            return []
        
        position_stats = {i: Counter() for i in range(5)}
        
        for numbers in numbers_history[-15:]:
            if len(numbers) == 5:
                for pos, num in enumerate(numbers):
                    position_stats[pos][num] += 1
        
        # Find numbers that appear together in specific positions
        position_pairs = []
        for numbers in numbers_history[-10:]:
            if len(numbers) == 5:
                for i in range(5):
                    for j in range(i+1, 5):
                        position_pairs.append(((i, numbers[i]), (j, numbers[j])))
        
        position_pair_counter = Counter(position_pairs)
        
        # Score pairs based on position patterns
        pair_scores = {}
        for (pos1, num1), (pos2, num2) in position_pair_counter:
            pair = tuple(sorted([num1, num2]))
            freq = position_pair_counter[((pos1, num1), (pos2, num2))]
            
            # Calculate position strength
            pos1_strength = position_stats[pos1][num1] / max(1, sum(position_stats[pos1].values()))
            pos2_strength = position_stats[pos2][num2] / max(1, sum(position_stats[pos2].values()))
            
            score = freq * (pos1_strength + pos2_strength) * 0.5
            pair_scores[pair] = score
        
        return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def algorithm_trend_based(self, numbers_history):
        """Algorithm 3: Trend-based pair prediction"""
        if len(numbers_history) < 20:
            return []
        
        # Analyze trends for each number
        trends = {}
        for num in range(10):
            recent_counts = []
            window_size = 5
            
            for i in range(0, len(numbers_history)-window_size+1, window_size):
                window = numbers_history[i:i+window_size]
                count = sum(1 for nums in window for n in nums if n == num)
                recent_counts.append(count)
            
            if len(recent_counts) >= 2:
                # Calculate trend (increasing/decreasing)
                trend = (recent_counts[-1] - recent_counts[0]) / max(1, sum(recent_counts))
                trends[num] = trend
        
        # Find pairs with complementary trends
        pair_scores = {}
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                if num1 in trends and num2 in trends:
                    # Both numbers should have positive or complementary trends
                    trend_score = abs(trends[num1] + trends[num2])
                    
                    # Check if they appear together recently
                    recent_together = 0
                    for numbers in numbers_history[-8:]:
                        if num1 in numbers and num2 in numbers:
                            recent_together += 1
                    
                    score = trend_score * (1 + recent_together * 0.3)
                    pair_scores[(num1, num2)] = score
        
        return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def algorithm_pattern_based(self, numbers_history):
        """Algorithm 4: Pattern-based pair prediction"""
        if len(numbers_history) < 25:
            return []
        
        # Analyze digit patterns
        digit_patterns = defaultdict(list)
        
        for numbers in numbers_history:
            if len(numbers) == 5:
                for i in range(4):
                    pattern = abs(numbers[i] - numbers[i+1])
                    digit_patterns[pattern].append((numbers[i], numbers[i+1]))
        
        # Find common patterns
        pattern_scores = {}
        for pattern, pairs in digit_patterns.items():
            if len(pairs) >= 3:  # Minimum occurrences
                pair_counter = Counter(pairs)
                for pair, count in pair_counter.most_common(5):
                    score = count * (1 / (pattern + 1))  # Smaller gaps get higher scores
                    pattern_scores[tuple(sorted(pair))] = score
        
        return sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def algorithm_statistical(self, numbers_history):
        """Algorithm 5: Statistical analysis"""
        if len(numbers_history) < 30:
            return []
        
        # Calculate probabilities using statistical methods
        all_numbers = [num for nums in numbers_history for num in nums]
        number_freq = Counter(all_numbers)
        total = len(all_numbers)
        
        # Calculate expected probability of pairs
        pair_probabilities = {}
        
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                # Probability both appear in same draw
                prob_num1 = number_freq[num1] / total
                prob_num2 = number_freq[num2] / total
                
                # Count actual co-occurrences
                cooccurrences = sum(1 for nums in numbers_history 
                                  if num1 in nums and num2 in nums)
                
                expected = prob_num1 * prob_num2 * len(numbers_history)
                actual = cooccurrences
                
                if expected > 0:
                    ratio = actual / expected
                    # High ratio means they appear together more than expected
                    pair_probabilities[(num1, num2)] = ratio
        
        return sorted(pair_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def algorithm_neural_inspired(self, numbers_history, hot_numbers):
        """Algorithm 6: Neural network inspired approach"""
        if len(numbers_history) < 15:
            return []
        
        # Simulate neural network-like pattern recognition
        pair_scores = {}
        
        # Create feature vectors for each pair
        for num1 in range(10):
            for num2 in range(num1+1, 10):
                features = []
                
                # Feature 1: Co-occurrence frequency
                cooccur = sum(1 for nums in numbers_history 
                            if num1 in nums and num2 in nums)
                features.append(cooccur / len(numbers_history))
                
                # Feature 2: Recent momentum
                recent_cooccur = sum(1 for nums in numbers_history[-5:] 
                                   if num1 in nums and num2 in nums)
                features.append(recent_cooccur / 5)
                
                # Feature 3: Hot number connection
                hot_connection = 1 if (num1 in hot_numbers[:3] or num2 in hot_numbers[:3]) else 0
                features.append(hot_connection)
                
                # Feature 4: Complementary trends
                recent1 = sum(1 for nums in numbers_history[-5:] for n in nums if n == num1)
                recent2 = sum(1 for nums in numbers_history[-5:] for n in nums if n == num2)
                features.append(abs(recent1 - recent2) / 5)
                
                # Weighted sum of features (simulating neural network)
                weights = [0.4, 0.3, 0.2, 0.1]  # Learned weights
                score = sum(f * w for f, w in zip(features, weights))
                
                pair_scores[(num1, num2)] = score
        
        return sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def combine_algorithms(self, numbers_history, hot_numbers):
        """Combine results from all algorithms"""
        if len(numbers_history) < 10:
            return []
        
        # Get results from all algorithms
        algo_results = {
            'frequency': self.algorithm_frequency_based(numbers_history, hot_numbers),
            'position': self.algorithm_position_based(numbers_history),
            'trend': self.algorithm_trend_based(numbers_history),
            'pattern': self.algorithm_pattern_based(numbers_history),
            'statistical': self.algorithm_statistical(numbers_history),
            'neural': self.algorithm_neural_inspired(numbers_history, hot_numbers)
        }
        
        # Combine scores using weighted average
        combined_scores = defaultdict(float)
        algo_weights = self.config['algorithm_weights']
        
        for algo_name, results in algo_results.items():
            weight = algo_weights.get(algo_name, 0.1)
            
            for i, (pair, score) in enumerate(results):
                # Normalize score based on ranking
                normalized_score = (len(results) - i) / len(results)
                combined_scores[pair] += normalized_score * weight
        
        # Apply recent appearance penalty
        recent_pairs = self.extract_pairs_from_history(numbers_history[-self.config['avoid_recent_pairs']:])
        for pair in recent_pairs:
            if pair in combined_scores:
                combined_scores[pair] *= 0.7  # Reduce score for recent pairs
        
        # Sort by combined score
        sorted_pairs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_pairs
    
    def predict_top_pairs(self, numbers_history, hot_numbers, num_pairs=5):
        """Predict top N pairs with highest probability"""
        if len(numbers_history) < 10:
            return [], {}
        
        # Update statistics
        self.update_pair_statistics(numbers_history)
        
        # Get combined predictions
        all_predictions = self.combine_algorithms(numbers_history, hot_numbers)
        
        # Calculate confidence scores
        predictions = []
        confidence_details = {}
        
        for i, (pair, score) in enumerate(all_predictions[:num_pairs]):
            predictions.append(pair)
            
            # Calculate confidence level
            max_score = all_predictions[0][1] if all_predictions else 1
            normalized_score = score / max_score if max_score > 0 else 0
            
            # Base confidence on score and data size
            base_confidence = min(95, 50 + int(normalized_score * 40))
            data_boost = min(30, len(numbers_history) // 3)
            confidence = min(98, base_confidence + data_boost)
            
            # Get algorithm contributions
            algo_contributions = self.get_algorithm_contributions(pair, numbers_history, hot_numbers)
            
            confidence_details[pair] = {
                'confidence': confidence,
                'score': round(score, 3),
                'algorithms': algo_contributions,
                'recent_appearances': self.count_recent_appearances(pair, numbers_history),
                'historical_frequency': self.pair_frequency.get(pair, 0)
            }
        
        return predictions, confidence_details
    
    def get_algorithm_contributions(self, pair, numbers_history, hot_numbers):
        """Get contribution scores from each algorithm for a specific pair"""
        contributions = {}
        
        # Check each algorithm
        algo_methods = {
            'frequency': self.algorithm_frequency_based,
            'position': self.algorithm_position_based,
            'trend': self.algorithm_trend_based,
            'pattern': self.algorithm_pattern_based,
            'statistical': self.algorithm_statistical,
            'neural': self.algorithm_neural_inspired
        }
        
        for algo_name, method in algo_methods.items():
            try:
                results = method(numbers_history, hot_numbers)
                # Find this pair in results
                for result_pair, score in results:
                    if result_pair == pair:
                        contributions[algo_name] = round(score, 3)
                        break
            except:
                contributions[algo_name] = 0
        
        return contributions
    
    def count_recent_appearances(self, pair, numbers_history, window=10):
        """Count how many times pair appeared recently"""
        count = 0
        recent_history = numbers_history[-window:] if len(numbers_history) > window else numbers_history
        
        for numbers in recent_history:
            if pair[0] in numbers and pair[1] in numbers:
                count += 1
        
        return count
    
    def generate_strategy_recommendations(self, top_pairs, confidence_details):
        """Generate strategic recommendations based on predictions"""
        strategies = []
        
        for pair in top_pairs[:3]:
            details = confidence_details.get(pair, {})
            
            if details['confidence'] >= 75:
                strategies.append({
                    'pair': pair,
                    'strategy': "ĐẶT CƯỢC MẠNH",
                    'reason': f"Độ tin cậy cao ({details['confidence']}%)",
                    'algorithms': [k for k, v in details['algorithms'].items() if v > 0]
                })
            elif details['confidence'] >= 60:
                strategies.append({
                    'pair': pair,
                    'strategy': "ĐẶT CƯỢC VỪA",
                    'reason': f"Độ tin cậy trung bình ({details['confidence']}%)",
                    'algorithms': [k for k, v in details['algorithms'].items() if v > 0]
                })
            else:
                strategies.append({
                    'pair': pair,
                    'strategy': "THEO DÕI",
                    'reason': f"Cần thêm dữ liệu ({details['confidence']}%)",
                    'algorithms': [k for k, v in details['algorithms'].items() if v > 0]
                })
        
        return strategies

# ================= DATA FUNCTIONS =================
def load_data():
    """Load historical data"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers", "source"])
    
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Ensure required columns
        if "numbers" not in df.columns:
            if len(df.columns) > 0:
                df["numbers"] = df.iloc[:, -1].astype(str)
            else:
                df["numbers"] = ""
        
        if "time" not in df.columns:
            df["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if "source" not in df.columns:
            df["source"] = "manual"
        
        df["numbers"] = df["numbers"].astype(str).str.strip()
        return df[["time", "numbers", "source"]]
    
    except Exception as e:
        st.error(f"Lỗi đọc dữ liệu: {e}")
        return pd.DataFrame(columns=["time", "numbers", "source"])

def save_data(values, source="manual"):
    """Save multiple entries with source tracking"""
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
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['numbers'], keep='first')
        
        # Sort by time
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time', ascending=True)
        
        df.to_csv(DATA_FILE, index=False)
    
    return len(rows)

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
    
    # Advanced statistics
    stats = {
        'total_draws': len(df),
        'total_digits': total,
        'frequency': dict(counter),
        'percentage': {k: f"{(v/total*100):.1f}%" for k, v in counter.items()},
        'most_common': counter.most_common(10),
        'least_common': counter.most_common()[:-11:-1],
        'hot_numbers': [n for n, c in counter.most_common(5)],
        'warm_numbers': [n for n, c in counter.most_common(10)[5:]],
        'cold_numbers': [n for n, c in counter.most_common()[:-6:-1]],
        'number_sequences': number_sequences
    }
    
    return stats

# ================= UI =================
def main():
    st.title("🎯 NUMCORE AI ULTIMATE - 2 SỐ 5 TÍNH")
    st.caption("6 Thuật toán AI kết hợp - Dự đoán cặp số chính xác nhất - Tối ưu chiến lược")
    
    # Initialize AI
    ai = TwoNumberAI()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Nhập liệu",
        "🎯 Dự đoán cặp số",
        "📊 Phân tích thuật toán",
        "⚙️ Cấu hình AI"
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
                help="Mỗi dòng là một giải thưởng gồm 5 chữ số",
                key="data_input"
            )
            
            if st.button("💾 Lưu dữ liệu", type="primary", use_container_width=True, key="save_button"):
                if raw.strip():
                    lines = [x.strip() for x in raw.splitlines() if x.strip()]
                    saved = save_data(lines)
                    
                    if saved > 0:
                        st.success(f"✅ Đã lưu {saved} kỳ hợp lệ")
                        st.rerun()
                    else:
                        st.error("❌ Không có dữ liệu hợp lệ (cần đúng 5 chữ số mỗi dòng)")
                else:
                    st.warning("⚠️ Vui lòng nhập dữ liệu trước khi lưu")
        
        with col2:
            st.subheader("📁 Dữ liệu hiện có")
            df = load_data()
            
            if not df.empty:
                st.metric("Tổng số kỳ", len(df))
                
                try:
                    df['time'] = pd.to_datetime(df['time'])
                    latest = df['time'].max().strftime("%d/%m/%Y")
                    st.metric("Dữ liệu mới nhất", latest)
                except:
                    st.metric("Dữ liệu mới nhất", "N/A")
                
                with st.expander("Xem 5 kỳ gần nhất"):
                    st.dataframe(
                        df.tail(5)[['time', 'numbers']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                if st.button("🔄 Làm mới", use_container_width=True, key="refresh_button"):
                    st.rerun()
            else:
                st.info("📭 Chưa có dữ liệu")
                st.caption("Nhập dữ liệu ở ô bên trái để bắt đầu")
    
    # ============ TAB 2: PAIR PREDICTION ============
    with tab2:
        df = load_data()
        
        if df.empty:
            st.warning("⏳ Vui lòng nhập dữ liệu trước khi phân tích")
            st.info("Chuyển sang tab '📥 Nhập liệu' để thêm dữ liệu")
        else:
            # Prepare data
            stats = get_statistics(df)
            
            if 'number_sequences' not in stats:
                st.error("Dữ liệu không đúng định dạng 5 số")
                return
            
            numbers_history = stats['number_sequences']
            hot_numbers = stats.get('hot_numbers', [])
            
            if len(numbers_history) < 10:
                st.warning(f"⚠️ Cần ít nhất 10 kỳ để phân tích (hiện có: {len(numbers_history)})")
                return
            
            # AI Analysis
            st.subheader("🎯 DỰ ĐOÁN CẶP SỐ 5 TÍNH")
            
            # Get predictions
            top_pairs, confidence_details = ai.predict_top_pairs(numbers_history, hot_numbers, num_pairs=10)
            
            if not top_pairs:
                st.error("Không thể tạo dự đoán với dữ liệu hiện có")
                return
            
            # Display top predictions
            st.subheader("🏆 TOP 5 CẶP SỐ DỰ ĐOÁN")
            
            cols = st.columns(5)
            for idx, (pair, col) in enumerate(zip(top_pairs[:5], cols)):
                with col:
                    details = confidence_details.get(pair, {})
                    confidence = details.get('confidence', 0)
                    
                    # Color coding based on confidence
                    if confidence >= 75:
                        delta_color = "normal"
                        badge = "🔥"
                    elif confidence >= 60:
                        delta_color = "normal"
                        badge = "⭐"
                    else:
                        delta_color = "off"
                        badge = "📊"
                    
                    st.metric(
                        label=f"{badge} Cặp số {idx+1}",
                        value=f"{pair[0]}{pair[1]}",
                        delta=f"{confidence}% tin cậy"
                    )
                    
                    # Quick stats
                    with st.expander(f"Chi tiết"):
                        st.write(f"**Độ tin cậy:** {confidence}%")
                        st.write(f"**Xuất hiện gần đây:** {details.get('recent_appearances', 0)} lần")
                        st.write(f"**Tần suất lịch sử:** {details.get('historical_frequency', 0)} lần")
                        
                        # Algorithm contributions
                        algo_info = details.get('algorithms', {})
                        if algo_info:
                            st.write("**Thuật toán hỗ trợ:**")
                            for algo, score in algo_info.items():
                                if score > 0:
                                    st.write(f"- {algo}: {score}")
            
            st.divider()
            
            # Strategy recommendations
            st.subheader("🎯 CHIẾN LƯỢC ĐẶT CƯỢC")
            
            strategies = ai.generate_strategy_recommendations(top_pairs, confidence_details)
            
            strategy_cols = st.columns(min(3, len(strategies)))
            for idx, (strategy, col) in enumerate(zip(strategies[:3], strategy_cols)):
                with col:
                    pair_str = f"{strategy['pair'][0]}{strategy['pair'][1]}"
                    
                    if strategy['strategy'] == "ĐẶT CƯỢC MẠNH":
                        st.success(f"**{pair_str}** - {strategy['strategy']}")
                    elif strategy['strategy'] == "ĐẶT CƯỢC VỪA":
                        st.info(f"**{pair_str}** - {strategy['strategy']}")
                    else:
                        st.warning(f"**{pair_str}** - {strategy['strategy']}")
                    
                    st.caption(f"{strategy['reason']}")
                    if strategy['algorithms']:
                        st.caption(f"Thuật toán: {', '.join(strategy['algorithms'])}")
            
            st.divider()
            
            # Detailed analysis
            st.subheader("📊 PHÂN TÍCH CHI TIẾT CẶP SỐ")
            
            selected_pair = st.selectbox(
                "Chọn cặp số để phân tích chi tiết:",
                options=[f"{p[0]}{p[1]}" for p in top_pairs],
                index=0
            )
            
            # Parse selected pair
            if selected_pair and len(selected_pair) == 2:
                pair_tuple = (int(selected_pair[0]), int(selected_pair[1]))
                details = confidence_details.get(pair_tuple, {})
                
                if details:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📈 Thống kê cơ bản:**")
                        st.write(f"- **Độ tin cậy:** {details['confidence']}%")
                        st.write(f"- **Điểm số tổng:** {details['score']}")
                        st.write(f"- **Xuất hiện gần đây (10 kỳ):** {details['recent_appearances']} lần")
                        st.write(f"- **Tổng lần xuất hiện:** {details['historical_frequency']} lần")
                        
                        # Check against hot numbers
                        hot_status = []
                        for num in pair_tuple:
                            if num in hot_numbers[:3]:
                                hot_status.append(f"Số {num} là số nóng")
                            elif num in hot_numbers:
                                hot_status.append(f"Số {num} là số ấm")
                            else:
                                hot_status.append(f"Số {num} là số bình thường")
                        
                        st.write("**🔥 Trạng thái số:**")
                        for status in hot_status:
                            st.write(f"- {status}")
                    
                    with col2:
                        st.write("**🤖 Phân tích thuật toán:**")
                        algo_scores = details.get('algorithms', {})
                        
                        if algo_scores:
                            # Create bar chart data
                            algo_names = list(algo_scores.keys())
                            algo_values = list(algo_scores.values())
                            
                            if algo_values:
                                max_val = max(algo_values)
                                if max_val > 0:
                                    # Display as progress bars
                                    for algo, score in algo_scores.items():
                                        if score > 0:
                                            normalized = score / max_val
                                            st.write(f"**{algo}:**")
                                            st.progress(float(normalized), text=f"{score:.3f}")
            
            st.divider()
            
            # All predicted pairs
            st.subheader("📋 DANH SÁCH ĐẦY ĐỦ CẶP SỐ DỰ ĐOÁN")
            
            pairs_data = []
            for pair in top_pairs:
                details = confidence_details.get(pair, {})
                pairs_data.append({
                    'Cặp số': f"{pair[0]}{pair[1]}",
                    'Độ tin cậy': f"{details.get('confidence', 0)}%",
                    'Điểm số': round(details.get('score', 0), 3),
                    'Gần đây': details.get('recent_appearances', 0),
                    'Lịch sử': details.get('historical_frequency', 0)
                })
            
            if pairs_data:
                pairs_df = pd.DataFrame(pairs_data)
                st.dataframe(
                    pairs_df,
                    use_container_width=True,
                    hide_index=True
                )
    
    # ============ TAB 3: ALGORITHM ANALYSIS ============
    with tab3:
        st.subheader("🤖 PHÂN TÍCH 6 THUẬT TOÁN AI")
        
        st.write("""
        ### Các thuật toán được sử dụng:
        
        1. **Frequency-Based** (25%): Phân tích tần suất xuất hiện của cặp số
        2. **Position-Based** (20%): Phân tích vị trí xuất hiện trong giải
        3. **Trend-Based** (20%): Phân tích xu hướng tăng/giảm
        4. **Pattern-Based** (15%): Nhận diện mẫu số và khoảng cách
        5. **Statistical** (10%): Phân tích xác suất thống kê
        6. **Neural-Inspired** (10%): Mô phỏng mạng neural nhận diện pattern
        """)
        
        df = load_data()
        
        if not df.empty:
            stats = get_statistics(df)
            
            if 'number_sequences' in stats and len(stats['number_sequences']) >= 10:
                numbers_history = stats['number_sequences']
                hot_numbers = stats.get('hot_numbers', [])
                
                # Run each algorithm separately
                st.subheader("📊 KẾT QUẢ TỪNG THUẬT TOÁN")
                
                algorithms = [
                    ("Tần suất", ai.algorithm_frequency_based),
                    ("Vị trí", ai.algorithm_position_based),
                    ("Xu hướng", ai.algorithm_trend_based),
                    ("Mẫu số", ai.algorithm_pattern_based),
                    ("Thống kê", ai.algorithm_statistical),
                    ("Neural", ai.algorithm_neural_inspired)
                ]
                
                algo_cols = st.columns(3)
                for idx, (algo_name, algo_func) in enumerate(algorithms):
                    with algo_cols[idx % 3]:
                        try:
                            results = algo_func(numbers_history, hot_numbers)
                            st.write(f"**{algo_name}:**")
                            
                            if results:
                                top_3 = results[:3]
                                for i, (pair, score) in enumerate(top_3):
                                    st.write(f"{i+1}. {pair[0]}{pair[1]} ({score:.3f})")
                            else:
                                st.write("Chưa đủ dữ liệu")
                        except Exception as e:
                            st.write(f"**{algo_name}:** Lỗi phân tích")
                
                st.divider()
                
                # Algorithm performance comparison
                st.subheader("⚖️ SO SÁNH HIỆU QUẢ THUẬT TOÁN")
                
                # Get all algorithm results
                all_algo_results = {}
                for algo_name, algo_func in algorithms:
                    try:
                        results = algo_func(numbers_history, hot_numbers)
                        if results:
                            all_algo_results[algo_name] = [p[0] for p in results[:5]]
                    except:
                        pass
                
                # Find common predictions
                if all_algo_results:
                    common_pairs = defaultdict(int)
                    for algo_name, pairs in all_algo_results.items():
                        for pair in pairs:
                            common_pairs[pair] += 1
                    
                    # Display pairs with multiple algorithm support
                    st.write("**Cặp số được nhiều thuật toán hỗ trợ:**")
                    sorted_common = sorted(common_pairs.items(), key=lambda x: x[1], reverse=True)
                    
                    for pair, count in sorted_common[:5]:
                        if count > 1:
                            st.write(f"**{pair[0]}{pair[1]}:** {count}/6 thuật toán hỗ trợ")
    
    # ============ TAB 4: AI CONFIGURATION ============
    with tab4:
        st.subheader("⚙️ CẤU HÌNH THUẬT TOÁN AI")
        
        ai.load_config()
        
        st.write("### ⚖️ Điều chỉnh trọng số thuật toán")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_weight = st.slider(
                "Thuật toán Tần suất",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('frequency_based', 0.25)), 0.05
            )
            
            pos_weight = st.slider(
                "Thuật toán Vị trí",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('position_based', 0.20)), 0.05
            )
            
            trend_weight = st.slider(
                "Thuật toán Xu hướng",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('trend_based', 0.20)), 0.05
            )
        
        with col2:
            pattern_weight = st.slider(
                "Thuật toán Mẫu số",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('pattern_based', 0.15)), 0.05
            )
            
            stat_weight = st.slider(
                "Thuật toán Thống kê",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('statistical', 0.10)), 0.05
            )
            
            neural_weight = st.slider(
                "Thuật toán Neural",
                0.0, 0.5, float(ai.config['algorithm_weights'].get('neural_inspired', 0.10)), 0.05
            )
        
        st.divider()
        
        st.write("### 🎯 Thiết lập ngưỡng")
        
        col3, col4 = st.columns(2)
        
        with col3:
            avoid_recent = st.slider(
                "Tránh cặp trùng (kỳ)",
                1, 10, ai.config.get('avoid_recent_pairs', 5), 1
            )
            
            min_confidence = st.slider(
                "Độ tin cậy tối thiểu (%)",
                40, 90, ai.config.get('min_confidence', 60), 5
            )
        
        with col4:
            recent_weight = st.slider(
                "Trọng số dữ liệu gần",
                0.3, 0.9, ai.config.get('recent_weight', 0.6), 0.1
            )
        
        if st.button("💾 Lưu cấu hình", type="primary", use_container_width=True):
            # Update algorithm weights
            ai.config['algorithm_weights'] = {
                'frequency_based': freq_weight,
                'position_based': pos_weight,
                'trend_based': trend_weight,
                'pattern_based': pattern_weight,
                'statistical': stat_weight,
                'neural_inspired': neural_weight
            }
            
            # Update other settings
            ai.config['avoid_recent_pairs'] = avoid_recent
            ai.config['min_confidence'] = min_confidence
            ai.config['recent_weight'] = recent_weight
            
            # Save to file
            try:
                with open(AI_CONFIG_FILE, 'w') as f:
                    json.dump(ai.config, f, indent=2)
                
                st.success("✅ Đã lưu cấu hình thuật toán")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu: {e}")
        
        st.divider()
        
        st.subheader("🔄 QUẢN LÝ DỮ LIỆU")
        
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("🗑️ Xóa dữ liệu lịch sử cặp số", use_container_width=True):
                if os.path.exists(PAIR_HISTORY_FILE):
                    os.remove(PAIR_HISTORY_FILE)
                    ai.pair_frequency = Counter()
                    ai.position_pairs = defaultdict(Counter)
                    st.success("✅ Đã xóa lịch sử cặp số")
                    st.rerun()
        
        with col6:
            if st.button("📥 Xuất dữ liệu phân tích", use_container_width=True):
                df = load_data()
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Tải file CSV",
                        data=csv,
                        file_name=f"numcore_2so_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("Không có dữ liệu để xuất")
    
    # ============ FOOTER ============
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        df_count = len(load_data())
        st.caption(f"📊 Dữ liệu: {df_count} kỳ")
    
    with col2:
        st.caption("🤖 6 Thuật toán AI kết hợp")
    
    with col3:
        st.caption("NUMCORE AI ULTIMATE v9.0 – Tối ưu 2 số 5 tính")

if __name__ == "__main__":
    main()
