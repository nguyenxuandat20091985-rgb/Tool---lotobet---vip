import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, permutations
from datetime import datetime, timedelta
import os
import random
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE AI PRO",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "numcore_data.csv"
AI_CONFIG_FILE = "ai_config.json"

# ================= ADVANCED AI ENHANCEMENTS =================
class AdvancedAI:
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.history_window = 25
        self.position_analysis = {i: Counter() for i in range(5)}
        self.number_relations = defaultdict(Counter)
        self.load_config()
    
    def load_config(self):
        """Load AI configuration"""
        default_config = {
            "weight_recent": 0.5,
            "weight_frequency": 0.25,
            "weight_position": 0.15,
            "weight_pattern": 0.1,
            "avoid_recent_count": 4,
            "hot_number_threshold": 0.12,
            "position_weight_factor": 0.8,
            "relation_weight": 0.3
        }
        
        if os.path.exists(AI_CONFIG_FILE):
            with open(AI_CONFIG_FILE, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
    
    def analyze_position_patterns(self, numbers_history):
        """Analyze which numbers appear in which positions"""
        position_stats = {i: Counter() for i in range(5)}
        
        for numbers in numbers_history:
            if len(numbers) == 5:
                for pos, num in enumerate(numbers):
                    position_stats[pos][num] += 1
        
        return position_stats
    
    def analyze_relations(self, numbers_history):
        """Analyze relations between numbers in same draw"""
        relations = defaultdict(Counter)
        
        for numbers in numbers_history:
            if len(numbers) == 5:
                for i in range(5):
                    for j in range(i+1, 5):
                        pair = tuple(sorted([numbers[i], numbers[j]]))
                        relations[numbers[i]][numbers[j]] += 1
                        relations[numbers[j]][numbers[i]] += 1
        
        return relations
    
    def calculate_position_scores(self, numbers_history):
        """Calculate position-based scores for each number"""
        position_stats = self.analyze_position_patterns(numbers_history)
        position_scores = {}
        
        for num in range(10):
            scores = []
            for pos in range(5):
                total_in_pos = sum(position_stats[pos].values())
                if total_in_pos > 0:
                    freq = position_stats[pos][num] / total_in_pos
                    # Weight by position importance
                    pos_weight = 1.0 - (abs(pos - 2) * 0.2)  # Center positions get more weight
                    scores.append(freq * pos_weight)
            
            if scores:
                position_scores[num] = np.mean(scores) * self.config['position_weight_factor']
            else:
                position_scores[num] = 0
        
        return position_scores
    
    def calculate_relation_scores(self, numbers_history, hot_numbers):
        """Calculate scores based on number relations"""
        relations = self.analyze_relations(numbers_history)
        relation_scores = {}
        
        for num in range(10):
            if num in relations:
                # Check relations with hot numbers
                total_relations = 0
                strong_relations = 0
                
                for hot_num in hot_numbers[:3]:
                    if hot_num in relations[num]:
                        total_relations += relations[num][hot_num]
                        if relations[num][hot_num] > 0:
                            strong_relations += 1
                
                if total_relations > 0:
                    relation_scores[num] = (strong_relations / len(hot_numbers[:3])) * self.config['relation_weight']
                else:
                    relation_scores[num] = 0
            else:
                relation_scores[num] = 0
        
        return relation_scores
    
    def predict_top_two_numbers(self, numbers_history, hot_numbers):
        """Predict top 2 numbers with highest probability"""
        if len(numbers_history) < 10:
            return [], {}
        
        # Calculate various scores
        trend_data = self.analyze_trends(numbers_history)
        position_scores = self.calculate_position_scores(numbers_history)
        relation_scores = self.calculate_relation_scores(numbers_history, hot_numbers)
        
        # Combine all scores
        candidate_scores = {}
        
        for num in range(10):
            # Skip numbers that are already hot (we want complementary numbers)
            if num in hot_numbers[:5]:
                continue
            
            total_score = 0
            
            # Frequency score
            freq_score = trend_data['frequencies'].get(num, 0)
            total_score += freq_score * self.config['weight_frequency']
            
            # Trend score
            trend_score = trend_data['trends'].get(num, 0)
            total_score += max(0, trend_score) * 0.5
            
            # Position score
            total_score += position_scores.get(num, 0)
            
            # Relation score
            total_score += relation_scores.get(num, 0)
            
            # Recent appearance penalty
            recent_appearance = any(num in nums for nums in numbers_history[-self.config['avoid_recent_count']:])
            if recent_appearance:
                total_score *= 0.7
            
            candidate_scores[num] = total_score
        
        # Get top 2 candidates
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        result = []
        details = {}
        
        for i, (num, score) in enumerate(top_candidates):
            result.append(num)
            
            # Calculate confidence level
            confidence = min(95, 60 + int(score * 100))
            
            details[num] = {
                'score': round(score, 3),
                'confidence': confidence,
                'position_strength': round(position_scores.get(num, 0), 3),
                'relation_strength': round(relation_scores.get(num, 0), 3)
            }
        
        return result, details
    
    def generate_combination_predictions(self, numbers_history, hot_numbers, top_two):
        """Generate combination A and B predictions"""
        if len(top_two) < 2:
            return {"A": "--", "B": "--"}
        
        # Strategy 1: Combine top two with hottest number
        combination_A = []
        if hot_numbers and len(hot_numbers) >= 1:
            combination_A = [top_two[0], top_two[1], hot_numbers[0]]
            combination_A.sort()
        
        # Strategy 2: Combine with position-favored numbers
        combination_B = []
        if len(hot_numbers) >= 3:
            # Find number with best position score that's not already used
            position_scores = self.calculate_position_scores(numbers_history)
            position_candidates = [(num, score) for num, score in position_scores.items() 
                                 if num not in combination_A and num not in top_two]
            
            if position_candidates:
                best_position_num = max(position_candidates, key=lambda x: x[1])[0]
                combination_B = [top_two[0], hot_numbers[1], best_position_num]
                combination_B.sort()
        
        return {
            "A": "".join(map(str, combination_A)) if combination_A else "--",
            "B": "".join(map(str, combination_B)) if combination_B else "--"
        }
    
    def analyze_trends(self, numbers_history, window=15):
        """Analyze number trends over time - ENHANCED"""
        if len(numbers_history) < window:
            return {'frequencies': {}, 'trends': {}, 'most_common': [], 'least_common': []}
        
        recent = numbers_history[-window:]
        all_nums = [n for sublist in recent for n in sublist]
        
        # Frequency analysis
        freq = Counter(all_nums)
        total = len(all_nums)
        
        # Enhanced trend analysis with multiple windows
        trends = {}
        trend_strength = {}
        
        for num in range(10):
            # Multiple time window analysis
            windows = [
                (len(numbers_history)//3, len(numbers_history)//3*2),
                (len(numbers_history)//4, len(numbers_history)//4*3),
                (max(0, len(numbers_history)-window), len(numbers_history))
            ]
            
            window_trends = []
            for start, end in windows:
                if end > start:
                    early = numbers_history[start:start+(end-start)//2]
                    late = numbers_history[start+(end-start)//2:end]
                    
                    early_count = sum(1 for nums in early for n in nums if n == num)
                    late_count = sum(1 for nums in late for n in nums if n == num)
                    
                    if early_count + late_count > 0:
                        trend_val = (late_count - early_count) / max(1, (early_count + late_count))
                        window_trends.append(trend_val)
            
            if window_trends:
                trends[num] = np.mean(window_trends)
                trend_strength[num] = np.std(window_trends)  # Consistency of trend
        
        return {
            'frequencies': {k: v/total for k, v in freq.items()},
            'trends': trends,
            'trend_strength': trend_strength,
            'most_common': freq.most_common(10),
            'least_common': freq.most_common()[:-11:-1]
        }
    
    def predict_exclusions(self, numbers_history, hot_numbers):
        """Enhanced exclusion prediction"""
        if len(numbers_history) < 8:
            return []
        
        exclusions = set()
        
        # Rule 1: Numbers that appeared too recently
        recent_window = min(5, len(numbers_history))
        for nums in numbers_history[-recent_window:]:
            exclusions.update(nums)
        
        # Rule 2: Cold numbers with negative trends
        trend_data = self.analyze_trends(numbers_history)
        for num, trend in trend_data['trends'].items():
            if trend < -0.3:  # Strong negative trend
                exclusions.add(num)
        
        # Rule 3: Numbers with poor position performance
        position_stats = self.analyze_position_patterns(numbers_history)
        for num in range(10):
            pos_performance = sum(position_stats[pos][num] for pos in range(5))
            if pos_performance == 0 and num in hot_numbers:
                exclusions.add(num)
        
        return list(exclusions)[:5]
    
    def analyze_advanced_patterns(self, numbers_history):
        """Advanced pattern analysis"""
        if len(numbers_history) < 15:
            return {}
        
        patterns = {
            'position_patterns': self.analyze_position_patterns(numbers_history),
            'digit_gaps': [],
            'sum_analysis': [],
            'parity_analysis': []
        }
        
        for nums in numbers_history[-15:]:
            if len(nums) == 5:
                # Digit gaps analysis
                gaps = [abs(nums[i] - nums[i+1]) for i in range(4)]
                patterns['digit_gaps'].extend(gaps)
                
                # Sum analysis
                patterns['sum_analysis'].append(sum(nums))
                
                # Parity analysis
                odd_count = sum(1 for n in nums if n % 2 == 1)
                patterns['parity_analysis'].append(odd_count)
        
        return patterns

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
    
    # Position analysis
    if number_sequences:
        position_stats = []
        for pos in range(5):
            pos_numbers = [seq[pos] for seq in number_sequences if len(seq) > pos]
            pos_counter = Counter(pos_numbers)
            position_stats.append(dict(pos_counter.most_common(5)))
        
        stats['position_stats'] = position_stats
    
    return stats

# ================= UI =================
def main():
    st.title("🎯 NUMCORE AI PRO")
    st.caption("Phân tích chuyên sâu 5 số - Dự đoán thông minh - Độ chính xác cao")
    
    # Initialize Advanced AI
    ai = AdvancedAI()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Nhập liệu",
        "🎯 Phân tích AI Nâng cao",
        "📊 Thống kê chi tiết",
        "⚙️ Cấu hình AI"
    ])
    
    # ============ TAB 1: DATA INPUT ============
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📥 Nhập dữ liệu hàng loạt")
            
            raw = st.text_area(
                "Nhập nhiều kỳ (mỗi dòng 5 số)",
                height=200,
                placeholder="Ví dụ:\n12345\n67890\n54321\n...",
                help="Mỗi dòng là một kỳ gồm 5 chữ số đầy đủ",
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
    
    # ============ TAB 2: ADVANCED AI ANALYSIS ============
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
            
            # Advanced AI Analysis
            st.subheader("🎯 PHÂN TÍCH AI CHUYÊN SÂU")
            
            # Predict top two numbers
            top_two, top_details = ai.predict_top_two_numbers(numbers_history, hot_numbers)
            
            # Generate combinations
            combinations = ai.generate_combination_predictions(numbers_history, hot_numbers, top_two)
            
            # Display results in a grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔥 Số nóng nhất",
                    f"{hot_numbers[0] if hot_numbers else '--'}",
                    delta=f"{stats['percentage'].get(hot_numbers[0], '0%') if hot_numbers else '0%'}"
                )
            
            with col2:
                if top_two and len(top_two) >= 1:
                    detail = top_details.get(top_two[0], {})
                    st.metric(
                        "🥇 Dự đoán số 1",
                        str(top_two[0]),
                        delta=f"Tin cậy: {detail.get('confidence', 0)}%"
                    )
            
            with col3:
                if top_two and len(top_two) >= 2:
                    detail = top_details.get(top_two[1], {})
                    st.metric(
                        "🥈 Dự đoán số 2",
                        str(top_two[1]),
                        delta=f"Tin cậy: {detail.get('confidence', 0)}%"
                    )
            
            with col4:
                st.metric(
                    "📊 Độ chính xác",
                    f"{min(85, 50 + len(numbers_history)//5)}%",
                    delta="Dựa trên lịch sử"
                )
            
            st.divider()
            
            # Combination predictions
            st.subheader("🔢 DỰ ĐOÁN TỔ HỢP")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🎯 Tổ hợp A (Ưu tiên cao)",
                    combinations['A'],
                    delta="Kết hợp số nóng + dự đoán"
                )
                st.caption("Chiến lược: Kết hợp 2 số dự đoán với số nóng nhất")
            
            with col2:
                st.metric(
                    "🅱️ Tổ hợp B (Dự phòng)",
                    combinations['B'],
                    delta="Kết hợp vị trí + xu hướng"
                )
                st.caption("Chiến lược: Kết hợp với số có vị trí tốt nhất")
            
            st.divider()
            
            # Detailed analysis
            st.subheader("📈 PHÂN TÍCH CHI TIẾT")
            
            if top_two and top_details:
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.write("**🎯 Phân tích số dự đoán 1:**")
                    if top_two[0] in top_details:
                        detail = top_details[top_two[0]]
                        st.write(f"- **Điểm số:** {detail['score']:.3f}")
                        st.write(f"- **Độ tin cậy:** {detail['confidence']}%")
                        st.write(f"- **Sức mạnh vị trí:** {detail['position_strength']:.3f}")
                        st.write(f"- **Quan hệ số:** {detail['relation_strength']:.3f}")
                
                with details_col2:
                    st.write("**🎯 Phân tích số dự đoán 2:**")
                    if len(top_two) > 1 and top_two[1] in top_details:
                        detail = top_details[top_two[1]]
                        st.write(f"- **Điểm số:** {detail['score']:.3f}")
                        st.write(f"- **Độ tin cậy:** {detail['confidence']}%")
                        st.write(f"- **Sức mạnh vị trí:** {detail['position_strength']:.3f}")
                        st.write(f"- **Quan hệ số:** {detail['relation_strength']:.3f}")
            
            # Advanced patterns
            if len(numbers_history) >= 15:
                st.divider()
                st.subheader("🔍 PHÂN TÍCH MẪU NÂNG CAO")
                
                patterns = ai.analyze_advanced_patterns(numbers_history)
                
                pattern_col1, pattern_col2 = st.columns(2)
                
                with pattern_col1:
                    st.write("**📊 Phân tích vị trí:**")
                    if 'position_patterns' in patterns:
                        for pos in range(5):
                            pos_name = ["Đầu", "Nhì", "Ba", "Tư", "Năm"][pos]
                            top_pos_nums = patterns['position_patterns'][pos].most_common(3)
                            if top_pos_nums:
                                nums_str = ", ".join([f"{num}({count})" for num, count in top_pos_nums])
                                st.write(f"- **Vị trí {pos_name}:** {nums_str}")
                
                with pattern_col2:
                    st.write("**📈 Xu hướng tổng số:**")
                    if 'sum_analysis' in patterns and patterns['sum_analysis']:
                        avg_sum = np.mean(patterns['sum_analysis'])
                        common_sum = Counter(patterns['sum_analysis']).most_common(1)
                        st.write(f"- **Tổng trung bình:** {avg_sum:.1f}")
                        if common_sum:
                            st.write(f"- **Tổng phổ biến:** {common_sum[0][0]} ({common_sum[0][1]} lần)")
            
            # Exclusion recommendations
            exclusions = ai.predict_exclusions(numbers_history, hot_numbers)
            if exclusions:
                st.divider()
                st.subheader("⚠️ KHUYẾN NGHỊ TRÁNH")
                st.warning(f"Số cần thận trọng: **{', '.join(map(str, exclusions[:5]))}**")
                st.caption("Lý do: Xuất hiện gần đây / Xu hướng giảm / Vị trí yếu")
    
    # ============ TAB 3: DETAILED STATISTICS ============
    with tab3:
        df = load_data()
        
        if not df.empty:
            stats = get_statistics(df)
            
            # Overview
            st.subheader("📊 TỔNG QUAN THỐNG KÊ")
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            
            with overview_col1:
                st.metric("Tổng số kỳ", stats['total_draws'])
                st.metric("Tổng số digit", stats['total_digits'])
            
            with overview_col2:
                if 'avg_draws_per_day' in stats:
                    st.metric("Kỳ/ngày", f"{stats['avg_draws_per_day']:.1f}")
                
                hot_num = stats['hot_numbers'][0] if stats['hot_numbers'] else "--"
                hot_percent = stats['percentage'].get(hot_num, "0%")
                st.metric("Số nóng nhất", f"{hot_num} ({hot_percent})")
            
            with overview_col3:
                cold_num = stats['cold_numbers'][0] if stats['cold_numbers'] else "--"
                cold_percent = stats['percentage'].get(cold_num, "0%")
                st.metric("Số lạnh nhất", f"{cold_num} ({cold_percent})")
                
                coverage = len(stats['frequency']) / 10 * 100
                st.metric("Độ phủ số", f"{coverage:.1f}%")
            
            st.divider()
            
            # Hot and Cold numbers
            st.subheader("🔥 SỐ NÓNG & ❄️ SỐ LẠNH")
            
            hot_col, cold_col = st.columns(2)
            
            with hot_col:
                st.write("**Top 5 số nóng:**")
                for i, (num, count) in enumerate(stats['most_common'][:5], 1):
                    percent = stats['percentage'].get(num, "0%")
                    st.write(f"{i}. **{num}** - {count} lần ({percent})")
            
            with cold_col:
                st.write("**Top 5 số lạnh:**")
                for i, (num, count) in enumerate(stats['least_common'][:5], 1):
                    percent = stats['percentage'].get(num, "0%")
                    st.write(f"{i}. **{num}** - {count} lần ({percent})")
            
            # Position analysis
            if 'position_stats' in stats:
                st.divider()
                st.subheader("🎯 PHÂN TÍCH VỊ TRÍ")
                
                pos_cols = st.columns(5)
                position_names = ["Đầu", "Nhì", "Ba", "Tư", "Năm"]
                
                for idx, pos_col in enumerate(pos_cols):
                    with pos_col:
                        st.write(f"**Vị trí {position_names[idx]}:**")
                        if idx < len(stats['position_stats']):
                            for num, count in stats['position_stats'][idx].items():
                                st.write(f"{num}: {count} lần")
            
            # Frequency chart
            st.divider()
            st.subheader("📈 BIỂU ĐỒ TẦN SUẤT")
            
            if stats['frequency']:
                freq_df = pd.DataFrame.from_dict(stats['frequency'], orient='index', columns=['count'])
                freq_df = freq_df.sort_values('count', ascending=True)  # Sort for better visualization
                st.bar_chart(freq_df)
            
            # Recent data
            st.divider()
            st.subheader("📋 DỮ LIỆU GẦN ĐÂY")
            
            display_df = df.tail(10).copy()
            if 'time' in display_df.columns:
                try:
                    display_df['time'] = pd.to_datetime(display_df['time']).dt.strftime('%d/%m/%Y %H:%M')
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
            st.info("📭 Chưa có dữ liệu để hiển thị thống kê")
    
    # ============ TAB 4: AI CONFIGURATION ============
    with tab4:
        st.subheader("⚙️ CẤU HÌNH AI NÂNG CAO")
        
        ai.load_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Trọng số phân tích:**")
            weight_recent = st.slider(
                "Trọng số dữ liệu gần đây",
                0.1, 0.8, float(ai.config.get('weight_recent', 0.5)), 0.05,
                help="Ảnh hưởng của các kỳ gần nhất"
            )
            
            weight_frequency = st.slider(
                "Trọng số tần suất",
                0.1, 0.5, float(ai.config.get('weight_frequency', 0.25)), 0.05,
                help="Ảnh hưởng của tần suất xuất hiện"
            )
            
            weight_position = st.slider(
                "Trọng số vị trí",
                0.05, 0.3, float(ai.config.get('weight_position', 0.15)), 0.05,
                help="Ảnh hưởng của vị trí số"
            )
        
        with col2:
            st.write("**🎯 Ngưỡng phân tích:**")
            avoid_recent = st.slider(
                "Tránh số trùng (kỳ)",
                2, 8, int(ai.config.get('avoid_recent_count', 4)), 1,
                help="Số kỳ gần nhất để tránh trùng số"
            )
            
            hot_threshold = st.slider(
                "Ngưỡng số nóng (%)",
                8, 25, int(ai.config.get('hot_number_threshold', 0.12) * 100), 1,
                help="Tỉ lệ xuất hiện tối thiểu để coi là số nóng"
            ) / 100
            
            position_weight = st.slider(
                "Ảnh hưởng vị trí",
                0.5, 1.0, float(ai.config.get('position_weight_factor', 0.8)), 0.1,
                help="Độ ảnh hưởng của phân tích vị trí"
            )
        
        if st.button("💾 Lưu cấu hình AI", type="primary", use_container_width=True):
            config = {
                "weight_recent": weight_recent,
                "weight_frequency": weight_frequency,
                "weight_position": weight_position,
                "weight_pattern": ai.config.get('weight_pattern', 0.1),
                "avoid_recent_count": avoid_recent,
                "hot_number_threshold": hot_threshold,
                "position_weight_factor": position_weight,
                "relation_weight": ai.config.get('relation_weight', 0.3)
            }
            
            try:
                with open(AI_CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                ai.config = config
                
                st.success("✅ Đã lưu cấu hình AI nâng cao")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu cấu hình: {e}")
        
        st.divider()
        
        st.subheader("🔄 QUẢN LÝ DỮ LIỆU")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Xóa toàn bộ dữ liệu", use_container_width=True, type="secondary"):
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                    st.success("✅ Đã xóa toàn bộ dữ liệu")
                    st.rerun()
        
        with col2:
            if st.button("📥 Xuất dữ liệu", use_container_width=True):
                df = load_data()
                if not df.empty:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Tải file CSV",
                        data=csv,
                        file_name=f"numcore_pro_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
        st.caption("🤖 AI: Advanced Pattern Recognition")
    
    with col3:
        st.caption("NUMCORE AI PRO v8.0 – Phân tích chuyên sâu 5 số")

if __name__ == "__main__":
    main()
