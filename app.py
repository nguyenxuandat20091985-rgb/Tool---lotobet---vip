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

# ================= CONFIG =================
st.set_page_config(
    page_title="NUMCORE AI+",
    layout="wide",
    page_icon="🔷"
)

DATA_FILE = "numcore_data.csv"
AI_CONFIG_FILE = "ai_config.json"

# ================= AI ENHANCEMENTS =================
class EnhancedAI:
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.history_window = 20
        self.load_config()
    
    def load_config(self):
        """Load AI configuration"""
        default_config = {
            "weight_recent": 0.6,
            "weight_frequency": 0.3,
            "weight_pattern": 0.1,
            "avoid_recent_count": 3,
            "hot_number_threshold": 0.15
        }
        
        if os.path.exists(AI_CONFIG_FILE):
            with open(AI_CONFIG_FILE, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
    
    def analyze_trends(self, numbers_history, window=10):
        """Analyze number trends over time"""
        if len(numbers_history) < window:
            return {}
        
        recent = numbers_history[-window:]
        all_nums = [n for sublist in recent for n in sublist]
        
        # Frequency analysis
        freq = Counter(all_nums)
        total = len(all_nums)
        
        # Trend direction (increasing/decreasing frequency)
        trends = {}
        for num in range(10):
            early_count = sum(1 for nums in numbers_history[-window:-window//2] for n in nums if n == num)
            late_count = sum(1 for nums in numbers_history[-window//2:] for n in nums if n == num)
            
            if early_count + late_count > 0:
                trend = (late_count - early_count) / max(1, (early_count + late_count))
                trends[num] = trend
        
        return {
            'frequencies': {k: v/total for k, v in freq.items()},
            'trends': trends,
            'most_common': freq.most_common(3),
            'least_common': freq.most_common()[:-4:-1]
        }
    
    def predict_exclusions(self, numbers_history, top_numbers):
        """Predict which numbers to exclude based on patterns"""
        if len(numbers_history) < 5:
            return []
        
        exclusions = []
        
        # Rule 1: Avoid numbers that appeared in last N draws
        recent_numbers = set()
        for nums in numbers_history[-self.config['avoid_recent_count']:]:
            recent_numbers.update(nums)
        
        # Rule 2: Avoid cold numbers (below threshold)
        trend_data = self.analyze_trends(numbers_history)
        cold_numbers = [num for num, freq in trend_data['frequencies'].items() 
                       if freq < self.config['hot_number_threshold']]
        
        exclusions = list(recent_numbers.union(cold_numbers))
        return [n for n in exclusions if n in top_numbers][:3]
    
    def generate_ai_pick(self, numbers_history, top_numbers):
        """Generate AI-enhanced picks"""
        if not numbers_history:
            return "--"
        
        trend_data = self.analyze_trends(numbers_history)
        
        # Weighted selection strategy
        candidate_scores = {}
        
        for num in range(10):
            if num in top_numbers:
                continue
                
            score = 0
            
            # Frequency weight
            freq_score = trend_data['frequencies'].get(num, 0)
            score += freq_score * self.config['weight_frequency']
            
            # Trend weight (positive trend gets higher score)
            trend_score = trend_data['trends'].get(num, 0)
            score += max(0, trend_score) * self.config['weight_pattern']
            
            # Recent avoidance penalty
            if num in set().union(*numbers_history[-2:]):
                score *= 0.5
            
            candidate_scores[num] = score
        
        # Select top 2 candidates
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if len(top_candidates) >= 2:
            return f"{top_candidates[0][0]}{top_candidates[1][0]}"
        elif len(top_candidates) == 1:
            # Find complementary number
            complement = random.choice([n for n in range(10) if n != top_candidates[0][0] and n not in top_numbers])
            return f"{top_candidates[0][0]}{complement}"
        
        return "--"
    
    def analyze_patterns(self, numbers_history):
        """Analyze digit patterns and sequences"""
        if len(numbers_history) < 10:
            return {}
        
        patterns = {
            'consecutive_pairs': Counter(),
            'digit_sum_trend': [],
            'odd_even_ratio': []
        }
        
        for nums in numbers_history[-10:]:
            # Consecutive pairs analysis
            for i in range(len(nums)-1):
                pair = (nums[i], nums[i+1])
                patterns['consecutive_pairs'][pair] += 1
            
            # Digit sum
            patterns['digit_sum_trend'].append(sum(nums) % 10)
            
            # Odd/even ratio
            odd_count = sum(1 for n in nums if n % 2 == 1)
            patterns['odd_even_ratio'].append(odd_count / len(nums))
        
        return patterns

# ================= DATA =================
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
        
        # Remove duplicates (same numbers in last 24 hours)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time', ascending=False)
        df = df.drop_duplicates(subset=['numbers', 'time'].dt.date, keep='first')
        
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
    for nums in df['numbers']:
        all_numbers.extend(parse_numbers(nums))
    
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
        'most_common': counter.most_common(5),
        'least_common': counter.most_common()[:-6:-1],
        'hot_numbers': [n for n, c in counter.most_common(3)],
        'cold_numbers': [n for n, c in counter.most_common()[:-4:-1]],
        'avg_draws_per_day': len(df) / max(1, (df['time'].max() - df['time'].min()).days)
    }
    
    return stats

# ================= UI =================
def main():
    st.title("🔷 NUMCORE AI+")
    st.caption("Phân tích nâng cao với AI – Dự đoán thông minh – Hiệu quả tối ưu")
    
    # Initialize AI
    ai = EnhancedAI()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Nhập liệu",
        "🎯 Phân tích AI",
        "📊 Thống kê",
        "⚙️ Cài đặt AI"
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
                help="Mỗi dòng là một kỳ gồm 5 chữ số"
            )
            
            if st.button("💾 Lưu dữ liệu", type="primary", use_container_width=True):
                lines = [x.strip() for x in raw.splitlines() if x.strip()]
                saved = save_data(lines)
                
                if saved > 0:
                    st.success(f"✅ Đã lưu {saved} kỳ hợp lệ")
                    st.rerun()
                else:
                    st.error("❌ Không có dữ liệu hợp lệ")
        
        with col2:
            st.subheader("📁 Dữ liệu hiện có")
            df = load_data()
            
            if not df.empty:
                st.metric("Tổng số kỳ", len(df))
                st.metric("Dữ liệu mới nhất", df['time'].max()[:10])
                
                if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
                    st.rerun()
            else:
                st.warning("Chưa có dữ liệu")
    
    # ============ TAB 2: AI ANALYSIS ============
    with tab2:
        df = load_data()
        
        if df.empty:
            st.warning("Vui lòng nhập dữ liệu trước khi phân tích")
        else:
            # Prepare data for AI
            numbers_history = [parse_numbers(nums) for nums in df['numbers'].tolist()]
            
            # Get top numbers
            stats = get_statistics(df)
            top_numbers = [n for n, _ in stats.get('most_common', [])[:5]]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🎯 Số trung tâm")
                if top_numbers:
                    groups = list(combinations(top_numbers[:4], 3))[:2]
                    
                    if len(groups) > 0:
                        st.metric(
                            "Tổ hợp A", 
                            "".join(map(str, groups[0])),
                            delta="Ưu tiên cao"
                        )
                    if len(groups) > 1:
                        st.metric(
                            "Tổ hợp B", 
                            "".join(map(str, groups[1])),
                            delta="Dự phòng"
                        )
            
            with col2:
                st.subheader("🧠 AI Dự đoán")
                ai_pick = ai.generate_ai_pick(numbers_history, top_numbers)
                
                st.metric(
                    "Lựa chọn AI", 
                    ai_pick,
                    delta="Độ tin cậy cao"
                )
                
                # AI confidence
                if len(numbers_history) >= 10:
                    confidence = min(85, 60 + len(numbers_history) // 10)
                    st.progress(confidence/100, text=f"Độ tin cậy: {confidence}%")
            
            with col3:
                st.subheader("⚠️ Cảnh báo AI")
                exclusions = ai.predict_exclusions(numbers_history, top_numbers)
                
                if exclusions:
                    st.warning(f"Tránh số: {', '.join(map(str, exclusions))}")
                else:
                    st.info("Không có cảnh báo đặc biệt")
            
            st.divider()
            
            # Pattern analysis
            st.subheader("📈 Phân tích mẫu số")
            patterns = ai.analyze_patterns(numbers_history)
            
            if patterns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Cặp số thường xuất hiện:**")
                    for pair, count in patterns['consecutive_pairs'].most_common(5):
                        st.write(f"`{pair[0]}{pair[1]}`: {count} lần")
                
                with col2:
                    if patterns['digit_sum_trend']:
                        avg_sum = np.mean(patterns['digit_sum_trend']) % 10
                        st.write(f"**Tổng số trung bình (mod 10):** `{avg_sum:.1f}`")
                        
                        odd_ratio = np.mean(patterns['odd_even_ratio'])
                        st.write(f"**Tỉ lệ số lẻ:** `{odd_ratio:.1%}`")
    
    # ============ TAB 3: STATISTICS ============
    with tab3:
        df = load_data()
        
        if not df.empty:
            stats = get_statistics(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Tổng quan")
                st.metric("Tổng số kỳ", stats['total_draws'])
                st.metric("Tổng số digit", stats['total_digits'])
                
                if 'avg_draws_per_day' in stats:
                    st.metric("Kỳ/ngày", f"{stats['avg_draws_per_day']:.1f}")
            
            with col2:
                st.subheader("🔥 Số nóng/lạnh")
                
                st.write("**Số xuất hiện nhiều nhất:**")
                for num, count in stats['most_common'][:3]:
                    st.write(f"`{num}`: {count} lần ({stats['percentage'][num]})")
                
                st.write("**Số xuất hiện ít nhất:**")
                for num, count in stats['least_common'][:3]:
                    st.write(f"`{num}`: {count} lần ({stats['percentage'][num]})")
            
            st.divider()
            
            # Frequency chart
            st.subheader("📈 Biểu đồ tần suất")
            freq_df = pd.DataFrame.from_dict(stats['frequency'], orient='index', columns=['count'])
            freq_df = freq_df.sort_values('count', ascending=False)
            st.bar_chart(freq_df)
            
            # Recent data
            st.subheader("📋 Dữ liệu gần đây")
            st.dataframe(
                df.tail(10).sort_values('time', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Chưa có dữ liệu để hiển thị thống kê")
    
    # ============ TAB 4: AI SETTINGS ============
    with tab4:
        st.subheader("⚙️ Cấu hình AI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight_recent = st.slider(
                "Trọng số dữ liệu gần đây",
                0.1, 0.9, 0.6, 0.1,
                help="Ảnh hưởng của các kỳ gần nhất"
            )
            
            weight_frequency = st.slider(
                "Trọng số tần suất",
                0.1, 0.9, 0.3, 0.1,
                help="Ảnh hưởng của tần suất xuất hiện"
            )
            
            avoid_recent = st.slider(
                "Tránh số trùng (kỳ)",
                1, 10, 3, 1,
                help="Số kỳ gần nhất để tránh trùng số"
            )
        
        with col2:
            weight_pattern = st.slider(
                "Trọng số mẫu pattern",
                0.0, 0.5, 0.1, 0.05,
                help="Ảnh hưởng của pattern phát hiện"
            )
            
            hot_threshold = st.slider(
                "Ngưỡng số nóng (%)",
                5, 30, 15, 1,
                help="Tỉ lệ xuất hiện tối thiểu để coi là số nóng"
            ) / 100
            
            history_window = st.slider(
                "Cửa sổ phân tích",
                5, 50, 20, 5,
                help="Số kỳ dùng để phân tích xu hướng"
            )
        
        if st.button("💾 Lưu cấu hình AI", type="primary"):
            config = {
                "weight_recent": weight_recent,
                "weight_frequency": weight_frequency,
                "weight_pattern": weight_pattern,
                "avoid_recent_count": avoid_recent,
                "hot_number_threshold": hot_threshold
            }
            
            with open(AI_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("✅ Đã lưu cấu hình AI")
            st.rerun()
        
        st.divider()
        
        st.subheader("🔄 Điều khiển dữ liệu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Xóa dữ liệu cũ (trước 30 ngày)", use_container_width=True):
                df = load_data()
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    cutoff = datetime.now() - timedelta(days=30)
                    df = df[df['time'] >= cutoff]
                    df.to_csv(DATA_FILE, index=False)
                    st.success("✅ Đã xóa dữ liệu cũ")
                    st.rerun()
        
        with col2:
            if st.button("📥 Xuất dữ liệu", use_container_width=True):
                df = load_data()
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Tải file CSV",
                    data=csv,
                    file_name=f"numcore_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # ============ FOOTER ============
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📊 Dữ liệu: {len(load_data())} kỳ")
    
    with col2:
        st.caption("🤖 AI: Enhanced Pattern Recognition")
    
    with col3:
        st.caption("NUMCORE AI+ v7.0 – Ổn định cao – Dự đoán thông minh")

if __name__ == "__main__":
    main()
