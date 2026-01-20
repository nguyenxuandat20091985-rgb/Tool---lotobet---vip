import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
st.set_page_config(
    page_title="AI 2 TINH LOTOBET - BẢN CHUẨN v2",
    layout="wide",
    page_icon="🎯"
)

DATA_FILE = "lotobet_data.csv"
AI_CONFIG_FILE = "ai_config_lotobet.json"

# ================= ENHANCED AI FOR LOTOBET 2-TINH =================
class LotobetTwoNumberAI:
    def __init__(self):
        self.number_history = []
        self.load_config()
    
    def load_config(self):
        """Load AI configuration"""
        default_config = {
            "min_draws": 15,
            "max_confidence": 95,
            "min_confidence": 60,
            "avoid_recent_appearance": 2,
            "prefer_gap_period": 5,
            "max_hot_duration": 3,
            "cold_threshold": 7
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
    
    def save_config(self):
        """Save AI configuration"""
        try:
            with open(AI_CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def analyze_single_numbers(self, numbers_history):
        """Phân tích từng số đơn (0-9) theo đặc tả"""
        if len(numbers_history) < 5:
            return {}
        
        analysis = {}
        
        for num in range(10):
            stats = {
                'appearances': [],
                'gaps': [],
                'states': [],
                'recent_periods': defaultdict(int),
                'current_state': 'UNKNOWN'
            }
            
            # Lịch sử xuất hiện
            for i, draw in enumerate(numbers_history):
                if num in draw:
                    stats['appearances'].append(i)
            
            # Tính khoảng cách giữa các lần xuất hiện
            if len(stats['appearances']) > 1:
                for i in range(1, len(stats['appearances'])):
                    gap = stats['appearances'][i] - stats['appearances'][i-1]
                    stats['gaps'].append(gap)
            
            # Phân loại cầu
            stats['bridge_type'] = self.detect_bridge_type(stats, numbers_history)
            
            # Gán trạng thái
            stats['current_state'] = self.determine_number_state(stats, numbers_history)
            
            # Thống kê theo khoảng thời gian
            recent_history = numbers_history[-self.config.get('cold_threshold', 7):]
            for i, draw in enumerate(recent_history):
                if num in draw:
                    stats['recent_periods'][len(recent_history)-i-1] += 1
            
            analysis[num] = stats
        
        return analysis
    
    def detect_bridge_type(self, stats, numbers_history):
        """Nhận diện loại cầu theo đặc tả"""
        if len(stats['appearances']) < 3:
            return "INSUFFICIENT_DATA"
        
        recent_appearances = stats['appearances'][-3:]
        recent_gaps = stats['gaps'][-2:] if len(stats['gaps']) >= 2 else []
        
        # CẦU BỆT: Xuất hiện liên tục nhiều kỳ
        if len(recent_appearances) >= 3:
            gaps = [recent_appearances[i+1] - recent_appearances[i] for i in range(len(recent_appearances)-1)]
            if all(gap == 1 for gap in gaps):
                return "BET"
        
        # CẦU NHẢY: Xuất hiện → nghỉ → xuất hiện (nhịp đều)
        if len(recent_gaps) >= 2:
            if all(2 <= gap <= 3 for gap in recent_gaps) and len(set(recent_gaps)) == 1:
                return "JUMP"
        
        # CẦU LẶP: Vừa ra kỳ trước lại ra tiếp
        if len(stats['appearances']) >= 2:
            last_gap = stats['appearances'][-1] - stats['appearances'][-2]
            if last_gap == 1:
                return "REPEAT"
        
        # CẦU HỒI: Biến mất nhiều kỳ rồi quay lại
        if len(stats['gaps']) > 0:
            last_gap = stats['gaps'][-1]
            if last_gap >= self.config.get('cold_threshold', 7):
                return "COMEBACK"
        
        return "NORMAL"
    
    def determine_number_state(self, stats, numbers_history):
        """Gán trạng thái cho số đơn"""
        if not stats['appearances']:
            return "COLD"
        
        recent_history = numbers_history[-10:] if len(numbers_history) >= 10 else numbers_history
        recent_appearances = [i for i in stats['appearances'] if i >= len(numbers_history) - len(recent_history)]
        
        # NÓNG: Ra dày, sát nhau
        if len(recent_appearances) >= 3:
            if max(recent_appearances) - min(recent_appearances) <= 4:
                return "HOT"
        
        # NGUY HIỂM: Vừa ra hoặc ra dồn
        if len(recent_appearances) >= 2:
            gaps = [recent_appearances[i+1] - recent_appearances[i] for i in range(len(recent_appearances)-1)]
            if any(gap == 1 for gap in gaps):
                return "RISKY"
        
        # ỔN ĐỊNH: Ra đều, có nhịp
        if len(stats['gaps']) >= 3:
            gap_std = np.std(stats['gaps']) if stats['gaps'] else 0
            if gap_std <= 1.5:
                return "STABLE"
        
        # YẾU: Ít xuất hiện
        if len(recent_appearances) <= 1:
            return "WEAK"
        
        return "NORMAL"
    
    def analyze_pair_compatibility(self, num1_stats, num2_stats, numbers_history):
        """Phân tích khả năng ghép cặp của 2 số"""
        compatibility = {
            'score': 50,
            'reasons': [],
            'warnings': []
        }
        
        # ❌ CẤM TUYỆT ĐỐI số chập
        if num1_stats['number'] == num2_stats['number']:
            compatibility['score'] = 0
            compatibility['warnings'].append("SỐ CHẬP - CẤM TUYỆT ĐỐI")
            return compatibility
        
        # ❌ Không ghép 2 số đều nóng
        if num1_stats['current_state'] == "HOT" and num2_stats['current_state'] == "HOT":
            compatibility['score'] *= 0.3
            compatibility['warnings'].append("2 số đều NÓNG - nguy cơ gãy cầu")
        
        # ❌ Không ghép 2 số đều nguy hiểm
        if num1_stats['current_state'] == "RISKY" and num2_stats['current_state'] == "RISKY":
            compatibility['score'] *= 0.4
            compatibility['warnings'].append("2 số đều NGUY HIỂM")
        
        # ❌ Không ghép 2 số đều yếu
        if num1_stats['current_state'] == "WEAK" and num2_stats['current_state'] == "WEAK":
            compatibility['score'] *= 0.5
            compatibility['warnings'].append("2 số đều YẾU - thiếu dữ liệu")
        
        # ✅ Ưu tiên: 1 số ổn định + 1 số đang hồi
        if (num1_stats['current_state'] == "STABLE" and num2_stats['bridge_type'] == "COMEBACK") or \
           (num2_stats['current_state'] == "STABLE" and num1_stats['bridge_type'] == "COMEBACK"):
            compatibility['score'] *= 1.4
            compatibility['reasons'].append("1 ổn định + 1 hồi cầu - tổ hợp tốt")
        
        # ✅ Ưu tiên: 1 số nhảy nhịp tốt + 1 số ổn định
        if (num1_stats['bridge_type'] == "JUMP" and num2_stats['current_state'] == "STABLE") or \
           (num2_stats['bridge_type'] == "JUMP" and num1_stats['current_state'] == "STABLE"):
            compatibility['score'] *= 1.3
            compatibility['reasons'].append("Nhảy nhịp + Ổn định - an toàn")
        
        # ⚠️ Cảnh báo cầu bệt
        if num1_stats['bridge_type'] == "BET" or num2_stats['bridge_type'] == "BET":
            compatibility['score'] *= 0.7
            compatibility['warnings'].append("Có số đang BỆT - nguy cơ gãy")
        
        # ⚠️ Cảnh báo cầu lặp
        if num1_stats['bridge_type'] == "REPEAT" or num2_stats['bridge_type'] == "REPEAT":
            compatibility['score'] *= 0.6
            compatibility['warnings'].append("Có số LẶP - xác suất thấp")
        
        # Kiểm tra xuất hiện cùng nhau trong lịch sử
        together_count = 0
        for draw in numbers_history[-20:]:
            if num1_stats['number'] in draw and num2_stats['number'] in draw:
                together_count += 1
        
        if together_count > 0:
            compatibility['score'] *= (1 + together_count * 0.1)
            compatibility['reasons'].append(f"Đã xuất hiện cùng nhau {together_count} lần")
        
        # Đảm bảo điểm số trong khoảng 0-100
        compatibility['score'] = max(0, min(100, compatibility['score']))
        
        return compatibility
    
    def should_skip_draw(self, numbers_analysis, numbers_history):
        """Logic KHÔNG ĐÁNH theo đặc tả"""
        reasons = []
        
        # 🚫 Dữ liệu không đủ
        if len(numbers_history) < self.config['min_draws']:
            return True, [f"Dữ liệu chỉ có {len(numbers_history)} kỳ, cần ít nhất {self.config['min_draws']} kỳ"]
        
        # Đếm số trạng thái
        state_counts = Counter()
        for num, stats in numbers_analysis.items():
            state_counts[stats['current_state']] += 1
        
        # 🚫 Toàn số quá nóng
        if state_counts.get('HOT', 0) >= 7:
            reasons.append("Quá nhiều số NÓNG (>7)")
        
        # 🚫 Nhiều số vừa ra kỳ trước
        recent_numbers = set()
        if len(numbers_history) > 0:
            recent_numbers = set(numbers_history[-1])
        
        recent_repeat_count = 0
        for num in recent_numbers:
            if num in numbers_analysis:
                if numbers_analysis[num]['bridge_type'] == 'REPEAT':
                    recent_repeat_count += 1
        
        if recent_repeat_count >= 3:
            reasons.append(f"Có {recent_repeat_count} số vừa ra kỳ trước")
        
        # 🚫 Phát hiện cầu gãy
        bridge_types = [stats['bridge_type'] for stats in numbers_analysis.values()]
        if bridge_types.count('BET') >= 2:
            reasons.append("Phát hiện nhiều cầu BỆT có thể gãy")
        
        if reasons:
            return True, reasons
        
        return False, []
    
    def predict_pairs(self, numbers_history):
        """Dự đoán cặp số theo đặc tả chuẩn v2"""
        if len(numbers_history) < 5:
            return [], {}, "INSUFFICIENT_DATA", []
        
        # Phân tích từng số đơn
        numbers_analysis = self.analyze_single_numbers(numbers_history)
        
        # Kiểm tra điều kiện KHÔNG ĐÁNH
        should_skip, skip_reasons = self.should_skip_draw(numbers_analysis, numbers_history)
        if should_skip:
            return [], {}, "SKIP", skip_reasons
        
        # Tạo danh sách số có trạng thái tốt để ghép
        candidate_numbers = []
        for num, stats in numbers_analysis.items():
            stats['number'] = num
            
            # Ưu tiên số có trạng thái tốt
            if stats['current_state'] in ['STABLE', 'NORMAL']:
                priority = 3
            elif stats['current_state'] == 'WEAK':
                priority = 2
            elif stats['bridge_type'] == 'COMEBACK':
                priority = 4  # Ưu tiên cầu hồi
            elif stats['bridge_type'] == 'JUMP':
                priority = 3  # Ưu tiên cầu nhảy
            else:
                priority = 1
            
            candidate_numbers.append((priority, num, stats))
        
        # Sắp xếp theo độ ưu tiên
        candidate_numbers.sort(key=lambda x: x[0], reverse=True)
        candidate_numbers = candidate_numbers[:8]  # Giới hạn số lượng
        
        # Ghép cặp và đánh giá
        pair_predictions = []
        
        for i in range(len(candidate_numbers)):
            for j in range(i+1, len(candidate_numbers)):
                _, num1, stats1 = candidate_numbers[i]
                _, num2, stats2 = candidate_numbers[j]
                
                # ❌ Bỏ qua số chập
                if num1 == num2:
                    continue
                
                compatibility = self.analyze_pair_compatibility(stats1, stats2, numbers_history)
                
                if compatibility['score'] >= self.config['min_confidence']:
                    pair = tuple(sorted([num1, num2]))
                    pair_info = {
                        'pair': pair,
                        'score': compatibility['score'],
                        'confidence': int(compatibility['score']),
                        'num1_state': stats1['current_state'],
                        'num2_state': stats2['current_state'],
                        'num1_bridge': stats1['bridge_type'],
                        'num2_bridge': stats2['bridge_type'],
                        'reasons': compatibility['reasons'],
                        'warnings': compatibility['warnings'],
                        'details': {
                            f"num{num1}": stats1,
                            f"num{num2}": stats2
                        }
                    }
                    pair_predictions.append(pair_info)
        
        # Sắp xếp theo độ tin cậy
        pair_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Giới hạn số lượng cặp
        top_pairs = pair_predictions[:2]  # Tối đa 2 cặp theo đặc tả
        
        # Chuyển đổi định dạng đầu ra
        final_pairs = [info['pair'] for info in top_pairs]
        confidence_details = {info['pair']: info for info in top_pairs}
        
        return final_pairs, confidence_details, "PREDICT", []

# ================= DATA FUNCTIONS =================
def load_data():
    """Load historical data"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["time", "numbers", "period"])
    
    try:
        df = pd.read_csv(DATA_FILE)
        
        required_cols = ["time", "numbers", "period"]
        for col in required_cols:
            if col not in df.columns:
                if col == "numbers":
                    df["numbers"] = df.iloc[:, 0].astype(str)
                elif col == "time":
                    df["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elif col == "period":
                    df["period"] = range(len(df), 0, -1)
        
        df["numbers"] = df["numbers"].astype(str).str.strip()
        return df[["time", "numbers", "period"]]
    
    except Exception as e:
        return pd.DataFrame(columns=["time", "numbers", "period"])

def save_data(values):
    """Save data entries"""
    try:
        df = load_data()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_rows = []
        for v in values:
            v_str = str(v).strip()
            if v_str.isdigit() and len(v_str) == 5:
                period = len(df) + len(new_rows) + 1
                new_rows.append({
                    "time": now, 
                    "numbers": v_str,
                    "period": period
                })
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
        
        return len(new_rows)
    except:
        return 0

def parse_numbers(v):
    """Parse string to list of integers"""
    try:
        return [int(x) for x in str(v) if x.isdigit()][:5]
    except:
        return []

def get_statistics(df):
    """Calculate statistics"""
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
    
    stats = {
        'total_draws': len(df),
        'number_sequences': number_sequences,
        'frequency': dict(counter),
        'percentage': {k: f"{(v/total*100):.1f}%" for k, v in counter.items()},
        'most_common': counter.most_common(5),
        'least_common': counter.most_common()[:-6:-1]
    }
    
    return stats

# ================= MAIN APP =================
def main():
    st.title("🎯 AI 2 TINH LOTOBET - BẢN CHUẨN v2")
    st.caption("""
    ⚠️ TUÂN THỦ ĐẶC TẢ LOGIC: 
    • Loại bỏ số chập (11, 22, 33...) 
    • Phân tích số đơn trước khi ghép 
    • Logic KHÔNG ĐÁNH khi cần thiết
    • Tối đa 1-2 cặp
    """)
    
    ai = LotobetTwoNumberAI()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Nhập liệu",
        "🎯 Dự đoán AI",
        "📊 Phân tích số",
        "⚙️ Cấu hình"
    ])
    
    # ============ TAB 1: DATA INPUT ============
    with tab1:
        st.subheader("📥 NHẬP DỮ LIỆU LOTOBET")
        
        raw = st.text_area(
            "Nhập kết quả các kỳ (mỗi dòng 5 số)",
            height=200,
            placeholder="Ví dụ:\n12345\n67890\n54321\n...",
            help="Mỗi dòng là 1 giải 5 số của Lotobet. AI sẽ tự động xử lý."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Lưu dữ liệu", type="primary", use_container_width=True):
                if raw.strip():
                    lines = [x.strip() for x in raw.splitlines() if x.strip()]
                    saved = save_data(lines)
                    
                    if saved > 0:
                        st.success(f"✅ Đã lưu {saved} kỳ hợp lệ")
                        st.rerun()
                    else:
                        st.error("❌ Không có dữ liệu hợp lệ (cần đúng 5 số)")
                else:
                    st.warning("⚠️ Vui lòng nhập dữ liệu")
        
        with col2:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.rerun()
        
        st.divider()
        
        df = load_data()
        if not df.empty:
            st.subheader("📋 DỮ LIỆU HIỆN CÓ")
            st.dataframe(
                df.tail(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "period": "Kỳ",
                    "time": "Thời gian",
                    "numbers": "Kết quả"
                }
            )
        else:
            st.info("📭 Chưa có dữ liệu. Vui lòng nhập ít nhất 15 kỳ để phân tích.")
    
    # ============ TAB 2: AI PREDICTION ============
    with tab2:
        df = load_data()
        
        if df.empty:
            st.warning("⏳ Vui lòng nhập dữ liệu ở tab '📥 Nhập liệu'")
            return
        
        stats = get_statistics(df)
        numbers_history = stats.get('number_sequences', [])
        
        st.subheader("🎯 PHÂN TÍCH & DỰ ĐOÁN")
        
        # Hiển thị thông tin cơ bản
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng số kỳ", len(numbers_history))
        with col2:
            st.metric("Số kỳ tối thiểu", f"{ai.config['min_draws']}+")
        with col3:
            st.metric("Độ tin cậy yêu cầu", f"{ai.config['min_confidence']}%")
        
        # Phân tích và dự đoán
        if len(numbers_history) < 5:
            st.warning(f"⚠️ Cần ít nhất 5 kỳ để phân tích (hiện có: {len(numbers_history)})")
        else:
            with st.spinner("🤖 AI đang phân tích theo đặc tả chuẩn..."):
                top_pairs, confidence_details, status, skip_reasons = ai.predict_pairs(numbers_history)
            
            if status == "INSUFFICIENT_DATA":
                st.error("❌ Dữ liệu không đủ để phân tích")
            elif status == "SKIP":
                st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
                st.warning("**Lý do:**")
                for reason in skip_reasons:
                    st.write(f"• {reason}")
                st.info("""
                **Theo đặc tả logic:**
                - Dữ liệu nhiễu / ít kỳ
                - Cầu gãy
                - Toàn số quá nóng
                - Nhiều số vừa ra kỳ trước
                - Không có cặp đạt ngưỡng an toàn
                """)
            elif status == "PREDICT":
                if not top_pairs:
                    st.warning("⚠️ Không tìm thấy cặp số đạt ngưỡng an toàn")
                    st.info("Xem xét 'KHÔNG ĐÁNH' theo đặc tả logic")
                else:
                    st.success(f"✅ AI đề xuất {len(top_pairs)} cặp số")
                    
                    # Hiển thị cặp số dự đoán
                    cols = st.columns(len(top_pairs))
                    for idx, (pair, col) in enumerate(zip(top_pairs, cols)):
                        with col:
                            details = confidence_details.get(pair, {})
                            confidence = details.get('confidence', 0)
                            
                            if confidence >= 75:
                                color = "#4CAF50"
                                badge = "🔥"
                            elif confidence >= 65:
                                color = "#2196F3"
                                badge = "⭐"
                            else:
                                color = "#FF9800"
                                badge = "📊"
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; border-radius: 10px; 
                                        background: white; border: 3px solid {color}; margin: 5px;">
                                <h3 style="color: {color}; margin: 0;">{badge} Cặp {idx+1}</h3>
                                <h1 style="font-size: 2.5em; margin: 10px 0; color: #2c3e50;">
                                    {pair[0]}{pair[1]}
                                </h1>
                                <div style="font-size: 1.2em; color: {color}; font-weight: bold;">
                                    {confidence}% tin cậy
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Hiển thị phân tích chi tiết
                    st.divider()
                    st.subheader("🔍 PHÂN TÍCH CHI TIẾT")
                    
                    for pair in top_pairs:
                        details = confidence_details.get(pair, {})
                        with st.expander(f"📊 Phân tích cặp {pair[0]}{pair[1]} ({details.get('confidence', 0)}%)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Số {pair[0]}:**")
                                st.write(f"- Trạng thái: {details.get('num1_state', 'N/A')}")
                                st.write(f"- Loại cầu: {details.get('num1_bridge', 'N/A')}")
                            
                            with col2:
                                st.write(f"**Số {pair[1]}:**")
                                st.write(f"- Trạng thái: {details.get('num2_state', 'N/A')}")
                                st.write(f"- Loại cầu: {details.get('num2_bridge', 'N/A')}")
                            
                            if details.get('reasons'):
                                st.success("**Ưu điểm:**")
                                for reason in details['reasons']:
                                    st.write(f"✓ {reason}")
                            
                            if details.get('warnings'):
                                st.warning("**Cảnh báo:**")
                                for warning in details['warnings']:
                                    st.write(f"⚠ {warning}")
                            
                            st.caption(f"**Lưu ý:** {details.get('confidence', 0)}% là độ tin cậy của AI, không phải xác suất trúng giải")
    
    # ============ TAB 3: NUMBER ANALYSIS ============
    with tab3:
        df = load_data()
        
        if df.empty:
            st.info("📭 Chưa có dữ liệu để phân tích")
            return
        
        stats = get_statistics(df)
        numbers_history = stats.get('number_sequences', [])
        
        if len(numbers_history) < 5:
            st.warning("Cần ít nhất 5 kỳ để phân tích")
            return
        
        st.subheader("📊 PHÂN TÍCH SỐ ĐƠN (0-9)")
        
        # Phân tích số đơn
        analysis = ai.analyze_single_numbers(numbers_history)
        
        # Hiển thị bảng phân tích
        analysis_data = []
        for num in range(10):
            stats = analysis.get(num, {})
            analysis_data.append({
                'Số': num,
                'Trạng thái': stats.get('current_state', 'UNKNOWN'),
                'Loại cầu': stats.get('bridge_type', 'N/A'),
                'Số lần xuất hiện': len(stats.get('appearances', [])),
                'Khoảng cách TB': np.mean(stats.get('gaps', [0])) if stats.get('gaps') else 0,
                'Xuất hiện gần nhất': len(numbers_history) - stats['appearances'][-1] if stats.get('appearances') else 'Chưa'
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Tô màu theo trạng thái
        def color_state(val):
            if val == 'HOT':
                return 'background-color: #ffcccc'
            elif val == 'RISKY':
                return 'background-color: #ffebcc'
            elif val == 'STABLE':
                return 'background-color: #ccffcc'
            elif val == 'WEAK':
                return 'background-color: #cce5ff'
            elif val == 'COLD':
                return 'background-color: #e6e6e6'
            return ''
        
        styled_df = analysis_df.style.applymap(color_state, subset=['Trạng thái'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Số": st.column_config.NumberColumn("Số", width="small"),
                "Trạng thái": st.column_config.TextColumn("Trạng thái", width="medium"),
                "Loại cầu": st.column_config.TextColumn("Loại cầu", width="medium"),
                "Số lần xuất hiện": st.column_config.NumberColumn("Số lần", width="small"),
                "Khoảng cách TB": st.column_config.NumberColumn("K.cách TB", format="%.1f", width="small"),
                "Xuất hiện gần nhất": st.column_config.TextColumn("Gần nhất", width="small")
            }
        )
        
        st.divider()
        
        # Phân tích cầu
        st.subheader("📈 PHÂN TÍCH LOẠI CẦU")
        
        bridge_counts = Counter()
        state_counts = Counter()
        
        for num, stats in analysis.items():
            bridge_counts[stats.get('bridge_type', 'UNKNOWN')] += 1
            state_counts[stats.get('current_state', 'UNKNOWN')] += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Phân bố loại cầu:**")
            for bridge_type, count in bridge_counts.most_common():
                st.write(f"{bridge_type}: {count} số")
        
        with col2:
            st.write("**Phân bố trạng thái:**")
            for state, count in state_counts.most_common():
                st.write(f"{state}: {count} số")
        
        st.divider()
        
        # Hiển thị giải thích
        st.subheader("📚 GIẢI THÍCH THUẬT NGỮ")
        
        with st.expander("Xem giải thích chi tiết"):
            st.markdown("""
            ### 🎯 TRẠNG THÁI SỐ:
            - **NÓNG**: Ra dày, sát nhau (≥3 lần trong 4-5 kỳ)
            - **NGUY HIỂM**: Vừa ra hoặc ra dồn (liên tiếp)
            - **ỔN ĐỊNH**: Ra đều, có nhịp (khoảng cách ổn định)
            - **YẾU**: Ít xuất hiện (≤1 lần trong 10 kỳ gần)
            - **LẠNH**: Chưa xuất hiện hoặc rất lâu
            
            ### 🌉 LOẠI CẦU:
            - **BỆT**: Lặp đi lặp lại nhiều kỳ (nguy cơ gãy)
            - **NHẢY**: Xuất hiện → nghỉ → xuất hiện (nhịp đều 2-3)
            - **LẶP**: Vừa ra kỳ trước lại ra tiếp (xác suất thấp)
            - **HỒI**: Biến mất nhiều kỳ rồi quay lại (tiềm năng)
            - **ĐẢO**: Hoán vị số (ví dụ 12 ↔ 21)
            
            ### ⚠️ QUY TẮC GHÉP CẶP:
            - ✅ Ưu tiên: 1 ổn định + 1 hồi, 1 nhảy + 1 ổn định
            - ❌ CẤM: Số chập (11, 22, 33...)
            - ❌ Tránh: 2 số đều nóng, 2 số đều nguy hiểm, 2 số đều yếu
            """)
    
    # ============ TAB 4: CONFIGURATION ============
    with tab4:
        st.subheader("⚙️ CẤU HÌNH AI LOTOBET")
        
        st.markdown("### 🎯 THIẾT LẬP THAM SỐ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_draws = st.slider(
                "Số kỳ tối thiểu để phân tích",
                5, 30, ai.config.get('min_draws', 15), 1,
                help="Số lượng kỳ tối thiểu cần có để AI phân tích"
            )
            
            min_confidence = st.slider(
                "Độ tin cậy tối thiểu (%)",
                50, 80, ai.config.get('min_confidence', 60), 5,
                help="Cặp số phải đạt độ tin cậy này mới được đề xuất"
            )
            
            avoid_recent = st.slider(
                "Tránh số mới xuất hiện (kỳ)",
                1, 5, ai.config.get('avoid_recent_appearance', 2), 1,
                help="Số vừa xuất hiện trong N kỳ gần sẽ bị giảm trọng số"
            )
        
        with col2:
            cold_threshold = st.slider(
                "Ngưỡng số lạnh (kỳ)",
                5, 15, ai.config.get('cold_threshold', 7), 1,
                help="Số không xuất hiện từ N kỳ trở lên được coi là lạnh"
            )
            
            max_hot_duration = st.slider(
                "Thời gian nóng tối đa (kỳ)",
                2, 5, ai.config.get('max_hot_duration', 3), 1,
                help="Số nóng liên tiếp N kỳ sẽ bị coi là nguy hiểm"
            )
            
            prefer_gap = st.slider(
                "Khoảng cách ưu tiên (kỳ)",
                3, 10, ai.config.get('prefer_gap_period', 5), 1,
                help="Khoảng cách lý tưởng giữa các lần xuất hiện"
            )
        
        if st.button("💾 Lưu cấu hình", type="primary", use_container_width=True):
            ai.config['min_draws'] = min_draws
            ai.config['min_confidence'] = min_confidence
            ai.config['avoid_recent_appearance'] = avoid_recent
            ai.config['cold_threshold'] = cold_threshold
            ai.config['max_hot_duration'] = max_hot_duration
            ai.config['prefer_gap_period'] = prefer_gap
            
            ai.save_config()
            st.success("✅ Đã lưu cấu hình!")
            st.rerun()
        
        st.divider()
        
        st.markdown("### 📊 THÔNG TIN HỆ THỐNG")
        
        df = load_data()
        
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Dữ liệu hiện có", f"{len(df)} kỳ")
        
        with info_cols[1]:
            st.metric("Phiên bản AI", "Lotobet v2.0")
        
        with info_cols[2]:
            st.metric("Trạng thái", "✅ Đang hoạt động")
        
        st.caption("""
        **AI 2 TINH LOTOBET - BẢN CHUẨN v2**  
        • Loại bỏ số chập tự động  
        • Logic KHÔNG ĐÁNH khi cần  
        • Phân tích số đơn trước khi ghép  
        • Tối đa 1-2 cặp đề xuất  
        """)

# ================= RUN APP =================
if __name__ == "__main__":
    main()
