"""
LOTOBET AI v1.0
Tool phân tích và dự đoán số xổ số
Author: Senior Python Developer + Data Analyst
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import re
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="LOTOBET AI v1.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INITIALIZATION ====================
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['number'])
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

# ==================== UTILITY FUNCTIONS ====================
def clean_number_string(num_str):
    """Làm sạch chuỗi số, chỉ giữ lại ký tự số"""
    if pd.isna(num_str):
        return ""
    # Chuyển sang string và loại bỏ mọi ký tự không phải số
    cleaned = re.sub(r'[^\d]', '', str(num_str))
    return cleaned

def validate_lottery_number(num_str):
    """Kiểm tra số hợp lệ (đúng 5 chữ số)"""
    cleaned = clean_number_string(num_str)
    return len(cleaned) == 5

def parse_input_data(input_text):
    """Phân tích dữ liệu đầu vào từ nhiều định dạng"""
    numbers = []
    lines = input_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Tách các số trên cùng một dòng (phân cách bởi dấu cách, dấu phẩy, tab)
        line_parts = re.split(r'[\s,\t]+', line)
        
        for part in line_parts:
            part = part.strip()
            if part:
                cleaned = clean_number_string(part)
                if len(cleaned) == 5:
                    numbers.append(cleaned)
    
    return list(set(numbers))  # Loại bỏ trùng lặp

def download_sample_data():
    """Tải dữ liệu mẫu từ các nguồn"""
    sample_data = """12345
67890
54321
09876
13579
24680
11223
44556
77889
99001"""
    return sample_data

# ==================== CORE ALGORITHMS ====================
class LotteryAnalyzer:
    """Lớp chứa thuật toán phân tích số"""
    
    def __init__(self, data):
        self.data = data
        self.numbers = data['number'].tolist()
        self.all_digits = []
        self._extract_digits()
    
    def _extract_digits(self):
        """Trích xuất tất cả các chữ số từ dữ liệu"""
        for num in self.numbers:
            self.all_digits.extend(list(num))
    
    def frequency_analysis(self):
        """Phân tích tần suất xuất hiện của các số"""
        freq = Counter(self.all_digits)
        total_digits = len(self.all_digits)
        
        results = {}
        for digit in '0123456789':
            count = freq.get(digit, 0)
            results[digit] = {
                'count': count,
                'frequency': count / total_digits if total_digits > 0 else 0
            }
        return results
    
    def delay_analysis(self):
        """Phân tích độ trễ giữa các lần xuất hiện"""
        digit_positions = {digit: [] for digit in '0123456789'}
        
        for idx, num in enumerate(self.numbers):
            for pos, digit in enumerate(num):
                digit_positions[digit].append(idx)
        
        delay_results = {}
        for digit in '0123456789':
            positions = digit_positions[digit]
            if len(positions) < 2:
                delay_results[digit] = {'avg_delay': 999, 'max_delay': 999}
                continue
            
            delays = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            delay_results[digit] = {
                'avg_delay': np.mean(delays) if delays else 999,
                'max_delay': max(delays) if delays else 999
            }
        
        return delay_results
    
    def cycle_analysis(self):
        """Phân tích chu kỳ xuất hiện"""
        digit_history = {digit: [] for digit in '0123456789'}
        
        for num in self.numbers:
            for digit in '0123456789':
                digit_history[digit].append(1 if digit in num else 0)
        
        cycle_results = {}
        for digit in '0123456789':
            history = digit_history[digit]
            if sum(history) < 3:
                cycle_results[digit] = {'cycle_strength': 0}
                continue
            
            # Tìm chu kỳ ngắn hạn (2-5 kỳ)
            cycles = []
            for cycle_len in range(2, 6):
                pattern_score = 0
                for i in range(len(history) - cycle_len):
                    pattern = history[i:i+cycle_len]
                    # Kiểm tra lặp lại
                    repeat_count = 0
                    for j in range(i+cycle_len, len(history)-cycle_len, cycle_len):
                        if history[j:j+cycle_len] == pattern:
                            repeat_count += 1
                    if repeat_count > 0:
                        pattern_score += repeat_count
                
                cycles.append({'length': cycle_len, 'score': pattern_score})
            
            # Tìm chu kỳ mạnh nhất
            best_cycle = max(cycles, key=lambda x: x['score'])
            cycle_results[digit] = {
                'cycle_strength': best_cycle['score'],
                'cycle_length': best_cycle['length']
            }
        
        return cycle_results
    
    def cooccurrence_analysis(self):
        """Phân tích đồng xuất hiện giữa các số"""
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for num in self.numbers:
            digits_in_num = set(num)
            for d1 in digits_in_num:
                for d2 in digits_in_num:
                    if d1 != d2:
                        cooccurrence[d1][d2] += 1
        
        # Tính điểm đồng xuất hiện
        cooccurrence_scores = {}
        for digit in '0123456789':
            total_pairs = sum(cooccurrence[digit].values())
            unique_partners = len(cooccurrence[digit])
            score = (total_pairs * 0.7) + (unique_partners * 0.3)
            cooccurrence_scores[digit] = score
        
        return cooccurrence_scores
    
    def pattern_analysis(self):
        """Phân tích pattern ngắn hạn"""
        pattern_scores = {digit: 0 for digit in '0123456789'}
        
        if len(self.numbers) < 5:
            return pattern_scores
        
        # Phân tích trend 5 kỳ gần nhất
        recent_numbers = self.numbers[-5:]
        recent_digits = []
        for num in recent_numbers:
            recent_digits.extend(list(num))
        
        recent_counter = Counter(recent_digits)
        total_recent = len(recent_digits)
        
        for digit in '0123456789':
            if total_recent > 0:
                recent_freq = recent_counter.get(digit, 0) / total_recent
                # Trend tăng/giảm
                trend_score = recent_freq * 1.5  # Ưu tiên số xuất hiện gần đây
                pattern_scores[digit] = trend_score
        
        return pattern_scores
    
    def entropy_analysis(self):
        """Phân tích entropy - độ bất định"""
        digit_probabilities = {}
        total_digits = len(self.all_digits)
        
        for digit in '0123456789':
            count = self.all_digits.count(digit)
            prob = count / total_digits if total_digits > 0 else 0
            digit_probabilities[digit] = prob
        
        # Tính entropy cho từng số
        entropy_scores = {}
        for digit in '0123456789':
            p = digit_probabilities[digit]
            if p > 0:
                entropy = -p * np.log2(p)
            else:
                entropy = 0
            entropy_scores[digit] = entropy
        
        return entropy_scores
    
    def noise_reduction_analysis(self):
        """Phân tích loại nhiễu và bias"""
        position_counts = {digit: [0, 0, 0, 0, 0] for digit in '0123456789'}
        
        for num in self.numbers:
            for pos, digit in enumerate(num):
                position_counts[digit][pos] += 1
        
        # Phát hiện bias theo vị trí
        bias_scores = {}
        for digit in '0123456789':
            counts = position_counts[digit]
            total = sum(counts)
            if total == 0:
                bias_scores[digit] = 1.0  # Không bias
                continue
            
            # Tính độ phân tán (variance)
            mean = total / 5
            variance = sum([(c - mean) ** 2 for c in counts]) / 5
            std_dev = np.sqrt(variance)
            
            # Score càng cao càng ít bias
            bias_score = 1.0 / (1.0 + std_dev)
            bias_scores[digit] = bias_score
        
        return bias_scores
    
    def analyze_all(self):
        """Chạy tất cả thuật toán và tổng hợp kết quả"""
        if len(self.numbers) < 3:
            st.warning("Cần ít nhất 3 số để phân tích hiệu quả")
            return None
        
        # Chạy 7 thuật toán
        results = {}
        
        # 1. Tần suất xuất hiện
        freq = self.frequency_analysis()
        
        # 2. Độ trễ
        delay = self.delay_analysis()
        
        # 3. Chu kỳ
        cycle = self.cycle_analysis()
        
        # 4. Đồng xuất hiện
        cooccurrence = self.cooccurrence_analysis()
        
        # 5. Pattern ngắn hạn
        pattern = self.pattern_analysis()
        
        # 6. Entropy
        entropy = self.entropy_analysis()
        
        # 7. Loại nhiễu
        noise = self.noise_reduction_analysis()
        
        # Tính điểm tổng hợp cho từng số
        for digit in '0123456789':
            # Trọng số cho từng thuật toán
            weights = {
                'frequency': 0.25,      # Tần suất: quan trọng nhất
                'delay': 0.15,          # Độ trễ
                'cycle': 0.15,          # Chu kỳ
                'cooccurrence': 0.10,   # Đồng xuất hiện
                'pattern': 0.15,        # Pattern ngắn hạn
                'entropy': 0.10,        # Entropy
                'noise': 0.10           # Loại nhiễu
            }
            
            # Chuẩn hóa điểm
            freq_score = freq[digit]['frequency']
            
            # Độ trễ: trễ càng lâu điểm càng cao (càng có khả năng xuất hiện)
            delay_score = 1.0 / (1.0 + delay[digit]['avg_delay'] / 10)
            
            # Chu kỳ
            cycle_score = min(cycle[digit]['cycle_strength'] / 5, 1.0)
            
            # Đồng xuất hiện: chuẩn hóa
            max_cooccur = max(cooccurrence.values()) if cooccurrence.values() else 1
            cooccur_score = cooccurrence[digit] / max_cooccur if max_cooccur > 0 else 0
            
            # Pattern
            pattern_score = pattern[digit]
            
            # Entropy: chuẩn hóa
            max_entropy = max(entropy.values()) if entropy.values() else 1
            entropy_score = entropy[digit] / max_entropy if max_entropy > 0 else 0
            
            # Loại nhiễu
            noise_score = noise[digit]
            
            # Tính tổng điểm có trọng số
            total_score = (
                freq_score * weights['frequency'] +
                delay_score * weights['delay'] +
                cycle_score * weights['cycle'] +
                cooccur_score * weights['cooccurrence'] +
                pattern_score * weights['pattern'] +
                entropy_score * weights['entropy'] +
                noise_score * weights['noise']
            )
            
            # Chuyển thành phần trăm
            percentage = min(total_score * 100, 99.9)
            
            # Khuyến nghị
            recommendation = "ĐÁNH" if percentage > 50 else "KHÔNG"
            
            results[digit] = {
                'percentage': percentage,
                'recommendation': recommendation,
                'details': {
                    'frequency': freq_score,
                    'delay': delay_score,
                    'cycle': cycle_score,
                    'cooccurrence': cooccur_score,
                    'pattern': pattern_score,
                    'entropy': entropy_score,
                    'noise': noise_score
                }
            }
        
        return results

# ==================== STREAMLIT UI ====================
def main():
    """Giao diện chính của ứng dụng"""
    
    # Header
    st.title("🎯 LOTOBET AI v1.0")
    st.markdown("### Tool Phân Tích & Dự Đoán Số Xổ Số")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lottery.png", width=100)
        st.markdown("### 📊 Thống Kê")
        
        if not st.session_state.data.empty:
            st.info(f"**Số lượng dữ liệu:** {len(st.session_state.data)} số")
            st.info(f"**Số gần nhất:** {st.session_state.data.iloc[-1]['number'] if len(st.session_state.data) > 0 else 'N/A'}")
        else:
            st.warning("Chưa có dữ liệu")
        
        st.markdown("---")
        st.markdown("### ⚙️ Cài Đặt")
        auto_analyze = st.checkbox("Tự động phân tích", value=True)
        st.markdown("---")
        st.markdown("#### 📱 Hỗ Trợ Mobile")
        st.caption("Tool được tối ưu cho điện thoại")
        st.caption("Phiên bản: 1.0")
    
    # Tabs chính
    tab1, tab2, tab3 = st.tabs(["📥 Thu Thập Dữ Liệu", "⚡ Phân Tích Nhanh", "📊 Phân Tích Chi Tiết"])
    
    # ========== TAB 1: DATA COLLECTION ==========
    with tab1:
        st.header("📥 Thu Thập Dữ Liệu Nhiều Nguồn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Nhập Dữ Liệu Thủ Công")
            input_method = st.radio(
                "Chọn phương thức nhập:",
                ["Nhập nhiều số", "Nhập từng số"]
            )
            
            if input_method == "Nhập nhiều số":
                data_input = st.text_area(
                    "Nhập các số 5 chữ số (mỗi số một dòng hoặc cách nhau bởi dấu cách/phẩy):",
                    height=200,
                    value=st.session_state.raw_data,
                    placeholder="""12345
67890
54321
09876
13579"""
                )
                
                if st.button("Xử lý dữ liệu", key="process_manual"):
                    numbers = parse_input_data(data_input)
                    if numbers:
                        st.session_state.data = pd.DataFrame({'number': numbers})
                        st.session_state.raw_data = data_input
                        st.success(f"✅ Đã nhập {len(numbers)} số hợp lệ!")
                        if auto_analyze:
                            analyzer = LotteryAnalyzer(st.session_state.data)
                            st.session_state.analysis_results = analyzer.analyze_all()
                    else:
                        st.error("❌ Không tìm thấy số hợp lệ nào!")
            
            else:  # Nhập từng số
                single_number = st.text_input("Nhập số 5 chữ số:", max_chars=5)
                if st.button("Thêm số", key="add_single"):
                    if validate_lottery_number(single_number):
                        cleaned = clean_number_string(single_number)
                        new_df = pd.DataFrame({'number': [cleaned]})
                        st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
                        st.success(f"✅ Đã thêm số: {cleaned}")
                        if auto_analyze:
                            analyzer = LotteryAnalyzer(st.session_state.data)
                            st.session_state.analysis_results = analyzer.analyze_all()
                    else:
                        st.error("❌ Số không hợp lệ! Phải đúng 5 chữ số.")
        
        with col2:
            st.subheader("2. Import File")
            file_type = st.selectbox("Chọn loại file:", ["TXT", "CSV"])
            
            uploaded_file = st.file_uploader(
                f"Chọn file {file_type}",
                type=[file_type.lower()],
                key="file_upload"
            )
            
            if uploaded_file is not None:
                try:
                    if file_type == "TXT":
                        content = uploaded_file.read().decode('utf-8')
                        numbers = parse_input_data(content)
                    else:  # CSV
                        df = pd.read_csv(uploaded_file)
                        # Tìm cột chứa số
                        number_col = None
                        for col in df.columns:
                            sample = df[col].iloc[0] if len(df) > 0 else ""
                            if isinstance(sample, str) and len(clean_number_string(sample)) == 5:
                                number_col = col
                                break
                        
                        if number_col:
                            numbers = [clean_number_string(str(x)) for x in df[number_col] if validate_lottery_number(str(x))]
                        else:
                            st.error("Không tìm thấy cột chứa số 5 chữ số!")
                            numbers = []
                    
                    if numbers:
                        st.session_state.data = pd.DataFrame({'number': numbers})
                        st.success(f"✅ Đã import {len(numbers)} số từ file!")
                        if auto_analyze:
                            analyzer = LotteryAnalyzer(st.session_state.data)
                            st.session_state.analysis_results = analyzer.analyze_all()
                    else:
                        st.error("Không tìm thấy số hợp lệ trong file!")
                        
                except Exception as e:
                    st.error(f"Lỗi khi đọc file: {str(e)}")
            
            st.subheader("3. Tải Dữ Liệu Mẫu")
            if st.button("Tải dữ liệu mẫu", key="load_sample"):
                sample_data = download_sample_data()
                numbers = parse_input_data(sample_data)
                st.session_state.data = pd.DataFrame({'number': numbers})
                st.session_state.raw_data = sample_data
                st.success(f"✅ Đã tải {len(numbers)} số mẫu!")
                if auto_analyze:
                    analyzer = LotteryAnalyzer(st.session_state.data)
                    st.session_state.analysis_results = analyzer.analyze_all()
        
        # Hiển thị dữ liệu hiện tại
        st.markdown("---")
        st.subheader("📋 Dữ Liệu Hiện Tại")
        
        if not st.session_state.data.empty:
            # Hiển thị với style đơn giản
            st.dataframe(
                st.session_state.data,
                column_config={
                    "number": st.column_config.TextColumn(
                        "Số",
                        width="medium"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Nút xóa dữ liệu
            col_actions1, col_actions2, col_actions3 = st.columns(3)
            with col_actions1:
                if st.button("Xóa tất cả dữ liệu", type="secondary"):
                    st.session_state.data = pd.DataFrame(columns=['number'])
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col_actions2:
                # Export TXT
                if st.button("Export TXT"):
                    txt_data = "\n".join(st.session_state.data['number'].tolist())
                    st.download_button(
                        label="Tải xuống TXT",
                        data=txt_data,
                        file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col_actions3:
                # Export CSV
                if st.button("Export CSV"):
                    csv_data = st.session_state.data.to_csv(index=False)
                    st.download_button(
                        label="Tải xuống CSV",
                        data=csv_data,
                        file_name=f"lotobet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("📭 Chưa có dữ liệu. Vui lòng nhập dữ liệu ở trên.")
    
    # ========== TAB 2: QUICK ANALYSIS ==========
    with tab2:
        st.header("⚡ Phân Tích Nhanh")
        
        if st.session_state.data.empty:
            st.warning("⏳ Vui lòng nhập dữ liệu ở Tab 1 trước!")
        else:
            if st.button("🚀 Chạy Phân Tích Nhanh", type="primary") or st.session_state.analysis_results:
                if not st.session_state.analysis_results:
                    with st.spinner("Đang phân tích dữ liệu..."):
                        analyzer = LotteryAnalyzer(st.session_state.data)
                        st.session_state.analysis_results = analyzer.analyze_all()
                
                if st.session_state.analysis_results:
                    # Tìm số có xác suất cao nhất
                    best_digit = max(
                        st.session_state.analysis_results.items(),
                        key=lambda x: x[1]['percentage']
                    )
                    
                    # Hiển thị kết quả nổi bật
                    st.markdown("---")
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.markdown("### 🎯 SỐ MẠNH NHẤT")
                        st.markdown(f"# **{best_digit[0]}**")
                        st.markdown(f"### {best_digit[1]['percentage']:.1f}%")
                        
                        # Hiển thị thanh tiến trình
                        progress_value = best_digit[1]['percentage'] / 100
                        st.progress(progress_value)
                        
                        st.markdown(f"**Khuyến nghị:** {best_digit[1]['recommendation']}")
                    
                    with col_result2:
                        st.markdown("### 📈 Chi Tiết Xác Suất")
                        # Hiển thị top 3 số
                        sorted_results = sorted(
                            st.session_state.analysis_results.items(),
                            key=lambda x: x[1]['percentage'],
                            reverse=True
                        )[:3]
                        
                        for digit, info in sorted_results:
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"### **{digit}**")
                            with col_b:
                                st.markdown(f"**{info['percentage']:.1f}%**")
                                st.progress(info['percentage'] / 100)
                    
                    # Giải thích kết quả
                    st.markdown("---")
                    st.markdown("#### 📝 Giải Thích")
                    st.info(f"Số **{best_digit[0]}** có xác suất xuất hiện cao nhất ({best_digit[1]['percentage']:.1f}%) "
                           f"trong giải đặc biệt 5 số kỳ tới. "
                           f"Khuyến nghị: **{best_digit[1]['recommendation']}**")
                    
                    # Lưu ý
                    st.markdown("---")
                    st.caption("⚠️ **Lưu ý quan trọng:** Đây là dự đoán dựa trên phân tích dữ liệu lịch sử. "
                              "Kết quả không đảm bảo 100% chính xác.")
                else:
                    st.error("Không thể phân tích dữ liệu. Vui lòng kiểm tra lại dữ liệu đầu vào.")
    
    # ========== TAB 3: DETAILED ANALYSIS ==========
    with tab3:
        st.header("📊 Phân Tích Chi Tiết")
        
        if st.session_state.data.empty:
            st.warning("⏳ Vui lòng nhập dữ liệu ở Tab 1 trước!")
        else:
            if not st.session_state.analysis_results:
                if st.button("📊 Chạy Phân Tích Chi Tiết", type="primary"):
                    with st.spinner("Đang phân tích chi tiết..."):
                        analyzer = LotteryAnalyzer(st.session_state.data)
                        st.session_state.analysis_results = analyzer.analyze_all()
                    st.rerun()
            
            if st.session_state.analysis_results:
                # Tạo DataFrame cho bảng kết quả
                analysis_df = pd.DataFrame([
                    {
                        'SỐ': digit,
                        '% XUẤT HIỆN': f"{info['percentage']:.1f}%",
                        'KHUYẾN NGHỊ': info['recommendation'],
                        'ĐIỂM CHI TIẾT': info['details']
                    }
                    for digit, info in st.session_state.analysis_results.items()
                ])
                
                # Sắp xếp theo phần trăm giảm dần
                analysis_df['SORT_KEY'] = analysis_df['% XUẤT HIỆN'].str.replace('%', '').astype(float)
                analysis_df = analysis_df.sort_values('SORT_KEY', ascending=False).drop('SORT_KEY', axis=1)
                
                # Hiển thị bảng
                st.dataframe(
                    analysis_df[['SỐ', '% XUẤT HIỆN', 'KHUYẾN NGHỊ']],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "SỐ": st.column_config.TextColumn(width="small"),
                        "% XUẤT HIỆN": st.column_config.ProgressColumn(
                            "% XUẤT HIỆN",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                            width="medium"
                        ),
                        "KHUYẾN NGHỊ": st.column_config.TextColumn(width="small")
                    }
                )
                
                # Visualizations
                st.markdown("---")
                st.subheader("📈 Biểu Đồ Phân Tích")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Bar chart
                    chart_data = pd.DataFrame([
                        {'Số': digit, 'Xác suất': info['percentage']}
                        for digit, info in st.session_state.analysis_results.items()
                    ])
                    chart_data = chart_data.sort_values('Xác suất', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(chart_data['Số'], chart_data['Xác suất'], 
                                 color=['#4CAF50' if x > 50 else '#F44336' for x in chart_data['Xác suất']])
                    ax.set_xlabel('Số')
                    ax.set_ylabel('Xác suất (%)')
                    ax.set_title('Xác suất xuất hiện của các số')
                    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Ngưỡng 50%')
                    ax.legend()
                    
                    # Thêm giá trị trên các cột
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                    
                    st.pyplot(fig)
                
                with col_viz2:
                    # Heatmap details
                    details_df = pd.DataFrame([
                        {**info['details'], 'Số': digit}
                        for digit, info in st.session_state.analysis_results.items()
                    ]).set_index('Số')
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.heatmap(details_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2)
                    ax2.set_title('Chi tiết điểm số từng thuật toán')
                    st.pyplot(fig2)
                
                # Tải xuống kết quả
                st.markdown("---")
                st.subheader("📥 Tải Xuống Kết Quả")
                
                # Chuẩn bị dữ liệu để export
                export_df = pd.DataFrame([
                    {
                        'Số': digit,
                        'Xác_suất_%': info['percentage'],
                        'Khuyến_nghị': info['recommendation'],
                        'Điểm_tần_suất': info['details']['frequency'],
                        'Điểm_độ_trễ': info['details']['delay'],
                        'Điểm_chu_kỳ': info['details']['cycle'],
                        'Điểm_đồng_xuất_hiện': info['details']['cooccurrence'],
                        'Điểm_pattern': info['details']['pattern'],
                        'Điểm_entropy': info['details']['entropy'],
                        'Điểm_loại_nhiễu': info['details']['noise']
                    }
                    for digit, info in st.session_state.analysis_results.items()
                ])
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # Export CSV
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Tải kết quả CSV",
                        data=csv_export,
                        file_name=f"lotobet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                
                with col_export2:
                    # Export JSON
                    json_export = export_df.to_json(orient='records', force_ascii=False)
                    st.download_button(
                        label="📄 Tải kết quả JSON",
                        data=json_export,
                        file_name=f"lotobet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_json"
                    )
                
                # Thống kê hệ thống
                st.markdown("---")
                with st.expander("📊 Thống Kê Hệ Thống"):
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Tổng số dữ liệu", len(st.session_state.data))
                    
                    with col_stats2:
                        avg_prob = np.mean([info['percentage'] for info in st.session_state.analysis_results.values()])
                        st.metric("Xác suất trung bình", f"{avg_prob:.1f}%")
                    
                    with col_stats3:
                        recommend_count = sum(1 for info in st.session_state.analysis_results.values() 
                                            if info['recommendation'] == 'ĐÁNH')
                        st.metric("Số được khuyến nghị", recommend_count)
            
            else:
                st.info("👈 Nhấn nút 'Chạy Phân Tích Chi Tiết' để xem kết quả")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>LOTOBET AI v1.0 | Sử dụng phân tích đa thuật toán | Phiên bản dành cho mobile</p>
        <p>⚠️ Tool hỗ trợ phân tích, không đảm bảo kết quả 100% chính xác</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    main()
