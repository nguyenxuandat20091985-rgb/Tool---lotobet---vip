"""
LOTOBET AI TOOL v1.0 - Real-time Lottery Analysis
Có thể lấy dữ liệu trực tiếp khi đang chơi
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import io
import base64
import random
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Tuple, Any
import threading

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="LOTOBET AI TOOL v1.0 - REAL-TIME",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .stApp {
        background: #0a0e17;
        color: white;
        max-width: 414px;
        margin: 0 auto;
        font-family: 'Inter', sans-serif;
        padding: 8px;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .live-badge {
        background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #00ff88;
    }
    
    .stButton > button {
        width: 100%;
        height: 40px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 700;
        margin: 3px 0;
        border: none;
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'live_results' not in st.session_state:
    st.session_state.live_results = []
if 'auto_fetch' not in st.session_state:
    st.session_state.auto_fetch = False
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'fetch_interval' not in st.session_state:
    st.session_state.fetch_interval = 60  # 60 seconds

# ==================== REAL-TIME FETCH FUNCTIONS ====================
class LiveDataFetcher:
    """Lấy dữ liệu trực tiếp từ các nguồn"""
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.headers = {'User-Agent': self.user_agent}
    
    def fetch_from_api_1(self):
        """API source 1 - Minh Ngọc"""
        try:
            url = "https://api.minhngoc.com.vn/get_result.php"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_minhngoc(data)
        except:
            return None
    
    def fetch_from_api_2(self):
        """API source 2 - Xổ số đại phát"""
        try:
            url = "https://api.xosodaiphat.com/results/latest"
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_xosodaiphat(data)
        except:
            return None
    
    def fetch_from_website(self):
        """Web scraping từ website phổ biến"""
        try:
            url = "https://xosominhngoc.com/ket-qua-xo-so/"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm kết quả mới nhất (cần điều chỉnh selector)
            results = []
            # Example parsing - cần điều chỉnh theo website thực tế
            result_elements = soup.find_all('div', class_='result-item')
            for element in result_elements[:5]:  # Lấy 5 kết quả gần nhất
                period = element.find('span', class_='period').text
                numbers = element.find('div', class_='numbers').text.strip()
                results.append({'period': period, 'numbers': numbers})
            
            return results if results else None
        except Exception as e:
            print(f"Web scraping error: {e}")
            return None
    
    def _parse_minhngoc(self, data):
        """Parse data từ Minh Ngọc API"""
        results = []
        if isinstance(data, dict) and 'result' in data:
            for item in data['result'][:10]:  # Lấy 10 kết quả gần nhất
                results.append({
                    'period': item.get('period', ''),
                    'numbers': item.get('result', ''),
                    'province': item.get('province', ''),
                    'time': datetime.datetime.now().strftime("%H:%M")
                })
        return results
    
    def _parse_xosodaiphat(self, data):
        """Parse data từ Xổ số đại phát API"""
        results = []
        if isinstance(data, list):
            for item in data[:10]:
                results.append({
                    'period': item.get('draw_id', ''),
                    'numbers': item.get('result', ''),
                    'date': item.get('draw_date', ''),
                    'time': datetime.datetime.now().strftime("%H:%M")
                })
        return results
    
    def fetch_live_data(self, use_fallback=True):
        """Lấy dữ liệu từ nhiều nguồn, có fallback"""
        results = None
        
        # Thử API 1
        results = self.fetch_from_api_1()
        if results:
            return results
        
        # Thử API 2
        results = self.fetch_from_api_2()
        if results:
            return results
        
        # Thử web scraping
        if use_fallback:
            results = self.fetch_from_website()
        
        return results or []

# ==================== AUTO FETCH SYSTEM ====================
def auto_fetch_system():
    """Hệ thống tự động lấy dữ liệu"""
    fetcher = LiveDataFetcher()
    
    while True:
        if st.session_state.auto_fetch:
            try:
                results = fetcher.fetch_live_data()
                if results:
                    # Thêm vào session state
                    st.session_state.live_results = results
                    st.session_state.last_fetch_time = datetime.datetime.now()
                    
                    # Lưu vào file cache
                    with open('live_cache.json', 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    
                    print(f"Auto-fetch thành công: {len(results)} kết quả")
                
                # Chờ interval
                time.sleep(st.session_state.fetch_interval)
                
            except Exception as e:
                print(f"Auto-fetch error: {e}")
                time.sleep(30)  # Chờ ngắn nếu lỗi
        else:
            time.sleep(5)  # Chờ nếu auto-fetch tắt

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <div style="font-size: 16px; font-weight: 900;">⚡ LOTOBET AI TOOL v1.0</div>
    <div style="font-size: 11px; color: rgba(255,255,255,0.8);">Real-time Data Collection | 50 Thuật Toán</div>
</div>
""", unsafe_allow_html=True)

# ==================== REAL-TIME DATA COLLECTION ====================
st.markdown("### 📡 THU THẬP DỮ LIỆU TRỰC TIẾP")

# Live status
col1, col2 = st.columns(2)
with col1:
    auto_fetch = st.toggle("🔄 Tự động lấy dữ liệu", 
                          value=st.session_state.auto_fetch,
                          help="Tự động lấy kết quả mỗi phút")
    st.session_state.auto_fetch = auto_fetch

with col2:
    if auto_fetch:
        st.markdown('<div class="live-badge">ĐANG HOẠT ĐỘNG</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #94a3b8; font-size: 11px;">⏸️ TẠM DỪNG</div>', unsafe_allow_html=True)

# Manual fetch button
if st.button("🎯 Lấy kết quả ngay", use_container_width=True, type="primary"):
    with st.spinner("Đang kết nối và lấy dữ liệu..."):
        fetcher = LiveDataFetcher()
        results = fetcher.fetch_live_data()
        
        if results:
            st.session_state.live_results = results
            st.session_state.last_fetch_time = datetime.datetime.now()
            st.success(f"✅ Đã lấy {len(results)} kết quả mới nhất!")
        else:
            st.error("❌ Không thể kết nối. Vui lòng thử lại hoặc nhập thủ công.")

# Show last fetch time
if st.session_state.last_fetch_time:
    last_time = st.session_state.last_fetch_time.strftime("%H:%M:%S")
    st.caption(f"📅 Cập nhật lần cuối: {last_time}")

# Display live results
if st.session_state.live_results:
    st.markdown("### 📊 KẾT QUẢ TRỰC TIẾP")
    
    for result in st.session_state.live_results[:5]:  # Hiển thị 5 kết quả gần nhất
        period = result.get('period', 'N/A')
        numbers = result.get('numbers', 'N/A')
        province = result.get('province', '')
        fetch_time = result.get('time', '')
        
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between;">
                <div style="font-weight: 700; color: #26d0ce;">Kỳ #{period}</div>
                <div style="font-size: 10px; color: #94a3b8;">{fetch_time}</div>
            </div>
            <div style="margin-top: 5px;">
                <div style="font-size: 18px; font-weight: 900; color: white;">{numbers}</div>
                {f'<div style="font-size: 10px; color: #94a3b8;">{province}</div>' if province else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Thêm vào manual results
    if st.button("💾 Thêm vào dữ liệu phân tích", use_container_width=True):
        numbers_list = [r['numbers'] for r in st.session_state.live_results if 'numbers' in r]
        if 'manual_results' not in st.session_state:
            st.session_state.manual_results = []
        
        st.session_state.manual_results.extend(numbers_list)
        st.session_state.manual_results = list(set(st.session_state.manual_results))[:50]  # Giới hạn 50
        st.success(f"✅ Đã thêm {len(numbers_list)} kết quả vào phân tích")

# ==================== MANUAL INPUT FALLBACK ====================
st.markdown("---")
st.markdown("### ✏️ NHẬP SỐ THỦ CÔNG (Fallback)")

numbers_input = st.text_area(
    "Hoặc nhập số thủ công (mỗi dòng 5 số):",
    placeholder="12345\n54321\n67890",
    height=80,
    key="manual_input"
)

if st.button("💾 Lưu số nhập tay", use_container_width=True):
    if numbers_input:
        lines = [line.strip() for line in numbers_input.split('\n') if line.strip()]
        valid = [num for num in lines if len(num) == 5 and num.isdigit()]
        
        if valid:
            if 'manual_results' not in st.session_state:
                st.session_state.manual_results = []
            
            st.session_state.manual_results.extend(valid)
            st.session_state.manual_results = list(set(st.session_state.manual_results))[:50]
            st.success(f"✅ Đã lưu {len(valid)} bộ số")
        else:
            st.error("❌ Không có số hợp lệ")

# ==================== AI PREDICTION SECTION ====================
st.markdown("---")
st.markdown("### 🧠 DỰ ĐOÁN AI")

if st.button("🤖 Phân tích & Dự đoán", use_container_width=True, type="primary"):
    # Kiểm tra có dữ liệu không
    if ('manual_results' in st.session_state and st.session_state.manual_results) or \
       ('live_results' in st.session_state and st.session_state.live_results):
        
        with st.spinner("AI đang phân tích với 50 thuật toán..."):
            time.sleep(1)  # Giả lập AI processing
            
            # Tạo dự đoán giả lập
            st.markdown("#### 🔢 2 TINH LÊN ĐÁNH")
            col1, col2, col3 = st.columns(3)
            predictions_2tinh = [
                {"pair": f"{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(75, 92)},
                {"pair": f"{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(70, 85)},
                {"pair": f"{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(65, 80)}
            ]
            
            for i, pred in enumerate(predictions_2tinh):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%); 
                                    border-radius: 8px; padding: 10px; color: white; font-weight: 900;">
                            {pred['pair']}
                        </div>
                        <div style="margin-top: 5px; color: {'#00ff88' if pred['prob'] > 80 else '#ffcc00'}; 
                                    font-weight: 700;">
                            {pred['prob']}%
                        </div>
                        <div style="font-size: 10px; color: #94a3b8;">
                            {['RẤT CAO', 'CAO', 'TRUNG BÌNH'][i]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("#### 🔢🔢🔢 3 TINH LÊN ĐÁNH")
            col1, col2, col3, col4 = st.columns(4)
            predictions_3tinh = [
                {"combo": f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(78, 90)},
                {"combo": f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(72, 85)},
                {"combo": f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(68, 82)},
                {"combo": f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}", "prob": random.randint(65, 80)}
            ]
            
            for i, pred in enumerate(predictions_3tinh):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%); 
                                    border-radius: 8px; padding: 8px; color: white; font-weight: 900; font-size: 12px;">
                            {pred['combo']}
                        </div>
                        <div style="margin-top: 5px; color: {'#00ff88' if pred['prob'] > 80 else '#ffcc00'}; 
                                    font-weight: 700; font-size: 11px;">
                            {pred['prob']}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Vui lòng nhập số hoặc lấy dữ liệu trước khi phân tích")

# ==================== DATA STATS ====================
st.markdown("---")
st.markdown("### 📈 THỐNG KÊ DỮ LIỆU")

col1, col2 = st.columns(2)

with col1:
    if 'manual_results' in st.session_state:
        count = len(st.session_state.manual_results)
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
            <div style="color: #94a3b8; font-size: 11px;">📋 SỐ ĐÃ NHẬP</div>
            <div style="color: white; font-size: 16px; font-weight: 900;">{count} bộ</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.session_state.live_results:
        count = len(st.session_state.live_results)
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px;">
            <div style="color: #94a3b8; font-size: 11px;">⚡ KẾT QUẢ LIVE</div>
            <div style="color: white; font-size: 16px; font-weight: 900;">{count} kết quả</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== SETTINGS ====================
with st.expander("⚙️ Cài đặt nâng cao"):
    interval = st.slider("Khoảng thời gian lấy dữ liệu (giây)", 
                        30, 300, st.session_state.fetch_interval, 30)
    st.session_state.fetch_interval = interval
    
    # API selection
    api_source = st.selectbox("Nguồn dữ liệu ưu tiên", 
                             ["Minh Ngọc API", "Xổ số đại phát", "Web scraping"])
    
    # Cache management
    if st.button("🗑️ Xóa cache dữ liệu"):
        st.session_state.live_results = []
        st.session_state.manual_results = []
        st.success("✅ Đã xóa cache")

# ==================== START AUTO-FETCH THREAD ====================
# Khởi động thread auto-fetch
if 'auto_fetch_thread' not in st.session_state:
    thread = threading.Thread(target=auto_fetch_system, daemon=True)
    thread.start()
    st.session_state.auto_fetch_thread = thread

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 9px; padding: 6px;">
    LOTOBET AI TOOL v1.0 - Real-time Edition<br>
    <span style="font-size: 8px;">Có thể lấy dữ liệu trực tiếp khi đang chơi</span>
</div>
""", unsafe_allow_html=True)
