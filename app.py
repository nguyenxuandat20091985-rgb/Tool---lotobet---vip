"""
LOTOBET ELITE v4.0 - Professional Lottery Analysis System
Streamlit Specialist Edition - Optimized for Android Mobile
No Plotly/Matplotlib Dependencies
Version: 4.0.0
Author: Senior Python Developer (Streamlit Specialist)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import itertools
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="LOTOBET ELITE v4.0",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS - MOBILE OPTIMIZED ====================
st.markdown("""
<style>
    /* Import Google Fonts for Mobile Readability */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    /* Base Styling - Ultra Dark Theme */
    .stApp {
        background-color: #050505 !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(255, 49, 49, 0.05) 0%, transparent 20%),
            radial-gradient(circle at 85% 30%, rgba(0, 255, 194, 0.05) 0%, transparent 20%);
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Main Container */
    .main-container {
        padding: 12px;
        max-width: 100%;
        overflow-x: hidden;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(20, 20, 20, 0.9);
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 49, 49, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(40, 40, 40, 0.8);
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        font-size: 14px;
        color: #AAAAAA;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 49, 49, 0.15), rgba(0, 255, 194, 0.15));
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 49, 49, 0.4);
        box-shadow: 0 4px 12px rgba(255, 49, 49, 0.2);
    }
    
    /* Card Styling - Neon Glass Effect */
    .neon-card {
        background: rgba(15, 15, 20, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid;
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        position: relative;
        overflow: hidden;
    }
    
    .neon-card-red {
        border-color: rgba(255, 49, 49, 0.3);
        box-shadow: 
            inset 0 1px 0 rgba(255, 49, 49, 0.1),
            0 8px 32px rgba(255, 49, 49, 0.15);
    }
    
    .neon-card-green {
        border-color: rgba(0, 255, 194, 0.3);
        box-shadow: 
            inset 0 1px 0 rgba(0, 255, 194, 0.1),
            0 8px 32px rgba(0, 255, 194, 0.15);
    }
    
    /* Neon Text Effects */
    .neon-text-red {
        color: #FF3131;
        text-shadow: 
            0 0 10px rgba(255, 49, 49, 0.5),
            0 0 20px rgba(255, 49, 49, 0.3),
            0 0 30px rgba(255, 49, 49, 0.1);
    }
    
    .neon-text-green {
        color: #00FFC2;
        text-shadow: 
            0 0 10px rgba(0, 255, 194, 0.5),
            0 0 20px rgba(0, 255, 194, 0.3),
            0 0 30px rgba(0, 255, 194, 0.1);
    }
    
    /* Prediction Number Display */
    .prediction-number {
        font-family: 'Roboto Mono', monospace;
        font-size: 64px;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
        padding: 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF3131, #D40000) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 14px 24px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(255, 49, 49, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 49, 49, 0.4) !important;
        background: linear-gradient(135deg, #FF4D4D, #E60000) !important;
    }
    
    /* Table Styling - Mobile Optimized */
    .dataframe {
        width: 100% !important;
        background: rgba(20, 20, 25, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        margin: 10px 0 !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #FF3131, #B20000) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 14px 12px !important;
        text-align: center !important;
        border: none !important;
    }
    
    .dataframe td {
        background: rgba(30, 30, 35, 0.9) !important;
        color: #FFFFFF !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 12px 10px !important;
        text-align: center !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .dataframe tr:hover td {
        background: rgba(255, 49, 49, 0.1) !important;
    }
    
    /* Input Fields */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        background: rgba(20, 20, 25, 0.9) !important;
        color: white !important;
        border: 1px solid rgba(255, 49, 49, 0.3) !important;
        border-radius: 12px !important;
        font-family: 'Roboto Mono', monospace !important;
        padding: 12px !important;
        font-size: 15px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF3131, #00FFC2) !important;
        border-radius: 10px !important;
    }
    
    /* Status Indicators */
    .status-hot {
        color: #FF3131;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(255, 49, 49, 0.1);
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-warm {
        color: #FFA500;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(255, 165, 0, 0.1);
        border-radius: 20px;
        display: inline-block;
    }
    
    .status-cool {
        color: #00FFC2;
        font-weight: 600;
        padding: 4px 12px;
        background: rgba(0, 255, 194, 0.1);
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Mobile Responsive Adjustments */
    @media (max-width: 768px) {
        .prediction-number {
            font-size: 48px;
            padding: 12px;
        }
        
        .neon-card {
            padding: 16px;
            margin: 12px 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 13px;
        }
        
        .dataframe td, .dataframe th {
            padding: 10px 8px !important;
            font-size: 14px !important;
        }
        
        .stButton > button {
            padding: 12px 20px !important;
            font-size: 15px !important;
        }
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF3131, #00FFC2);
        border-radius: 3px;
    }
    
    /* Loading Animation */
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 3px solid rgba(255, 49, 49, 0.3);
        border-radius: 50%;
        border-top-color: #FF3131;
        animation: spin 1s ease-in-out infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
if 'lottery_data' not in st.session_state:
    st.session_state.lottery_data = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'system_settings' not in st.session_state:
    st.session_state.system_settings = {
        'recency_weight': 15,
        'min_data_points': 10,
        'auto_analyze': True,
        'theme': 'dark'
    }
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ==================== CORE ALGORITHM ENGINE ====================
class LotteryAnalyzerV4:
    """Advanced Lottery Analysis Engine with Combination Scanning"""
    
    def __init__(self, data_list, recency_weight=15):
        """
        Initialize analyzer with data and settings
        
        Args:
            data_list: List of 5-digit lottery numbers
            recency_weight: Number of recent draws to prioritize (default: 15)
        """
        self.data = data_list
        self.recency_weight = recency_weight
        self.combination_cache = {}
        
    def _parse_draw(self, draw_str):
        """Parse a single draw and return unique digits"""
        return set(str(draw_str).strip())
    
    def _get_recent_data(self):
        """Get recent data with weights"""
        if len(self.data) <= self.recency_weight:
            return self.data, list(range(len(self.data), 0, -1))
        
        recent_data = self.data[-self.recency_weight:]
        weights = list(range(self.recency_weight, 0, -1))
        return recent_data, weights
    
    def analyze_single_digits(self):
        """Analyze single digits (0-9) with recency weighting"""
        digit_stats = {str(i): {'count': 0, 'weighted_count': 0, 'recent_appearances': 0} for i in range(10)}
        
        recent_data, weights = self._get_recent_data()
        
        # Count appearances
        for idx, draw in enumerate(self.data):
            digits = self._parse_draw(draw)
            weight = 1
            
            # Apply recency weight if in recent data
            if draw in recent_data:
                recent_idx = recent_data.index(draw)
                weight = weights[recent_idx]
            
            for digit in digits:
                digit_stats[digit]['count'] += 1
                digit_stats[digit]['weighted_count'] += weight
        
        # Calculate recent appearances
        for draw in recent_data:
            digits = self._parse_draw(draw)
            for digit in digits:
                digit_stats[digit]['recent_appearances'] += 1
        
        # Calculate scores
        results = []
        max_weighted = max([stats['weighted_count'] for stats in digit_stats.values()]) or 1
        
        for digit, stats in digit_stats.items():
            # Base frequency
            frequency = (stats['count'] / len(self.data)) * 100 if self.data else 0
            
            # Weighted score (recent draws matter more)
            weighted_score = (stats['weighted_count'] / max_weighted) * 100
            
            # Recent activity
            recent_activity = (stats['recent_appearances'] / self.recency_weight) * 100
            
            # Final score (40% frequency, 40% weighted, 20% recent)
            final_score = (frequency * 0.4) + (weighted_score * 0.4) + (recent_activity * 0.2)
            
            # Determine status
            if final_score >= 70:
                status = "HOT"
                status_class = "status-hot"
            elif final_score >= 50:
                status = "WARM"
                status_class = "status-warm"
            else:
                status = "COOL"
                status_class = "status-cool"
            
            results.append({
                'Digit': digit,
                'Score': f"{final_score:.1f}%",
                'Frequency': f"{frequency:.1f}%",
                'Recent': stats['recent_appearances'],
                'Status': status,
                '_score_value': final_score,
                '_status_class': status_class
            })
        
        return sorted(results, key=lambda x: x['_score_value'], reverse=True)
    
    def analyze_xien_2(self):
        """Analyze 2-number combinations (Xien 2)"""
        if len(self.data) < 2:
            return []
        
        combo_counter = Counter()
        weighted_counter = Counter()
        
        recent_data, weights = self._get_recent_data()
        
        for idx, draw in enumerate(self.data):
            digits = list(self._parse_draw(draw))
            
            # Generate all unique 2-number combinations from the draw
            if len(digits) >= 2:
                combos = list(itertools.combinations(sorted(digits), 2))
                weight = 1
                
                # Apply recency weight
                if draw in recent_data:
                    recent_idx = recent_data.index(draw)
                    weight = weights[recent_idx]
                
                for combo in combos:
                    combo_str = f"{combo[0]}{combo[1]}"
                    combo_counter[combo_str] += 1
                    weighted_counter[combo_str] += weight
        
        # Process results
        results = []
        max_weighted = max(weighted_counter.values()) or 1
        
        for combo_str, count in combo_counter.most_common(50):  # Top 50 combinations
            weighted_count = weighted_counter[combo_str]
            
            # Calculate scores
            frequency = (count / len(self.data)) * 100
            weighted_score = (weighted_count / max_weighted) * 100
            
            # Final score
            final_score = (frequency * 0.6) + (weighted_score * 0.4)
            
            # Determine status
            if final_score >= 60:
                status = "HOT"
                status_class = "status-hot"
            elif final_score >= 40:
                status = "WARM"
                status_class = "status-warm"
            else:
                status = "COOL"
                status_class = "status-cool"
            
            results.append({
                'Combo': combo_str,
                'Score': f"{final_score:.1f}%",
                'Count': count,
                'Weight': weighted_count,
                'Status': status,
                '_score_value': final_score,
                '_status_class': status_class
            })
        
        return sorted(results, key=lambda x: x['_score_value'], reverse=True)[:20]  # Top 20
    
    def analyze_xien_3(self):
        """Analyze 3-number combinations (Xien 3)"""
        if len(self.data) < 3:
            return []
        
        combo_counter = Counter()
        weighted_counter = Counter()
        
        recent_data, weights = self._get_recent_data()
        
        for idx, draw in enumerate(self.data):
            digits = list(self._parse_draw(draw))
            
            # Generate all unique 3-number combinations from the draw
            if len(digits) >= 3:
                combos = list(itertools.combinations(sorted(digits), 3))
                weight = 1
                
                # Apply recency weight
                if draw in recent_data:
                    recent_idx = recent_data.index(draw)
                    weight = weights[recent_idx]
                
                for combo in combos:
                    combo_str = f"{combo[0]}{combo[1]}{combo[2]}"
                    combo_counter[combo_str] += 1
                    weighted_counter[combo_str] += weight
        
        # Process results
        results = []
        max_weighted = max(weighted_counter.values()) or 1
        
        for combo_str, count in combo_counter.most_common(30):  # Top 30 combinations
            weighted_count = weighted_counter[combo_str]
            
            # Calculate scores
            frequency = (count / len(self.data)) * 100
            weighted_score = (weighted_count / max_weighted) * 100
            
            # Final score
            final_score = (frequency * 0.6) + (weighted_score * 0.4)
            
            # Determine status
            if final_score >= 50:
                status = "HOT"
                status_class = "status-hot"
            elif final_score >= 30:
                status = "WARM"
                status_class = "status-warm"
            else:
                status = "COOL"
                status_class = "status-cool"
            
            results.append({
                'Combo': combo_str,
                'Score': f"{final_score:.1f}%",
                'Count': count,
                'Weight': weighted_count,
                'Status': status,
                '_score_value': final_score,
                '_status_class': status_class
            })
        
        return sorted(results, key=lambda x: x['_score_value'], reverse=True)[:15]  # Top 15
    
    def get_top_predictions(self, analysis_type='single'):
        """Get top predictions based on analysis type"""
        if analysis_type == 'single':
            results = self.analyze_single_digits()
            return results[:3]
        elif analysis_type == 'xien2':
            results = self.analyze_xien_2()
            return results[:5]
        elif analysis_type == 'xien3':
            results = self.analyze_xien_3()
            return results[:3]
        return []

# ==================== HELPER FUNCTIONS ====================
def parse_input_data(input_text):
    """Parse and validate input data"""
    if not input_text:
        return []
    
    # Find all 5-digit numbers
    matches = re.findall(r'\b\d{5}\b', input_text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)
    
    return unique_matches

def create_neon_html(content, color="red", size="medium"):
    """Create HTML with neon effect"""
    color_map = {
        "red": "#FF3131",
        "green": "#00FFC2",
        "blue": "#00A3FF",
        "yellow": "#FFD700"
    }
    
    size_map = {
        "small": "24px",
        "medium": "32px",
        "large": "48px",
        "xlarge": "64px"
    }
    
    hex_color = color_map.get(color, "#FF3131")
    font_size = size_map.get(size, "32px")
    
    return f"""
    <div style="
        color: {hex_color};
        font-size: {font_size};
        font-weight: 700;
        text-align: center;
        text-shadow: 
            0 0 10px {hex_color}80,
            0 0 20px {hex_color}40,
            0 0 30px {hex_color}20;
        padding: 10px;
        margin: 15px 0;
        font-family: 'Roboto Mono', monospace;
    ">
        {content}
    </div>
    """

def format_table_dataframe(df, highlight_col=None):
    """Format DataFrame for display with optional highlighting"""
    display_df = df.copy()
    
    if highlight_col and highlight_col in display_df.columns:
        # Remove internal columns
        internal_cols = [col for col in display_df.columns if col.startswith('_')]
        display_df = display_df.drop(columns=internal_cols)
    
    return display_df

# ==================== MAIN APPLICATION ====================
def main():
    """Main Streamlit Application with 3-Tab Architecture"""
    
    # App Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0 10px 0;'>
            <h1 style='color: #FF3131; margin-bottom: 5px;'>
                🔥 LOTOBET ELITE v4.0
            </h1>
            <p style='color: #00FFC2; font-size: 16px; margin-top: 0;'>
                Advanced Lottery Analysis System • Mobile Optimized
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 SYSTEM", "🎯 PREDICTOR", "🔢 MATRIX"])
    
    # ==================== TAB 1: SYSTEM ====================
    with tab1:
        st.markdown("""
            <div class='neon-card neon-card-red'>
                <h3 style='color: #FF3131; text-align: center; margin-bottom: 20px;'>
                    ⚙️ SYSTEM CONFIGURATION
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 DATA INPUT")
            input_data = st.text_area(
                "Enter/Paste lottery draws (5-digit numbers):",
                height=200,
                placeholder="""Example:
12345
67890
54321
09876
13579
24680
...""",
                help="Enter one 5-digit number per line or space-separated"
            )
            
            if st.button("🔄 PROCESS DATA", use_container_width=True):
                if input_data:
                    parsed_data = parse_input_data(input_data)
                    if parsed_data:
                        st.session_state.lottery_data = parsed_data
                        st.success(f"✅ Loaded {len(parsed_data)} valid draws!")
                        st.rerun()
                    else:
                        st.error("❌ No valid 5-digit numbers found!")
                else:
                    st.warning("⚠️ Please enter data first!")
        
        with col2:
            st.markdown("### ⚙️ SETTINGS")
            
            # Recency weight setting
            recency_weight = st.slider(
                "Recency Weight (draws):",
                min_value=5,
                max_value=30,
                value=st.session_state.system_settings['recency_weight'],
                help="Number of recent draws to prioritize"
            )
            
            # Auto-analysis toggle
            auto_analyze = st.checkbox(
                "Enable auto-analysis",
                value=st.session_state.system_settings['auto_analyze']
            )
            
            # Update settings
            if st.button("💾 SAVE SETTINGS", use_container_width=True):
                st.session_state.system_settings.update({
                    'recency_weight': recency_weight,
                    'auto_analyze': auto_analyze
                })
                st.success("✅ Settings saved!")
            
            st.markdown("---")
            st.markdown("### 📈 DATA STATISTICS")
            
            if st.session_state.lottery_data:
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Total Draws", len(st.session_state.lottery_data))
                with stats_col2:
                    st.metric("Latest Draw", st.session_state.lottery_data[-1] if st.session_state.lottery_data else "N/A")
                with stats_col3:
                    unique_digits = len(set(''.join(st.session_state.lottery_data)))
                    st.metric("Unique Digits", unique_digits)
                
                # Show recent draws
                st.markdown("#### 🕐 RECENT DRAWS")
                recent_df = pd.DataFrame({
                    'Draw': st.session_state.lottery_data[-10:] if len(st.session_state.lottery_data) >= 10 else st.session_state.lottery_data
                })
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
                
                # Clear data button
                if st.button("🗑️ CLEAR ALL DATA", type="secondary"):
                    st.session_state.lottery_data = []
                    st.session_state.analysis_results = {}
                    st.rerun()
            else:
                st.info("📭 No data loaded. Please enter data in the left panel.")
    
    # ==================== TAB 2: PREDICTOR ====================
    with tab2:
        if not st.session_state.lottery_data:
            st.warning("""
            ⚠️ **No data available!** 
            Please load data in the [SYSTEM] tab first.
            """)
        else:
            # Sub-tabs for different prediction types
            pred_tab1, pred_tab2, pred_tab3 = st.tabs(["🔢 SINGLE", "🎲 XIEN 2", "🎰 XIEN 3"])
            
            # Initialize analyzer
            analyzer = LotteryAnalyzerV4(
                st.session_state.lottery_data,
                recency_weight=st.session_state.system_settings['recency_weight']
            )
            
            # ========== SUB-TAB 1: SINGLE DIGITS ==========
            with pred_tab1:
                st.markdown("""
                    <div class='neon-card neon-card-green'>
                        <h3 style='color: #00FFC2; text-align: center; margin-bottom: 15px;'>
                            🔢 SINGLE DIGIT PREDICTIONS (0-9)
                        </h3>
                        <p style='color: #AAAAAA; text-align: center; font-size: 14px;'>
                            Analysis of individual digits appearing in any position
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Run analysis
                with st.spinner("Analyzing single digits..."):
                    single_results = analyzer.analyze_single_digits()
                
                # Top predictions
                st.markdown("### 🎯 TOP PREDICTIONS")
                top_single = single_results[:3]
                
                col1, col2, col3 = st.columns(3)
                colors = ["red", "green", "blue"]
                
                for idx, (result, color) in enumerate(zip(top_single, colors)):
                    with [col1, col2, col3][idx]:
                        st.markdown(f"""
                            <div style='
                                background: rgba(30, 30, 35, 0.9);
                                border: 2px solid rgba({'255, 49, 49' if color == 'red' else '0, 255, 194' if color == 'green' else '0, 163, 255'}, 0.4);
                                border-radius: 12px;
                                padding: 15px;
                                text-align: center;
                                margin: 10px 0;
                            '>
                                <div style='font-size: 42px; font-weight: 700; color: {'#FF3131' if color == 'red' else '#00FFC2' if color == 'green' else '#00A3FF'};'>
                                    {result['Digit']}
                                </div>
                                <div style='font-size: 18px; color: #FFFFFF; margin: 5px 0;'>
                                    {result['Score']}
                                </div>
                                <div style='font-size: 14px; color: #AAAAAA;'>
                                    {result['Status']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Detailed table
                st.markdown("### 📊 DETAILED ANALYSIS")
                display_df = format_table_dataframe(pd.DataFrame(single_results))
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Insights
                st.markdown("### 💡 INSIGHTS")
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    hot_digits = [r['Digit'] for r in single_results if r['Status'] == 'HOT'][:3]
                    st.markdown(f"""
                        <div class='neon-card neon-card-red'>
                            <h4 style='color: #FF3131;'>🔥 HOT DIGITS</h4>
                            <p style='font-size: 24px; font-weight: 700; text-align: center;'>
                                {' '.join(hot_digits) if hot_digits else 'None'}
                            </p>
                            <p style='color: #AAAAAA; font-size: 14px;'>
                                Most likely to appear in next draw
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with insights_col2:
                    cool_digits = [r['Digit'] for r in single_results if r['Status'] == 'COOL'][:3]
                    st.markdown(f"""
                        <div class='neon-card neon-card-green'>
                            <h4 style='color: #00FFC2;'>❄️ COOL DIGITS</h4>
                            <p style='font-size: 24px; font-weight: 700; text-align: center;'>
                                {' '.join(cool_digits) if cool_digits else 'None'}
                            </p>
                            <p style='color: #AAAAAA; font-size: 14px;'>
                                Less likely, consider for long shots
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # ========== SUB-TAB 2: XIEN 2 ==========
            with pred_tab2:
                st.markdown("""
                    <div class='neon-card neon-card-green'>
                        <h3 style='color: #00FFC2; text-align: center; margin-bottom: 15px;'>
                            🎲 XIEN 2 PREDICTIONS
                        </h3>
                        <p style='color: #AAAAAA; text-align: center; font-size: 14px;'>
                            2-number combinations within the same draw
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Run analysis
                with st.spinner("Analyzing 2-number combinations..."):
                    xien2_results = analyzer.analyze_xien_2()
                
                if xien2_results:
                    # Top combinations
                    st.markdown("### 🎯 TOP COMBINATIONS")
                    top_xien2 = xien2_results[:5]
                    
                    cols = st.columns(5)
                    for idx, result in enumerate(top_xien2):
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='
                                    background: rgba(30, 30, 35, 0.9);
                                    border: 2px solid rgba(0, 255, 194, 0.4);
                                    border-radius: 12px;
                                    padding: 12px;
                                    text-align: center;
                                    margin: 8px 0;
                                '>
                                    <div style='font-size: 24px; font-weight: 700; color: #00FFC2;'>
                                        {result['Combo']}
                                    </div>
                                    <div style='font-size: 16px; color: #FFFFFF; margin: 5px 0;'>
                                        {result['Score']}
                                    </div>
                                    <div style='font-size: 12px; color: #AAAAAA;'>
                                        Count: {result['Count']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed table
                    st.markdown("### 📊 COMBINATION ANALYSIS")
                    display_df = format_table_dataframe(pd.DataFrame(xien2_results))
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Combination matrix
                    st.markdown("### 🔢 COMBINATION MATRIX")
                    matrix_data = []
                    for result in xien2_results[:10]:
                        combo = result['Combo']
                        matrix_data.append({
                            'Combo': combo,
                            'Digit 1': combo[0],
                            'Digit 2': combo[1],
                            'Score': result['Score'],
                            'Status': result['Status']
                        })
                    
                    if matrix_data:
                        matrix_df = pd.DataFrame(matrix_data)
                        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
                else:
                    st.info("ℹ️ Not enough data for Xien 2 analysis. Need at least 2 draws.")
            
            # ========== SUB-TAB 3: XIEN 3 ==========
            with pred_tab3:
                st.markdown("""
                    <div class='neon-card neon-card-green'>
                        <h3 style='color: #00FFC2; text-align: center; margin-bottom: 15px;'>
                            🎰 XIEN 3 PREDICTIONS
                        </h3>
                        <p style='color: #AAAAAA; text-align: center; font-size: 14px;'>
                            3-number combinations within the same draw
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Run analysis
                with st.spinner("Analyzing 3-number combinations..."):
                    xien3_results = analyzer.analyze_xien_3()
                
                if xien3_results:
                    # Top combinations
                    st.markdown("### 🎯 TOP COMBINATIONS")
                    top_xien3 = xien3_results[:3]
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(top_xien3):
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='
                                    background: rgba(30, 30, 35, 0.9);
                                    border: 2px solid rgba(255, 49, 49, 0.4);
                                    border-radius: 12px;
                                    padding: 15px;
                                    text-align: center;
                                    margin: 10px 0;
                                '>
                                    <div style='font-size: 28px; font-weight: 700; color: #FF3131;'>
                                        {result['Combo']}
                                    </div>
                                    <div style='font-size: 18px; color: #FFFFFF; margin: 5px 0;'>
                                        {result['Score']}
                                    </div>
                                    <div style='font-size: 14px; color: #AAAAAA;'>
                                        Count: {result['Count']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed table
                    st.markdown("### 📊 COMBINATION ANALYSIS")
                    display_df = format_table_dataframe(pd.DataFrame(xien3_results))
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Pattern insights
                    st.markdown("### 💎 PATTERN INSIGHTS")
                    pattern_insights = []
                    for result in xien3_results[:5]:
                        combo = result['Combo']
                        digits = list(combo)
                        pattern_type = "Mixed" if len(set(digits)) == 3 else "Pair"
                        pattern_insights.append({
                            'Combo': combo,
                            'Type': pattern_type,
                            'Digits': '-'.join(digits),
                            'Score': result['Score']
                        })
                    
                    if pattern_insights:
                        pattern_df = pd.DataFrame(pattern_insights)
                        st.dataframe(pattern_df, use_container_width=True, hide_index=True)
                else:
                    st.info("ℹ️ Not enough data for Xien 3 analysis. Need at least 3 draws.")
    
    # ==================== TAB 3: MATRIX ====================
    with tab3:
        if not st.session_state.lottery_data:
            st.warning("""
            ⚠️ **No data available!** 
            Please load data in the [SYSTEM] tab first.
            """)
        else:
            st.markdown("""
                <div class='neon-card neon-card-red'>
                    <h3 style='color: #FF3131; text-align: center; margin-bottom: 20px;'>
                        🔢 DIGIT MATRIX ANALYSIS
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Frequency matrix
            st.markdown("### 📊 DIGIT FREQUENCY MATRIX")
            
            # Create frequency matrix for digits 0-9
            freq_matrix = np.zeros((10, 5), dtype=int)  # 10 digits x 5 positions
            
            for draw in st.session_state.lottery_data:
                for pos, digit in enumerate(str(draw)):
                    if digit.isdigit():
                        freq_matrix[int(digit)][pos] += 1
            
            # Convert to DataFrame
            matrix_df = pd.DataFrame(
                freq_matrix,
                index=[str(i) for i in range(10)],
                columns=[f'Pos {i+1}' for i in range(5)]
            )
            
            # Display matrix with custom styling
            st.dataframe(matrix_df.style.background_gradient(cmap='Reds'), use_container_width=True)
            
            # Position analysis
            st.markdown("### 🎯 POSITION ANALYSIS")
            pos_cols = st.columns(5)
            
            for pos in range(5):
                with pos_cols[pos]:
                    pos_data = freq_matrix[:, pos]
                    max_digit = str(np.argmax(pos_data))
                    max_count = np.max(pos_data)
                    
                    st.markdown(f"""
                        <div style='
                            background: rgba(30, 30, 35, 0.9);
                            border: 1px solid rgba(255, 49, 49, 0.3);
                            border-radius: 10px;
                            padding: 15px;
                            text-align: center;
                            margin: 5px 0;
                        '>
                            <div style='color: #00FFC2; font-size: 14px; margin-bottom: 5px;'>
                                Position {pos+1}
                            </div>
                            <div style='color: #FF3131; font-size: 24px; font-weight: 700;'>
                                {max_digit}
                            </div>
                            <div style='color: #AAAAAA; font-size: 12px;'>
                                {max_count} appearances
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Digit pair matrix
            st.markdown("### 🔗 DIGIT PAIR MATRIX")
            
            # Create digit pair frequency matrix
            pair_matrix = np.zeros((10, 10), dtype=int)
            
            for draw in st.session_state.lottery_data:
                digits = list(str(draw))
                for i in range(len(digits)):
                    for j in range(i + 1, len(digits)):
                        d1, d2 = int(digits[i]), int(digits[j])
                        pair_matrix[d1][d2] += 1
                        pair_matrix[d2][d1] += 1
            
            # Convert to DataFrame
            pair_df = pd.DataFrame(
                pair_matrix,
                index=[str(i) for i in range(10)],
                columns=[str(i) for i in range(10)]
            )
            
            # Display pair matrix
            st.dataframe(pair_df.style.background_gradient(cmap='Greens'), use_container_width=True)
            
            # Hot pairs
            st.markdown("### 🔥 HOT DIGIT PAIRS")
            
            # Find top pairs
            pair_data = []
            for i in range(10):
                for j in range(i + 1, 10):
                    count = pair_matrix[i][j]
                    if count > 0:
                        pair_data.append({
                            'Pair': f"{i}{j}",
                            'Count': count,
                            'Frequency': f"{(count / len(st.session_state.lottery_data)) * 100:.1f}%"
                        })
            
            if pair_data:
                top_pairs = sorted(pair_data, key=lambda x: x['Count'], reverse=True)[:10]
                pairs_df = pd.DataFrame(top_pairs)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
            
            # Export options
            st.markdown("### 💾 EXPORT DATA")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📄 EXPORT MATRIX CSV", use_container_width=True):
                    csv = matrix_df.to_csv()
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"lottery_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with export_col2:
                if st.button("📊 EXPORT PAIRS CSV", use_container_width=True):
                    csv = pair_df.to_csv()
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"lottery_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666666; font-size: 12px; padding: 20px;'>
            <p>LOTOBET ELITE v4.0 • Streamlit Specialist Edition • Mobile Optimized</p>
            <p>⚠️ Analytical tool only • Results are predictions, not guarantees</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
