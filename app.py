import streamlit as st
import re
import pandas as pd
from collections import Counter

st.set_page_config(page_title="SUPER AI LOTOBET V3", layout="wide")

st.title("🚀 SUPER AI LOTOBET - HỆ THỐNG TỔNG HỢP ĐA LUỒNG")
st.markdown("---")

# --- KHU VỰC NHẬP DỮ LIỆU ---
st.subheader("📊 Tổng hợp dữ liệu đa nguồn")
input_data = st.text_area("Dán tất cả dữ liệu bạn thu thập được vào đây:", height=200, placeholder="Ví dụ: Kỳ 123: 91043, Kỳ 124: 34193...")

if st.button("⚡ PHÂN TÍCH CHUYÊN SÂU & CHỐT SỐ"):
    if input_data:
        # Lọc dữ liệu số
        digits = re.findall(r'\d', input_data)
        if len(digits) >= 10:
            kỳ_quays = ["".join(digits[i:i+5]) for i in range(0, len(digits)-4, 5)]
            
            # 1. Thống kê tần suất
            all_num_str = "".join(kỳ_quays)
            counts = Counter(all_num_str)
            
            # 2. Phân tích nhịp cầu (logic nâng cao)
            st.success(f"🤖 AI đã tổng hợp thành công {len(kỳ_quays)} chu kỳ dữ liệu.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("📉 Tần Suất Tổng")
                df_counts = pd.DataFrame(counts.items(), columns=['Số', 'Lần về']).sort_values(by='Lần về', ascending=False)
                st.table(df_counts)

            with col2:
                st.warning("🎯 Dự Đoán Vị Trí")
                # Lấy số hay về ở vị trí cuối (giải đặc biệt)
                last_digits = [k[-1] for k in kỳ_quays]
                last_counts = Counter(last_digits)
                top_last = last_counts.most_common(2)
                st.write(f"Vị trí cuối tiềm năng: **{top_last[0][0]}**")
                st.write(f"Nhịp cầu đang chạy: **{top_last[1][0]}**")

            with col3:
                st.error("💎 CHỐT SỐ TỪ AI")
                most_common = counts.most_common(3)
                s1, s2, s3 = most_common[0][0], most_common[1][0], most_common[2][0]
                
                st.metric("BẠCH THỦ", f"{s1}")
                st.metric("SONG THỦ", f"{s1}{s2} - {s2}{s1}")
                st.metric("XIÊN/HẬU NHỊ", f"{s1}{s3}")

            st.write("---")
            st.caption("Lưu ý: Độ chính xác tăng lên khi bạn dán trên 50 kỳ quay liên tiếp.")
        else:
            st.error("Dữ liệu quá ít để AI có thể phân tích đa nguồn. Vui lòng dán thêm!")
