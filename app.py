def deep_analyze():
    data = st.session_state.history

    if len(data) < 5:
        return None, None, None

    # --- PHÂN TÍCH 2 SỐ 5 TINH ĐÚNG NGHĨA ---
    pair_counter = Counter()

    for row in data:
        unique_nums = set(row)
        for a in unique_nums:
            for b in unique_nums:
                if a < b:
                    pair_counter[(a, b)] += 1

    top_pairs = pair_counter.most_common(3)

    # --- CẦU BỆT (RA LIÊN TIẾP) ---
    last_5 = data[-5:]
    streak_counter = Counter()

    for i in range(1, len(last_5)):
        common = set(last_5[i]) & set(last_5[i-1])
        for num in common:
            streak_counter[num] += 1

    bet_nums = [num for num, cnt in streak_counter.items() if cnt >= 2]

    # --- TAM THỦ ---
    all_nums = [n for row in data for n in row]
    top_3 = [n for n, _ in Counter(all_nums).most_common(3)]

    return top_pairs, bet_nums, top_3
