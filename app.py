<!-- Tạo file mới: index.html -->
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LOTTOBET AI TOOL</title>
    <!-- Đơn giản hóa - chỉ dùng Tailwind -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            margin: 0;
            font-family: system-ui, sans-serif;
        }
        .glass {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body>
    <!-- TOÀN BỘ CODE HTML/JS Ở ĐÂY -->
    <div class="container mx-auto px-4 py-8">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-bold text-white mb-2">🎯 LOTTOBET AI TOOL v1.0</h1>
            <p class="text-gray-300">50 Thuật toán AI - Dự đoán chính xác - Quản lý vốn thông minh</p>
        </div>

        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="glass rounded-2xl p-6">
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
                        <span class="text-2xl">🧠</span>
                    </div>
                    <div>
                        <p class="text-gray-400 text-sm">AI Status</p>
                        <p class="text-white font-bold text-xl">Đang hoạt động</p>
                    </div>
                </div>
            </div>

            <div class="glass rounded-2xl p-6">
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-full bg-purple-500/20 flex items-center justify-center">
                        <span class="text-2xl">⏱️</span>
                    </div>
                    <div>
                        <p class="text-gray-400 text-sm">Kỳ hiện tại</p>
                        <p class="text-white font-bold text-xl">#123456</p>
                    </div>
                </div>
            </div>

            <div class="glass rounded-2xl p-6">
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                        <span class="text-2xl">📈</span>
                    </div>
                    <div>
                        <p class="text-gray-400 text-sm">Win Rate</p>
                        <p class="text-white font-bold text-xl">68.4%</p>
                    </div>
                </div>
            </div>

            <div class="glass rounded-2xl p-6">
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center">
                        <span class="text-2xl">💰</span>
                    </div>
                    <div>
                        <p class="text-gray-400 text-sm">Số dư</p>
                        <p class="text-white font-bold text-xl">8,450,000đ</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Real Time Monitor -->
        <div class="glass rounded-2xl p-6 mb-8">
            <h2 class="text-2xl font-bold text-white mb-6">⏰ Theo dõi thời gian thực</h2>
            <div class="text-center">
                <div class="text-6xl font-bold text-white font-mono mb-4" id="countdown">02:30</div>
                <div class="text-gray-300">KU/Lotobet • Kỳ #123456 • Lotto A</div>
                <div class="mt-4">
                    <span class="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 text-green-400 rounded-full">
                        <span class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                        Đang mở cược
                    </span>
                </div>
            </div>
        </div>

        <!-- Analysis Tabs -->
        <div class="mb-8">
            <div class="flex gap-2 mb-4">
                <button class="px-6 py-3 bg-blue-600 text-white rounded-lg font-bold">5 TINH</button>
                <button class="px-6 py-3 bg-gray-800 text-gray-300 rounded-lg font-bold hover:bg-gray-700">2 TINH</button>
                <button class="px-6 py-3 bg-gray-800 text-gray-300 rounded-lg font-bold hover:bg-gray-700">3 TINH</button>
                <button class="px-6 py-3 bg-gray-800 text-gray-300 rounded-lg font-bold hover:bg-gray-700">TÀI/XỈU</button>
            </div>

            <div class="glass rounded-2xl p-6">
                <h3 class="text-xl font-bold text-white mb-4">🔮 Phân tích 5 Tinh</h3>
                
                <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <!-- Chục ngàn -->
                    <div class="bg-gray-900/50 rounded-xl p-4">
                        <h4 class="text-white font-bold text-center mb-3">Chục ngàn</h4>
                        <div class="space-y-2">
                            {[1,2,3,4,5].map(num => (
                                <div key={num} class="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                                    <span class="text-white font-bold">{num}</span>
                                    <span class="text-green-400">{50 - num*5}%</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <!-- Ngàn -->
                    <div class="bg-gray-900/50 rounded-xl p-4">
                        <h4 class="text-white font-bold text-center mb-3">Ngàn</h4>
                        <div class="space-y-2">
                            {[6,7,8,9,0].map(num => (
                                <div key={num} class="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                                    <span class="text-white font-bold">{num}</span>
                                    <span class="text-green-400">{45 - num}%</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <!-- Trăm -->
                    <div class="bg-gray-900/50 rounded-xl p-4">
                        <h4 class="text-white font-bold text-center mb-3">Trăm</h4>
                        <div class="space-y-2">
                            {[3,5,7,9,1].map(num => (
                                <div key={num} class="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                                    <span class="text-white font-bold">{num}</span>
                                    <span class="text-green-400">{55 - num*3}%</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <!-- Chục -->
                    <div class="bg-gray-900/50 rounded-xl p-4">
                        <h4 class="text-white font-bold text-center mb-3">Chục</h4>
                        <div class="space-y-2">
                            {[2,4,6,8,0].map(num => (
                                <div key={num} class="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                                    <span class="text-white font-bold">{num}</span>
                                    <span class="text-green-400">{60 - num*4}%</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <!-- Đơn vị -->
                    <div class="bg-gray-900/50 rounded-xl p-4">
                        <h4 class="text-white font-bold text-center mb-3">Đơn vị</h4>
                        <div class="space-y-2">
                            {[5,7,9,3,1].map(num => (
                                <div key={num} class="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                                    <span class="text-white font-bold">{num}</span>
                                    <span class="text-green-400">{65 - num*2}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Number Matrix -->
        <div class="glass rounded-2xl p-6 mb-8">
            <h2 class="text-2xl font-bold text-white mb-4">🎯 Ma trận số 1-99</h2>
            <div class="grid grid-cols-10 gap-2">
                {Array.from({length: 99}, (_, i) => i + 1).map(num => (
                    <button key={num} 
                        class="w-12 h-12 rounded-lg bg-gray-800 hover:bg-gray-700 text-white font-bold transition-colors">
                        {num}
                    </button>
                ))}
            </div>
        </div>

        <!-- Capital Management -->
        <div class="glass rounded-2xl p-6">
            <h2 class="text-2xl font-bold text-white mb-4">💰 Quản lý vốn thông minh</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-gray-900/50 rounded-xl p-4">
                    <h3 class="text-white font-bold mb-3">Vốn hiện có</h3>
                    <div class="text-3xl font-bold text-green-400">10,000,000 đ</div>
                    <div class="text-gray-400 text-sm mt-2">Tổng số dư khả dụng</div>
                </div>

                <div class="bg-gray-900/50 rounded-xl p-4">
                    <h3 class="text-white font-bold mb-3">Cược tối đa</h3>
                    <div class="text-3xl font-bold text-blue-400">300,000 đ</div>
                    <div class="text-gray-400 text-sm mt-2">3% tổng vốn/kỳ</div>
                </div>

                <div class="bg-gray-900/50 rounded-xl p-4">
                    <h3 class="text-white font-bold mb-3">Stop-loss</h3>
                    <div class="text-3xl font-bold text-red-400">30%</div>
                    <div class="text-gray-400 text-sm mt-2">Dừng khi lỗ 30% vốn ngày</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Countdown Timer
        let timeLeft = 150; // 2.5 phút
        
        function updateCountdown() {
            const minutes = Math.floor(timeLeft / 60);
            const seconds = timeLeft % 60;
            document.getElementById('countdown').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            if (timeLeft > 0) {
                timeLeft--;
            } else {
                timeLeft = 150; // Reset
            }
        }
        
        setInterval(updateCountdown, 1000);
        updateCountdown();

        // Number matrix click effect
        document.querySelectorAll('.grid button').forEach(button => {
            button.addEventListener('click', function() {
                this.classList.toggle('bg-gray-800');
                this.classList.toggle('bg-yellow-600');
            });
        });
    </script>
</body>
</html>
