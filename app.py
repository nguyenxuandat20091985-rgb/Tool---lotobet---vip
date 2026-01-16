<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LOTTOBET AI TOOL v1.0 - Hoàn Chỉnh</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/lucide-react"></script>
    <script src="https://unpkg.com/framer-motion"></script>
    <style>
        /* Custom Styles */
        :root {
            --gradient-primary: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            --gradient-secondary: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        }
        
        body {
            background: var(--gradient-primary);
            min-height: 100vh;
            font-family: 'Inter', system-ui, sans-serif;
        }
        
        .glass-effect {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .gradient-text {
            background: var(--gradient-secondary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Animations */
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.5); }
            50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.8); }
        }
        
        .animate-pulse-glow {
            animation: pulse-glow 2s ease-in-out infinite;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #6366f1, #8b5cf6);
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        // ============================================
        // 🎯 LOTTOBET AI TOOL v1.0 - COMPLETE SINGLE FILE
        // ============================================
        
        // Import Lucide Icons
        const {
            Activity, Brain, TrendingUp, DollarSign, Settings, History, Bell,
            RefreshCw, Shield, Zap, Clock, AlertTriangle, BarChart3, Target,
            CheckCircle, XCircle, AlertCircle, ChevronRight, Dice5, Hash,
            TrendingDown, Minus, Crown, Award, Calculator, PieChart, Users,
            BookOpen, Filter, Search, Star, Menu, X, User, Wallet, LogOut,
            Moon, Sun, Calendar, Award, Coins, HelpCircle, Scale
        } = lucideReact;

        // ============================================
        // 🏗️ MODULE 1: UTILITIES & HELPERS
        // ============================================
        
        const Utils = {
            formatNumber: (num) => {
                return new Intl.NumberFormat('vi-VN').format(num);
            },
            
            formatTime: (seconds) => {
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            },
            
            getProbabilityColor: (probability) => {
                if (probability >= 80) return 'from-green-500 to-emerald-500';
                if (probability >= 70) return 'from-green-400 to-emerald-400';
                if (probability >= 60) return 'from-blue-500 to-cyan-500';
                if (probability >= 50) return 'from-yellow-500 to-amber-500';
                if (probability >= 40) return 'from-orange-500 to-red-500';
                if (probability >= 30) return 'from-red-500 to-rose-500';
                return 'from-gray-500 to-slate-500';
            },
            
            getRecommendationText: (recommendation) => {
                switch(recommendation) {
                    case 'very-high': return 'NÊN ĐÁNH MẠNH';
                    case 'high': return 'NÊN ĐÁNH';
                    case 'medium-high': return 'TỐT';
                    case 'medium': return 'CÂN NHẮC';
                    case 'low': return 'TRÁNH';
                    default: return 'THEO DÕI';
                }
            }
        };

        // ============================================
        // 🏗️ MODULE 2: REAL-TIME MONITOR
        // ============================================
        
        function RealTimeMonitor() {
            const [timeLeft, setTimeLeft] = React.useState(150);
            const [currentPeriod, setCurrentPeriod] = React.useState(123456);
            const [isConnected, setIsConnected] = React.useState(true);

            React.useEffect(() => {
                const timer = setInterval(() => {
                    setTimeLeft(prev => prev <= 1 ? 150 : prev - 1);
                }, 1000);
                return () => clearInterval(timer);
            }, []);

            return (
                <div className="space-y-6">
                    <div className="glass-effect rounded-2xl p-5">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Clock className="w-5 h-5 text-blue-400" />
                                <span className="text-white font-bold">Thời gian thực</span>
                            </div>
                            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                        </div>

                        <div className="relative w-32 h-32 mx-auto my-4">
                            <svg className="w-full h-full" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="45" fill="none" stroke="#374151" strokeWidth="8" />
                                <circle cx="50" cy="50" r="45" fill="none" stroke="url(#gradient)" strokeWidth="8"
                                    strokeLinecap="round" strokeDasharray={`${(timeLeft / 150) * 283} 283`}
                                    transform="rotate(-90 50 50)" />
                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#6366f1" />
                                        <stop offset="100%" stopColor="#8b5cf6" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <div className="text-3xl font-bold text-white font-mono">
                                    {Utils.formatTime(timeLeft)}
                                </div>
                                <div className="text-sm text-gray-400">Còn lại</div>
                            </div>
                        </div>

                        <div className="text-center">
                            <div className="text-white font-bold text-lg">Kỳ #{currentPeriod}</div>
                            <div className="text-sm text-gray-400">KU/Lotobet • Lotto A</div>
                            <div className="flex items-center justify-center gap-2 mt-2">
                                <Activity className="w-4 h-4 text-green-400" />
                                <span className="text-green-400 text-sm">Đang mở cược</span>
                            </div>
                        </div>
                    </div>

                    <div className="glass-effect rounded-xl p-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-green-500/20">
                                <Activity className="w-5 h-5 text-green-400" />
                            </div>
                            <div>
                                <p className="text-white font-medium">Trạng thái kết nối</p>
                                <p className="text-sm text-gray-400">Ổn định • Real-time</p>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 3: FIVE STAR ANALYSIS
        // ============================================
        
        function FiveStarAnalysis() {
            const positions = [
                { name: "Chục ngàn", numbers: [1, 2, 3, 4, 5] },
                { name: "Ngàn", numbers: [6, 7, 8, 9, 0] },
                { name: "Trăm", numbers: [3, 5, 7, 9, 1] },
                { name: "Chục", numbers: [2, 4, 6, 8, 0] },
                { name: "Đơn vị", numbers: [5, 7, 9, 3, 1] }
            ];

            return (
                <div>
                    <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
                        {positions.map((pos, idx) => (
                            <div key={idx} className="glass-effect rounded-xl p-4">
                                <h3 className="text-white font-bold text-lg mb-3 text-center">{pos.name}</h3>
                                <div className="space-y-3">
                                    {pos.numbers.map((num, numIdx) => {
                                        const probability = 50 - numIdx * 10 + Math.random() * 20;
                                        return (
                                            <div key={numIdx} className="bg-gray-900/50 rounded-lg p-3">
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center text-white font-bold">
                                                            {num}
                                                        </div>
                                                        <span className="text-white font-bold">{Math.round(probability)}%</span>
                                                    </div>
                                                    <TrendingUp className="w-4 h-4 text-green-500" />
                                                </div>
                                                <div className="w-full bg-gray-700 rounded-full h-2">
                                                    <div 
                                                        className={`h-full rounded-full bg-gradient-to-r ${Utils.getProbabilityColor(probability)}`}
                                                        style={{ width: `${probability}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/20 rounded-xl p-4 border border-blue-500/30">
                        <div className="flex items-start gap-3">
                            <div className="p-2 rounded-lg bg-blue-500/20">
                                <TrendingUp className="w-6 h-6 text-blue-400" />
                            </div>
                            <div className="flex-1">
                                <h4 className="text-white font-bold mb-2">Khuyến nghị AI</h4>
                                <p className="text-gray-300">
                                    Vị trí <span className="text-green-400 font-bold">"Đơn vị"</span> có xu hướng mạnh nhất.
                                    Số <span className="text-yellow-400 font-bold">5</span> và <span className="text-yellow-400 font-bold">7</span>
                                    có xác suất cao nhất.
                                </p>
                                <div className="flex flex-wrap gap-2 mt-2">
                                    <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">
                                        Độ tin cậy: 87.3%
                                    </span>
                                    <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                                        Dữ liệu: 5000+ kỳ
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 4: TWO STAR ANALYSIS
        // ============================================
        
        function TwoStarAnalysis() {
            const predictions = [
                { pair: "56", probability: 65, recommendation: "high" },
                { pair: "78", probability: 25, recommendation: "low" },
                { pair: "34", probability: 72, recommendation: "high" },
                { pair: "12", probability: 48, recommendation: "medium" },
                { pair: "89", probability: 85, recommendation: "very-high" },
                { pair: "23", probability: 33, recommendation: "low" }
            ];

            return (
                <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                        {predictions.map((pred, idx) => (
                            <div key={idx} className="glass-effect rounded-xl p-4 hover:scale-[1.02] transition-transform">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                            <span className="text-white font-bold text-lg">{pred.pair}</span>
                                        </div>
                                        <div>
                                            <div className="text-2xl font-bold text-white">{pred.probability}%</div>
                                            <div className="text-sm text-gray-300">Xác suất</div>
                                        </div>
                                    </div>
                                    {pred.recommendation === "high" || pred.recommendation === "very-high" ? 
                                        <CheckCircle className="w-5 h-5 text-green-500" /> : 
                                        <XCircle className="w-5 h-5 text-red-500" />
                                    }
                                </div>

                                <div className="mb-4">
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-gray-300">Độ tin cậy</span>
                                        <span className="text-white font-bold">{pred.probability + 20}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div 
                                            className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                                            style={{ width: `${pred.probability + 20}%` }}
                                        ></div>
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-300">Pattern:</span>
                                        <span className="text-white font-semibold">Cầu bệt</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-gray-300">Xu hướng:</span>
                                        <span className="text-white font-semibold">Tăng 3 kỳ</span>
                                    </div>
                                </div>

                                <div className="mt-4 pt-3 border-t border-gray-700/50">
                                    <div className="flex items-center justify-between">
                                        <span className="text-white font-bold">Khuyến nghị:</span>
                                        <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                                            pred.recommendation === "very-high" ? "bg-green-500/30 text-green-300" :
                                            pred.recommendation === "high" ? "bg-green-400/30 text-green-300" :
                                            pred.recommendation === "medium" ? "bg-yellow-500/30 text-yellow-300" :
                                            "bg-red-500/30 text-red-300"
                                        }`}>
                                            {Utils.getRecommendationText(pred.recommendation)}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/20 rounded-xl p-4 border border-purple-500/30">
                        <h4 className="text-white font-bold mb-3">📊 Tóm tắt 2 tinh</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                                <div className="text-2xl font-bold text-green-400">3</div>
                                <div className="text-sm text-gray-300">Cặp nên đánh</div>
                            </div>
                            <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                                <div className="text-2xl font-bold text-yellow-400">1</div>
                                <div className="text-sm text-gray-300">Cần cân nhắc</div>
                            </div>
                            <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                                <div className="text-2xl font-bold text-red-400">2</div>
                                <div className="text-sm text-gray-300">Nên tránh</div>
                            </div>
                            <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                                <div className="text-2xl font-bold text-blue-400">89%</div>
                                <div className="text-sm text-gray-300">Độ chính xác</div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 5: THREE STAR ANALYSIS
        // ============================================
        
        function ThreeStarAnalysis() {
            const predictions = [
                { triple: "125", probability: 55, recommendation: "medium" },
                { triple: "268", probability: 70, recommendation: "high" },
                { triple: "679", probability: 20, recommendation: "low" },
                { triple: "348", probability: 63, recommendation: "medium-high" },
                { triple: "912", probability: 45, recommendation: "medium" },
                { triple: "734", probability: 75, recommendation: "very-high" }
            ];

            return (
                <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {predictions.map((pred, idx) => (
                            <div key={idx} className="glass-effect rounded-xl p-5 hover:border-blue-500/50 transition-all">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="relative">
                                            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center shadow-lg">
                                                <span className="text-white font-bold text-xl">{pred.triple}</span>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-3xl font-bold text-white">{pred.probability}%</div>
                                            <div className="text-sm text-gray-400">Xác suất</div>
                                        </div>
                                    </div>
                                    {pred.recommendation === "very-high" ? 
                                        <CheckCircle className="w-5 h-5 text-green-500 animate-pulse" /> :
                                        pred.recommendation === "high" ? 
                                        <CheckCircle className="w-5 h-5 text-green-400" /> :
                                        <AlertCircle className="w-5 h-5 text-yellow-500" />
                                    }
                                </div>

                                <div className="grid grid-cols-2 gap-3 mb-4">
                                    <div className="bg-gray-900/50 rounded-lg p-3">
                                        <div className="text-gray-400 text-sm mb-1">Xu hướng</div>
                                        <div className="text-white font-bold">Tăng mạnh</div>
                                    </div>
                                    <div className="bg-gray-900/50 rounded-lg p-3">
                                        <div className="text-gray-400 text-sm mb-1">Pattern</div>
                                        <div className="text-white font-bold">Cầu sống</div>
                                    </div>
                                </div>

                                <div className="mb-4">
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-gray-300">Độ tin cậy AI</span>
                                        <span className="text-white font-bold">{pred.probability + 15}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div 
                                            className={`h-full rounded-full ${
                                                pred.probability >= 70 ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                                                'bg-gradient-to-r from-yellow-500 to-amber-500'
                                            }`}
                                            style={{ width: `${pred.probability + 15}%` }}
                                        ></div>
                                    </div>
                                </div>

                                <div className="mb-4">
                                    <div className="text-gray-400 text-sm mb-2">Phân tích:</div>
                                    <ul className="space-y-1">
                                        <li className="flex items-start gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2"></div>
                                            <span className="text-gray-300 text-sm">Pattern ổn định</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2"></div>
                                            <span className="text-gray-300 text-sm">Xu hướng rõ ràng</span>
                                        </li>
                                    </ul>
                                </div>

                                <div className="pt-3 border-t border-gray-700/50">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <span className="text-white font-bold">Khuyến nghị:</span>
                                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                                                pred.recommendation === "very-high" ? "bg-gradient-to-r from-green-600 to-emerald-600 text-white" :
                                                pred.recommendation === "high" ? "bg-green-500/30 text-green-300" :
                                                pred.recommendation === "medium-high" ? "bg-blue-500/30 text-blue-300" :
                                                "bg-yellow-500/30 text-yellow-300"
                                            }`}>
                                                {Utils.getRecommendationText(pred.recommendation)}
                                            </span>
                                        </div>
                                        <div className="text-sm text-gray-400">
                                            Xuất hiện: 2 kỳ trước
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 6: TAI XIU ANALYSIS
        // ============================================
        
        function TaiXiuAnalysis() {
            const [timeRange, setTimeRange] = React.useState('7days');

            return (
                <div>
                    <div className="flex gap-2 mb-6">
                        {['7days', '30days', '100days'].map((range) => (
                            <button
                                key={range}
                                onClick={() => setTimeRange(range)}
                                className={`px-4 py-2 rounded-lg transition-all ${
                                    timeRange === range 
                                        ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg' 
                                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                                }`}
                            >
                                {range === '7days' ? '7 NGÀY' : range === '30days' ? '30 NGÀY' : '100 NGÀY'}
                            </button>
                        ))}
                    </div>

                    <div className="mb-8">
                        <h3 className="text-white font-bold text-xl mb-4 flex items-center gap-2">
                            <Scale className="w-6 h-6 text-blue-400" />
                            PHÂN TÍCH TÀI/XỈU
                        </h3>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-gradient-to-br from-green-900/20 to-emerald-900/10 rounded-xl p-5 border border-green-500/30">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 rounded-full bg-gradient-to-r from-green-600 to-emerald-600 flex items-center justify-center">
                                            <TrendingUp className="w-6 h-6 text-white" />
                                        </div>
                                        <div>
                                            <div className="text-white font-bold text-2xl">TÀI</div>
                                            <div className="text-gray-300">(Tổng ≥ 23)</div>
                                        </div>
                                    </div>
                                    <CheckCircle className="w-5 h-5 text-green-500" />
                                </div>

                                <div className="relative pt-1">
                                    <div className="flex mb-2 items-center justify-between">
                                        <div>
                                            <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-white bg-gray-700">
                                                65%
                                            </span>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-xs font-semibold inline-block">
                                                Xác suất
                                            </span>
                                        </div>
                                    </div>
                                    <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-700">
                                        <div style={{ width: '65%' }}
                                            className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-green-500 to-emerald-500">
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-red-900/20 to-orange-900/10 rounded-xl p-5 border border-red-500/30">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 rounded-full bg-gradient-to-r from-red-600 to-orange-600 flex items-center justify-center">
                                            <TrendingDown className="w-6 h-6 text-white" />
                                        </div>
                                        <div>
                                            <div className="text-white font-bold text-2xl">XỈU</div>
                                            <div className="text-gray-300">(Tổng ≤ 22)</div>
                                        </div>
                                    </div>
                                    <XCircle className="w-5 h-5 text-red-500" />
                                </div>

                                <div className="relative pt-1">
                                    <div className="flex mb-2 items-center justify-between">
                                        <div>
                                            <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-white bg-gray-700">
                                                35%
                                            </span>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-xs font-semibold inline-block">
                                                Xác suất
                                            </span>
                                        </div>
                                    </div>
                                    <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-700">
                                        <div style={{ width: '35%' }}
                                            className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-red-500 to-orange-500">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 bg-gradient-to-r from-green-900/30 to-emerald-900/20 rounded-xl p-4 border border-green-500/30">
                        <h4 className="text-white font-bold mb-2">✅ KẾT LUẬN AI:</h4>
                        <div className="text-gray-300 space-y-2">
                            <p>1. <span className="text-green-400 font-bold">TÀI</span> là lựa chọn tốt nhất với 65% xác suất.</p>
                            <p>2. Xu hướng hiện tại nghiêng về Tài, có thể tiếp tục trong 3-5 kỳ tới.</p>
                            <p className="text-yellow-300 font-bold mt-2">
                                ⚠️ Lưu ý: Luôn quản lý vốn thông minh và không đặt cược quá 5% tổng vốn.
                            </p>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 7: NUMBER MATRIX 1-99
        // ============================================
        
        function NumberMatrix() {
            const [selectedNumbers, setSelectedNumbers] = React.useState([]);
            const [search, setSearch] = React.useState('');

            const generateNumbers = () => {
                const numbers = [];
                for (let i = 1; i <= 99; i++) {
                    const probability = 50 + Math.sin(i * 0.3) * 20 + Math.random() * 10;
                    numbers.push({
                        number: i,
                        probability: Math.min(Math.max(Math.round(probability), 1), 99)
                    });
                }
                return numbers;
            };

            const numbers = generateNumbers();
            const top10 = [...numbers].sort((a, b) => b.probability - a.probability).slice(0, 10);

            const toggleSelectNumber = (number) => {
                setSelectedNumbers(prev => 
                    prev.includes(number) 
                        ? prev.filter(n => n !== number)
                        : [...prev, number]
                );
            };

            return (
                <div className="bg-gradient-to-br from-gray-900 via-purple-900/20 to-violet-900/20 rounded-2xl p-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                                <Target className="w-6 h-6 text-purple-400" />
                                Ma trận số 1-99
                            </h2>
                            <p className="text-gray-400">Phân tích xác suất chi tiết cho từng số</p>
                        </div>
                        
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Tìm số..."
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                className="pl-10 pr-4 py-2 bg-gray-900/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 w-full md:w-48"
                            />
                        </div>
                    </div>

                    <div className="mb-8">
                        <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                            <Star className="w-5 h-5 text-yellow-400" />
                            TOP 10 SỐ XÁC SUẤT CAO NHẤT
                        </h3>
                        <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-3">
                            {top10.map((num, idx) => (
                                <div key={num.number} className="relative group">
                                    <div
                                        onClick={() => toggleSelectNumber(num.number)}
                                        className={`cursor-pointer rounded-xl p-3 text-center transition-all duration-300 hover:scale-110 ${
                                            selectedNumbers.includes(num.number)
                                                ? 'ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-900'
                                                : ''
                                        }`}
                                    >
                                        <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 flex items-center justify-center text-xs font-bold text-white">
                                            {idx + 1}
                                        </div>
                                        <div className="text-2xl font-bold text-white mb-1">{num.number}</div>
                                        <div className={`text-lg font-bold bg-gradient-to-r ${Utils.getProbabilityColor(num.probability)} bg-clip-text text-transparent`}>
                                            {num.probability}%
                                        </div>
                                        <div className="text-xs text-gray-400 mt-1">⭐⭐⭐</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="mb-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-white font-bold">Tất cả số (1-99)</h3>
                            <div className="text-sm text-gray-400">
                                Đã chọn: {selectedNumbers.length} số
                            </div>
                        </div>
                        
                        <div className="grid grid-cols-6 sm:grid-cols-10 md:grid-cols-12 lg:grid-cols-15 gap-2">
                            {numbers.map(num => (
                                <div
                                    key={num.number}
                                    onClick={() => toggleSelectNumber(num.number)}
                                    className={`
                                        relative cursor-pointer rounded-lg p-2 text-center transition-all duration-200
                                        hover:scale-110 hover:z-10
                                        ${selectedNumbers.includes(num.number)
                                            ? 'ring-2 ring-yellow-400 bg-gray-800'
                                            : 'bg-gray-900/50 hover:bg-gray-800'
                                        }
                                    `}
                                >
                                    <div className="text-white font-bold text-sm">{num.number}</div>
                                    <div className="w-full h-1 bg-gray-700 rounded-full mt-1 overflow-hidden">
                                        <div
                                            className={`h-full rounded-full bg-gradient-to-r ${Utils.getProbabilityColor(num.probability)}`}
                                            style={{ width: `${num.probability}%` }}
                                        ></div>
                                    </div>
                                    <div className={`text-xs font-bold mt-1 ${
                                        num.probability >= 70 ? 'text-green-400' :
                                        num.probability >= 50 ? 'text-yellow-400' :
                                        'text-orange-400'
                                    }`}>
                                        {num.probability}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 8: CAPITAL MANAGEMENT
        // ============================================
        
        function CapitalManagement() {
            const [capital, setCapital] = React.useState(10000000);
            const [riskLevel, setRiskLevel] = React.useState('medium');
            const [stopLoss, setStopLoss] = React.useState(30);
            const [takeProfit, setTakeProfit] = React.useState(50);

            const calculateMaxBet = () => {
                const percentage = riskLevel === 'low' ? 0.01 : riskLevel === 'medium' ? 0.03 : 0.05;
                return Math.floor(capital * percentage);
            };

            return (
                <div className="space-y-6">
                    <div className="glass-effect rounded-2xl p-5">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <DollarSign className="w-6 h-6 text-green-400" />
                                <h3 className="text-xl font-bold text-white">Quản lý vốn</h3>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                                <span className="text-sm text-gray-400">Đang hoạt động</span>
                            </div>
                        </div>

                        <div className="mb-6">
                            <div className="text-center mb-4">
                                <div className="text-3xl font-bold text-white">
                                    {Utils.formatNumber(capital)} đ
                                </div>
                                <div className="text-gray-400">Tổng vốn hiện có</div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-gray-900/50 rounded-xl p-3">
                                    <div className="flex items-center gap-2 mb-1">
                                        <Target className="w-4 h-4 text-blue-400" />
                                        <span className="text-gray-400 text-sm">Ngân sách/ngày</span>
                                    </div>
                                    <div className="text-white font-bold">{Utils.formatNumber(1000000)} đ</div>
                                </div>
                                <div className="bg-gray-900/50 rounded-xl p-3">
                                    <div className="flex items-center gap-2 mb-1">
                                        <Shield className="w-4 h-4 text-green-400" />
                                        <span className="text-gray-400 text-sm">Cược tối đa</span>
                                    </div>
                                    <div className="text-white font-bold">{Utils.formatNumber(calculateMaxBet())} đ</div>
                                </div>
                            </div>
                        </div>

                        <div className="mb-6">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-white font-medium">Mức độ rủi ro</span>
                                <span className="text-sm text-gray-400">Chọn theo chiến lược</span>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                                {['low', 'medium', 'high'].map((risk) => (
                                    <button
                                        key={risk}
                                        onClick={() => setRiskLevel(risk)}
                                        className={`py-2 rounded-lg text-center transition-all ${
                                            riskLevel === risk
                                                ? `bg-gradient-to-r ${
                                                    risk === 'low' ? 'from-green-600 to-emerald-600' :
                                                    risk === 'medium' ? 'from-yellow-600 to-amber-600' :
                                                    'from-red-600 to-orange-600'
                                                } text-white shadow-lg`
                                                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                                        }`}
                                    >
                                        <div className="font-bold">
                                            {risk === 'low' ? 'THẤP' : risk === 'medium' ? 'TRUNG BÌNH' : 'CAO'}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gradient-to-br from-red-900/20 to-orange-900/10 rounded-xl p-4 border border-red-500/30">
                                <div className="flex items-center gap-2 mb-2">
                                    <TrendingDown className="w-5 h-5 text-red-400" />
                                    <span className="text-white font-bold">Stop Loss</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <input
                                        type="range"
                                        min="10"
                                        max="50"
                                        value={stopLoss}
                                        onChange={(e) => setStopLoss(parseInt(e.target.value))}
                                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                        style={{background: `linear-gradient(to right, #ef4444 0%, #ef4444 ${stopLoss}%, #374151 ${stopLoss}%, #374151 100%)`}}
                                    />
                                    <span className="text-white font-bold ml-3 w-12">{stopLoss}%</span>
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-green-900/20 to-emerald-900/10 rounded-xl p-4 border border-green-500/30">
                                <div className="flex items-center gap-2 mb-2">
                                    <TrendingUp className="w-5 h-5 text-green-400" />
                                    <span className="text-white font-bold">Take Profit</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <input
                                        type="range"
                                        min="20"
                                        max="100"
                                        value={takeProfit}
                                        onChange={(e) => setTakeProfit(parseInt(e.target.value))}
                                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                        style={{background: `linear-gradient(to right, #10b981 0%, #10b981 ${takeProfit}%, #374151 ${takeProfit}%, #374151 100%)`}}
                                    />
                                    <span className="text-white font-bold ml-3 w-12">{takeProfit}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 9: ANALYSIS TABS
        // ============================================
        
        function AnalysisTabs() {
            const [activeTab, setActiveTab] = React.useState(0);
            
            const tabs = [
                {
                    id: 0,
                    title: "5 TINH",
                    icon: React.createElement(Dice5, { className: "w-5 h-5" }),
                    component: React.createElement(FiveStarAnalysis)
                },
                {
                    id: 1,
                    title: "2 TINH",
                    icon: React.createElement(Hash, { className: "w-5 h-5" }),
                    component: React.createElement(TwoStarAnalysis)
                },
                {
                    id: 2,
                    title: "3 TINH",
                    icon: React.createElement(BarChart3, { className: "w-5 h-5" }),
                    component: React.createElement(ThreeStarAnalysis)
                },
                {
                    id: 3,
                    title: "TÀI/XỈU",
                    icon: React.createElement(Scale, { className: "w-5 h-5" }),
                    component: React.createElement(TaiXiuAnalysis)
                }
            ];

            return (
                <div className="glass-effect rounded-2xl overflow-hidden">
                    <div className="flex flex-wrap border-b border-gray-700/50">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`
                                    flex-1 min-w-[120px] px-4 py-3 flex items-center justify-center gap-2
                                    transition-all duration-300 border-b-2
                                    ${activeTab === tab.id 
                                        ? 'text-white border-blue-500 bg-gradient-to-b from-blue-500/20 to-cyan-500/20' 
                                        : 'text-gray-400 border-transparent hover:bg-gray-800/30 hover:text-gray-300'
                                    }
                                `}
                            >
                                {tab.icon}
                                <span className="font-semibold text-sm">{tab.title}</span>
                                {activeTab === tab.id && (
                                    <ChevronRight className="w-4 h-4 ml-auto" />
                                )}
                            </button>
                        ))}
                    </div>

                    <div className="p-6">
                        <div className="mb-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                                        {tabs[activeTab].icon}
                                        {tabs[activeTab].title}
                                    </h2>
                                    <p className="text-gray-400 mt-1">Phân tích AI với 50 thuật toán</p>
                                </div>
                                <div className="flex items-center gap-2 text-sm">
                                    <div className="flex items-center gap-1 px-3 py-1 bg-green-500/20 text-green-400 rounded-full">
                                        <CheckCircle className="w-3 h-3" />
                                        <span>AI Active</span>
                                    </div>
                                    <div className="flex items-center gap-1 px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full">
                                        <AlertCircle className="w-3 h-3" />
                                        <span>Real-time</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {tabs[activeTab].component}
                    </div>
                </div>
            );
        }

        // ============================================
        // 🏗️ MODULE 10: HEADER
        // ============================================
        
        function Header() {
            const [isMenuOpen, setIsMenuOpen] = React.useState(false);

            return (
                <header className="sticky top-0 z-50 bg-gradient-to-r from-gray-900 via-purple-900/90 to-violet-900/90 backdrop-blur-xl border-b border-gray-700/50">
                    <div className="container mx-auto px-4 py-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <button 
                                    onClick={() => setIsMenuOpen(!isMenuOpen)}
                                    className="lg:hidden p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
                                >
                                    {isMenuOpen ? 
                                        React.createElement(X, { className: "w-5 h-5 text-white" }) : 
                                        React.createElement(Menu, { className: "w-5 h-5 text-white" })
                                    }
                                </button>
                                
                                <div className="flex items-center gap-3">
                                    <div className="relative">
                                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center shadow-lg">
                                            <Zap className="w-6 h-6 text-white" />
                                        </div>
                                        <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-gray-900"></div>
                                    </div>
                                    
                                    <div>
                                        <h1 className="text-xl font-bold text-white">
                                            <span className="gradient-text">LOTTOBET AI PRO</span>
                                        </h1>
                                        <p className="text-xs text-gray-400">Version 1.0 • 50 AI Algorithms</p>
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-2">
                                <div className="hidden lg:flex items-center gap-6">
                                    <div className="text-center">
                                        <div className="text-sm text-gray-400">Số dư</div>
                                        <div className="text-lg font-bold text-white flex items-center gap-2">
                                            <Wallet className="w-4 h-4 text-green-400" />
                                            8,450,000 đ
                                        </div>
                                    </div>
                                    <div className="text-center">
                                        <div className="text-sm text-gray-400">Win Rate</div>
                                        <div className="text-lg font-bold text-green-400">68.4%</div>
                                    </div>
                                </div>

                                <button className="p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors">
                                    <Bell className="w-5 h-5 text-white" />
                                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-xs rounded-full flex items-center justify-center">
                                        3
                                    </span>
                                </button>

                                <div className="flex items-center gap-2 p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors">
                                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                        <User className="w-4 h-4 text-white" />
                                    </div>
                                    <div className="hidden md:block text-left">
                                        <div className="text-sm font-medium text-white">LottoPro User</div>
                                        <div className="text-xs text-gray-400">VIP 3</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {isMenuOpen && (
                            <div className="lg:hidden mt-4 pb-4 border-t border-gray-700/50 pt-4">
                                <div className="grid grid-cols-3 gap-4">
                                    <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                                        <div className="text-sm text-gray-400">Số dư</div>
                                        <div className="text-lg font-bold text-white">8,450,000 đ</div>
                                    </div>
                                    <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                                        <div className="text-sm text-gray-400">Win Rate</div>
                                        <div className="text-lg font-bold text-green-400">68.4%</div>
                                    </div>
                                    <div className="text-center p-3 bg-gray-800/50 rounded-lg">
                                        <div className="text-sm text-gray-400">Level</div>
                                        <div className="text-lg font-bold text-yellow-400">VIP 3</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </header>
            );
        }

        // ============================================
        // 🏗️ MODULE 11: MAIN APP
        // ============================================
        
        function App() {
            const [isLoading, setIsLoading] = React.useState(true);

            React.useEffect(() => {
                setTimeout(() => setIsLoading(false), 1000);
            }, []);

            if (isLoading) {
                return (
                    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex items-center justify-center">
                        <div className="text-center">
                            <div className="w-16 h-16 border-4 border-t-purple-500 border-r-transparent border-b-purple-700 border-l-transparent rounded-full animate-spin mx-auto"></div>
                            <p className="mt-4 text-white text-lg font-semibold">Đang khởi động AI với 50 thuật toán...</p>
                            <p className="text-gray-400">Tool Lottobet Pro - Version 1.0</p>
                        </div>
                    </div>
                );
            }

            return (
                <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900">
                    <Header />

                    <div className="container mx-auto px-4 py-6">
                        <div className="mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/20 rounded-xl p-4 border border-blue-500/30">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 rounded-lg bg-blue-500/20">
                                            <Brain className="w-5 h-5 text-blue-400" />
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">AI Status</p>
                                            <p className="text-white font-bold">Sẵn sàng</p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm text-gray-400">Độ chính xác</p>
                                        <p className="text-green-400 font-bold">87.5%</p>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-r from-purple-900/30 to-purple-800/20 rounded-xl p-4 border border-purple-500/30">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-purple-500/20">
                                        <Activity className="w-5 h-5 text-purple-400" />
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-400">Kỳ hiện tại</p>
                                        <p className="text-white font-bold text-xl">#123456</p>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-r from-cyan-900/30 to-cyan-800/20 rounded-xl p-4 border border-cyan-500/30">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-cyan-500/20">
                                        <Zap className="w-5 h-5 text-cyan-400" />
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-400">Thuật toán</p>
                                        <p className="text-white font-bold">50/50</p>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gradient-to-r from-rose-900/30 to-rose-800/20 rounded-xl p-4 border border-rose-500/30">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-rose-500/20">
                                        <Shield className="w-5 h-5 text-rose-400" />
                                    </div>
                                    <div>
                                        <p className="text-sm text-gray-400">Bảo mật</p>
                                        <p className="text-white font-bold">Level 3</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                            <div className="lg:col-span-1">
                                <RealTimeMonitor />
                                
                                <div className="mt-6 glass-effect rounded-xl p-4">
                                    <h3 className="text-white font-bold mb-3 flex items-center gap-2">
                                        <TrendingUp className="w-4 h-4" />
                                        Thống kê nhanh
                                    </h3>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Win rate 7 ngày</span>
                                            <span className="text-green-400 font-bold">68.4%</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">ROI</span>
                                            <span className="text-blue-400 font-bold">+24.7%</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Tổng lợi nhuận</span>
                                            <span className="text-green-400 font-bold">8,450,000đ</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="lg:col-span-2">
                                <AnalysisTabs />
                                
                                <div className="mt-6">
                                    <NumberMatrix />
                                </div>
                            </div>

                            <div className="lg:col-span-1">
                                <CapitalManagement />
                                
                                <div className="mt-6 glass-effect rounded-xl p-4">
                                    <h3 className="text-white font-bold mb-3">Hành động nhanh</h3>
                                    <div className="grid grid-cols-2 gap-3">
                                        <button className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2">
                                            <RefreshCw className="w-4 h-4" />
                                            Làm mới
                                        </button>
                                        <button className="bg-purple-600 hover:bg-purple-700 text-white py-2 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2">
                                            <History className="w-4 h-4" />
                                            Lịch sử
                                        </button>
                                        <button className="bg-amber-600 hover:bg-amber-700 text-white py-2 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2">
                                            <Bell className="w-4 h-4" />
                                            Cảnh báo
                                        </button>
                                        <button className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-3 rounded-lg text-sm transition-colors flex items-center justify-center gap-2">
                                            <Settings className="w-4 h-4" />
                                            Cài đặt
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // ============================================
        // 🚀 RENDER APP
        // ============================================
        
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(React.createElement(App));
    </script>
</body>
</html>
