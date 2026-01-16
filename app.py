<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LOTTOBET AI TOOL v1.0</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { 
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            margin: 0;
            font-family: system-ui, -apple-system, sans-serif;
        }
        .glass {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .gradient-text {
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <!-- React & Babel -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <!-- Icons (using Font Awesome as backup) -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <script type="text/babel">
        // ==================== SIMPLIFIED VERSION ====================
        // No external dependencies, pure React + Tailwind
        
        const { useState, useEffect } = React;
        
        // Icon Components (Fallback with emojis)
        const Icon = ({ name, className = "w-5 h-5" }) => {
            const icons = {
                brain: '🧠',
                activity: '⚡',
                trendingUp: '📈',
                dollar: '💰',
                clock: '⏱️',
                bell: '🔔',
                user: '👤',
                shield: '🛡️',
                zap: '⚡',
                target: '🎯',
                check: '✅',
                close: '❌',
                alert: '⚠️',
                dice: '🎲',
                hash: '#️⃣',
                chart: '📊',
                scale: '⚖️',
                star: '⭐',
                search: '🔍',
                menu: '☰',
                wallet: '💳',
                settings: '⚙️',
                history: '📜',
                refresh: '🔄',
                calendar: '📅',
                award: '🏆',
                calculator: '🧮',
                pie: '📊',
                book: '📖',
                filter: '🔧'
            };
            return <span className={className}>{icons[name] || '📦'}</span>;
        };

        // ========== COMPONENTS ==========
        function RealTimeMonitor() {
            const [time, setTime] = useState(150);
            
            useEffect(() => {
                const timer = setInterval(() => {
                    setTime(t => t <= 1 ? 150 : t - 1);
                }, 1000);
                return () => clearInterval(timer);
            }, []);
            
            const formatTime = (s) => {
                const m = Math.floor(s / 60);
                const sec = s % 60;
                return `${m}:${sec.toString().padStart(2, '0')}`;
            };
            
            return (
                <div className="glass rounded-2xl p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <Icon name="clock" className="text-blue-400" />
                        <span className="text-white font-bold">Thời gian thực</span>
                    </div>
                    
                    <div className="text-center">
                        <div className="text-4xl font-bold text-white font-mono mb-2">
                            {formatTime(time)}
                        </div>
                        <div className="text-gray-400">Kỳ #123456</div>
                        <div className="flex items-center justify-center gap-2 mt-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                            <span className="text-green-400 text-sm">Đang mở cược</span>
                        </div>
                    </div>
                </div>
            );
        }

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
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
                        {positions.map((pos, idx) => (
                            <div key={idx} className="glass rounded-xl p-3">
                                <h3 className="text-white font-bold text-center mb-2">{pos.name}</h3>
                                {pos.numbers.map((num, i) => (
                                    <div key={i} className="flex items-center justify-between mb-2 p-2 bg-gray-900/50 rounded">
                                        <div className="w-7 h-7 rounded-full bg-purple-600 flex items-center justify-center text-white font-bold">
                                            {num}
                                        </div>
                                        <span className="text-white">{50 - i * 10}%</span>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        function TwoStarAnalysis() {
            const pairs = [
                { pair: "56", prob: 65, rec: "high" },
                { pair: "78", prob: 25, rec: "low" },
                { pair: "34", prob: 72, rec: "high" },
                { pair: "12", prob: 48, rec: "medium" }
            ];
            
            return (
                <div className="grid grid-cols-2 gap-3">
                    {pairs.map((p, i) => (
                        <div key={i} className="glass rounded-xl p-4">
                            <div className="flex justify-between items-center mb-3">
                                <div className="w-12 h-12 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                    <span className="text-white font-bold text-lg">{p.pair}</span>
                                </div>
                                <div className="text-right">
                                    <div className="text-2xl font-bold text-white">{p.prob}%</div>
                                    <div className="text-xs text-gray-400">Xác suất</div>
                                </div>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                                <div className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-500" 
                                     style={{width: `${p.prob}%`}}></div>
                            </div>
                            <div className={`text-center font-bold ${
                                p.rec === 'high' ? 'text-green-400' : 
                                p.rec === 'medium' ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                                {p.rec === 'high' ? 'NÊN ĐÁNH' : p.rec === 'medium' ? 'CÂN NHẮC' : 'TRÁNH'}
                            </div>
                        </div>
                    ))}
                </div>
            );
        }

        function NumberMatrix() {
            const [selected, setSelected] = useState([]);
            
            const numbers = Array.from({length: 99}, (_, i) => i + 1);
            
            return (
                <div className="glass rounded-2xl p-4">
                    <h3 className="text-white font-bold text-xl mb-4">Ma trận số 1-99</h3>
                    <div className="grid grid-cols-8 sm:grid-cols-10 md:grid-cols-12 gap-1">
                        {numbers.map(num => (
                            <button
                                key={num}
                                onClick={() => setSelected(prev => 
                                    prev.includes(num) ? prev.filter(n => n !== num) : [...prev, num]
                                )}
                                className={`w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold transition-all ${
                                    selected.includes(num) 
                                        ? 'bg-gradient-to-r from-yellow-600 to-amber-600 scale-110' 
                                        : 'bg-gray-800 hover:bg-gray-700'
                                }`}
                            >
                                {num}
                            </button>
                        ))}
                    </div>
                    <div className="mt-4 text-gray-400 text-sm">
                        Đã chọn: {selected.length} số | Click để chọn/bỏ chọn
                    </div>
                </div>
            );
        }

        function CapitalManagement() {
            const [capital, setCapital] = useState(10000000);
            const [risk, setRisk] = useState('medium');
            
            const formatMoney = (num) => {
                return new Intl.NumberFormat('vi-VN').format(num) + ' đ';
            };
            
            return (
                <div className="glass rounded-2xl p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <Icon name="dollar" className="text-green-400" />
                        <h3 className="text-white font-bold text-xl">Quản lý vốn</h3>
                    </div>
                    
                    <div className="text-center mb-6">
                        <div className="text-3xl font-bold text-white">{formatMoney(capital)}</div>
                        <div className="text-gray-400">Tổng vốn hiện có</div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 mb-6">
                        {['low', 'medium', 'high'].map(lvl => (
                            <button
                                key={lvl}
                                onClick={() => setRisk(lvl)}
                                className={`py-2 rounded-lg font-bold ${
                                    risk === lvl 
                                        ? lvl === 'low' ? 'bg-green-600 text-white' :
                                          lvl === 'medium' ? 'bg-yellow-600 text-white' :
                                          'bg-red-600 text-white'
                                        : 'bg-gray-800 text-gray-300'
                                }`}
                            >
                                {lvl === 'low' ? 'THẤP' : lvl === 'medium' ? 'TRUNG BÌNH' : 'CAO'}
                            </button>
                        ))}
                    </div>
                    
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-300">Cược tối đa:</span>
                            <span className="text-white font-bold">
                                {formatMoney(capital * (risk === 'low' ? 0.01 : risk === 'medium' ? 0.03 : 0.05))}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-300">Stop-loss:</span>
                            <span className="text-red-400 font-bold">30%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-300">Take-profit:</span>
                            <span className="text-green-400 font-bold">50%</span>
                        </div>
                    </div>
                </div>
            );
        }

        function Header() {
            return (
                <header className="glass border-b border-gray-700/50 py-3 px-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                <Icon name="zap" className="text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold gradient-text">LOTTOBET AI PRO</h1>
                                <p className="text-xs text-gray-400">v1.0 • 50 AI Algorithms</p>
                            </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                            <button className="relative p-2">
                                <Icon name="bell" />
                                <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-xs rounded-full flex items-center justify-center">3</span>
                            </button>
                            <div className="flex items-center gap-2 p-2 rounded-lg bg-gray-800/50">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                    <Icon name="user" className="text-white" />
                                </div>
                                <span className="text-white font-medium hidden sm:block">VIP User</span>
                            </div>
                        </div>
                    </div>
                </header>
            );
        }

        function AnalysisTabs() {
            const [activeTab, setActiveTab] = useState(0);
            
            const tabs = [
                { id: 0, title: "5 TINH", icon: "dice", component: <FiveStarAnalysis /> },
                { id: 1, title: "2 TINH", icon: "hash", component: <TwoStarAnalysis /> },
                { id: 2, title: "SỐ MATRIX", icon: "target", component: <NumberMatrix /> }
            ];
            
            return (
                <div className="glass rounded-2xl overflow-hidden">
                    <div className="flex border-b border-gray-700/50">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 py-3 flex items-center justify-center gap-2 border-b-2 ${
                                    activeTab === tab.id 
                                        ? 'text-white border-blue-500 bg-blue-500/10' 
                                        : 'text-gray-400 border-transparent'
                                }`}
                            >
                                <Icon name={tab.icon} />
                                <span className="font-semibold">{tab.title}</span>
                            </button>
                        ))}
                    </div>
                    
                    <div className="p-4">
                        {tabs[activeTab].component}
                    </div>
                </div>
            );
        }

        function App() {
            return (
                <div className="min-h-screen">
                    <Header />
                    
                    <div className="container mx-auto px-4 py-6">
                        {/* Stats Cards */}
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                            <div className="glass rounded-xl p-4">
                                <div className="flex items-center gap-3">
                                    <Icon name="brain" className="text-blue-400" />
                                    <div>
                                        <p className="text-sm text-gray-400">AI Status</p>
                                        <p className="text-white font-bold">Sẵn sàng</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="glass rounded-xl p-4">
                                <div className="flex items-center gap-3">
                                    <Icon name="activity" className="text-purple-400" />
                                    <div>
                                        <p className="text-sm text-gray-400">Kỳ hiện tại</p>
                                        <p className="text-white font-bold">#123456</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="glass rounded-xl p-4">
                                <div className="flex items-center gap-3">
                                    <Icon name="trendingUp" className="text-green-400" />
                                    <div>
                                        <p className="text-sm text-gray-400">Win Rate</p>
                                        <p className="text-green-400 font-bold">68.4%</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="glass rounded-xl p-4">
                                <div className="flex items-center gap-3">
                                    <Icon name="shield" className="text-yellow-400" />
                                    <div>
                                        <p className="text-sm text-gray-400">Bảo mật</p>
                                        <p className="text-white font-bold">Level 3</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        {/* Main Content */}
                        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                            {/* Left Sidebar */}
                            <div className="lg:col-span-1 space-y-6">
                                <RealTimeMonitor />
                                
                                <div className="glass rounded-xl p-4">
                                    <h3 className="text-white font-bold mb-3">Thống kê nhanh</h3>
                                    <div className="space-y-2">
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Win rate 7 ngày</span>
                                            <span className="text-green-400">68.4%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">ROI</span>
                                            <span className="text-blue-400">+24.7%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Tổng lợi nhuận</span>
                                            <span className="text-green-400">8,450,000đ</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Main Area */}
                            <div className="lg:col-span-2">
                                <AnalysisTabs />
                            </div>
                            
                            {/* Right Sidebar */}
                            <div className="lg:col-span-1">
                                <CapitalManagement />
                                
                                <div className="glass rounded-xl p-4 mt-6">
                                    <h3 className="text-white font-bold mb-3">Hành động nhanh</h3>
                                    <div className="grid grid-cols-2 gap-2">
                                        <button className="bg-blue-600 hover:bg-blue-700 text-white py-2 rounded text-sm">
                                            <Icon name="refresh" className="inline mr-1" /> Làm mới
                                        </button>
                                        <button className="bg-purple-600 hover:bg-purple-700 text-white py-2 rounded text-sm">
                                            <Icon name="history" className="inline mr-1" /> Lịch sử
                                        </button>
                                        <button className="bg-amber-600 hover:bg-amber-700 text-white py-2 rounded text-sm">
                                            <Icon name="bell" className="inline mr-1" /> Cảnh báo
                                        </button>
                                        <button className="bg-gray-700 hover:bg-gray-600 text-white py-2 rounded text-sm">
                                            <Icon name="settings" className="inline mr-1" /> Cài đặt
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }

        // Render the app
        const rootElement = document.getElementById('root');
        const root = ReactDOM.createRoot(rootElement);
        root.render(<App />);
    </script>
</body>
</html>
