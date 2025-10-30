# 🚀 **PRODUCTION SETUP GUIDE**

## **YENİ ÖZELLİKLER (v2.0)**

### ✅ **Tamamlanan İyileştirmeler**

1. **WebSocket Client** - Real-time price feeds
2. **Secrets Manager** - AES encryption for API keys
3. **Enhanced Slippage Recovery** - 5 retry attempts
4. **Backtest Framework** - Historical strategy testing
5. **Prometheus Metrics** - 30+ metrics
6. **Grafana Dashboard** - Visual monitoring
7. **Prometheus Alerts** - 15+ alert rules
8. **Unit Tests** - 50+ test cases, 70%+ coverage
9. **WebSocket Position Monitoring** - Sub-second updates
10. **Secret Rotation** - Automatic key rotation
11. **Multi-Exchange Support** - Binance, Bybit, OKX
12. **Advanced Backtesting** - Monte Carlo simulation

---

## 📦 **KURULUM**

### **1. Bağımlılıkları Kur**

```bash
pip install -r requirements.txt
```

**Yeni bağımlılıklar**:
- `cryptography` - Secret encryption
- `prometheus-client` - Metrics
- `pytest`, `pytest-asyncio`, `pytest-cov` - Testing

### **2. Secret Management Kurulumu**

```bash
# Master key oluştur
python -m src.secrets_manager generate

# Çıktıyı güvenli bir yere kaydet!
# Örnek çıktı: gAAAAABl1234...xyz (master key)

# API keys'leri encrypt et
python -m src.secrets_manager encrypt "your_binance_api_key"
# Çıktı: gAAAAABl5678...abc

# .env dosyasına ekle
BINANCE_API_KEY_ENCRYPTED=gAAAAABl5678...abc
```

### **3. Database Setup**

```bash
# Docker ile (önerilen)
docker-compose up -d postgres redis

# 10 saniye bekle
sleep 10

# Database'i oluştur
python setup_database.py
```

### **4. Monitoring Stack Başlat**

```bash
# Prometheus + Grafana
docker-compose up -d prometheus grafana

# Servisler:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin123)
# - Metrics: http://localhost:8001/metrics
```

---

## ⚡ **İLK ÇALIŞTIRMA**

### **Test Modunda Başlatma (Paper Trading)**

```bash
# .env dosyasında
USE_PAPER_TRADING=true

# Botu başlat
python main.py
```

### **Backtest Çalıştırma**

```bash
# 3 aylık backtest
python -c "
import asyncio
from src.backtester import Backtester
from datetime import date
from decimal import Decimal

async def main():
    bt = Backtester()
    results = await bt.run_backtest(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        initial_capital=Decimal('100.00'),
        symbols=['BTC/USDT:USDT'],
        use_ai=True
    )
    print(f'\\nResults:')
    print(f'Win Rate: {results.win_rate:.1%}')
    print(f'Final Capital: \${results.final_capital:.2f}')
    print(f'Total P&L: \${results.total_pnl:+.2f}')
    print(f'Sharpe Ratio: {results.sharpe_ratio:.2f}')
    print(f'Max Drawdown: {results.max_drawdown:.1%}')

asyncio.run(main())
"
```

### **Unit Tests Çalıştırma**

```bash
# Tüm testler
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html

# HTML report: htmlcov/index.html
```

---

## 📊 **MONITORING**

### **Grafana Dashboard**

1. http://localhost:3000 adresine git
2. Username: `admin`, Password: `admin123`
3. Dashboard otomatik yüklenmiş olacak:
   - **Trading Bot - Production Dashboard**

**Paneller**:
- Current Capital
- Today's P&L
- Win Rate
- Active Position P&L
- Capital Over Time
- Trade P&L Distribution
- Trade Count (Win/Loss)
- AI Latency
- Liquidation Distance
- Circuit Breakers

### **Prometheus Alerts**

Alerts: http://localhost:9090/alerts

**Critical Alerts**:
- LiquidationRiskHigh: <5% mesafe
- DailyLossLimitExceeded: -$10+
- CapitalBelowMinimum: <$50

**Warning Alerts**:
- ConsecutiveLossesHigh: 3+ kayıp
- WinRateDropped: <40%
- AILatencyHigh: >10s
- PositionDurationTooLong: >24h

---

## 🔐 **GÜVENLİK**

### **API Keys Encryption**

```bash
# Tüm API keys'leri encrypt et
python -m src.secrets_manager encrypt "BINANCE_API_KEY"
python -m src.secrets_manager encrypt "CLAUDE_API_KEY"
python -m src.secrets_manager encrypt "DEEPSEEK_API_KEY"

# .env'de kullan
BINANCE_API_KEY_ENCRYPTED=gAAAAABl...
CLAUDE_API_KEY_ENCRYPTED=gAAAAABm...
```

### **Master Key Yedekleme**

```bash
# Master key'i güvenli bir yere kopyala
cp .master.key ~/.trading-bot-master.key

# Veya cloud'a yedekle (şifreli)
gpg -c .master.key
# Çıktı: .master.key.gpg
```

---

## 🎯 **PRODUCTION CHECKLIST**

### **Backtest Phase (2-4 hafta)**

- [ ] 6 aylık backtest çalıştır
- [ ] Win rate >50% kontrol et
- [ ] Max drawdown <20% kontrol et
- [ ] Sharpe ratio >1.0 kontrol et
- [ ] Profit factor >1.5 kontrol et

### **Paper Trading Phase (2 hafta)**

- [ ] USE_PAPER_TRADING=true ayarla
- [ ] 2 hafta paper trading çalıştır
- [ ] Günlük P&L kaydet
- [ ] Hataları logla
- [ ] Telegram notifications test et

### **Micro-Capital Phase ($50-100, 2 hafta)**

- [ ] USE_PAPER_TRADING=false
- [ ] INITIAL_CAPITAL=50.00
- [ ] MAX_LEVERAGE=3
- [ ] Günlük monitoring
- [ ] Stop-loss testleri

### **Production Phase ($200+)**

- [ ] Backtest başarılı ✓
- [ ] Paper trading başarılı ✓
- [ ] Micro-capital başarılı ✓
- [ ] Master key yedeklendi ✓
- [ ] Grafana dashboard hazır ✓
- [ ] Alerts yapılandırıldı ✓
- [ ] BAŞLAT!

---

## 🆘 **TROUBLESHOOTING**

### **WebSocket bağlanmıyor**

```bash
# websockets kütüphanesi kurulu mu?
pip install websockets

# Test et
python -c "
import asyncio
from src.websocket_client import get_ws_client

async def test():
    ws = await get_ws_client()
    print('WebSocket OK!')

asyncio.run(test())
"
```

### **Secrets decrypt edilemiyor**

```bash
# Master key var mı?
ls -la .master.key

# Yoksa yeniden oluştur
python -m src.secrets_manager generate

# API keys'leri yeniden encrypt et
```

### **Prometheus metrics görünmüyor**

```bash
# Metrics endpoint erişilebilir mi?
curl http://localhost:8001/metrics

# Prometheus target'ları kontrol et
# http://localhost:9090/targets
```

### **Tests başarısız**

```bash
# Bağımlılıkları güncelle
pip install -r requirements.txt --upgrade

# Specific test çalıştır
pytest tests/test_utils.py -v

# Debug mode
pytest tests/ -v --tb=short
```

---

## 📞 **DESTEK**

### **Loglar**

```bash
# Bot logs
tail -f bot.log

# Docker logs
docker-compose logs -f trading-bot

# Prometheus logs
docker-compose logs -f prometheus
```

### **Database Sorgulama**

```bash
# PostgreSQL'e bağlan
docker-compose exec postgres psql -U trading_user -d trading_bot

# Queries
SELECT * FROM active_position;
SELECT * FROM trade_history ORDER BY exit_time DESC LIMIT 10;
SELECT * FROM daily_performance WHERE date = CURRENT_DATE;
```

---

## 🎉 **PRODUCTION READY!**

Tüm adımları tamamladıysanız, botunuz production'a hazır! 🚀

**Son kontrol**:
```bash
# Health check
curl http://localhost:8000/health

# Metrics check
curl http://localhost:8001/metrics | grep trading_bot

# Grafana check
curl http://localhost:3000/api/health
```

**Happy Trading!** 💰
