# 🚀 Railway.app Deployment Guide

Bu bot'u 7/24 cloud'da çalıştırmak için adım adım rehber.

## 📋 Ön Hazırlık (5 dakika)

### 1. GitHub'a Push Et

```bash
git add .
git commit -m "feat: Railway deployment ready"
git push origin main
```

### 2. Railway Hesabı Oluştur

1. **Railway.app'e Git**: https://railway.app
2. **"Start a New Project"** tıkla
3. **GitHub ile giriş yap** (OAuth)
4. Railway'e GitHub repo erişimi ver

## 🔧 Deployment (3 dakika)

### 3. Projeyi Oluştur

1. Railway Dashboard'da **"New Project"**
2. **"Deploy from GitHub repo"** seç
3. **`Autonomous-Leverage-Trading-Bot`** repo'sunu seç
4. Railway otomatik olarak deploy başlatsın (bekle)

### 4. PostgreSQL Ekle

1. Proje sayfasında **"+ New"** → **"Database"** → **"Add PostgreSQL"**
2. Railway otomatik olarak `DATABASE_URL` environment variable'ı ekler
3. 1-2 dakika bekle (database hazırlanıyor)

### 5. Redis Ekle

1. Proje sayfasında **"+ New"** → **"Database"** → **"Add Redis"**
2. Railway otomatik olarak `REDIS_URL` environment variable'ı ekler

## 🔐 Environment Variables Ekle (5 dakika)

Proje sayfasında **"Variables"** sekmesine git ve şunları ekle:

### Exchange API Keys
```
BINANCE_API_KEY=ZufKfC3v2CjZDIhx3AcSvgynR0Y9MBGF9gbrRbw8877HXMGQsehTtTTzCLvaN8Hw
BINANCE_SECRET_KEY=ifUpDOkT9Mz276rA8JfTj0yQPIAcjW9hiDOLxWJLwxc1emWMYcNF8eB8F9VjhWgC
```

### AI API Keys
```
OPENROUTER_API_KEY=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
DEEPSEEK_API_KEY=sk-4b591636c1d74c999a6754170237ca65
```

### Telegram
```
TELEGRAM_BOT_TOKEN=8360346117:AAE8e9iR6IsSNi-nddqBghmKdp42_1_STvI
TELEGRAM_CHAT_ID=1392205221
```

### Trading Configuration
```
INITIAL_CAPITAL=100.00
MAX_LEVERAGE=50
POSITION_SIZE_PERCENT=0.80
MIN_STOP_LOSS_PERCENT=0.05
MAX_STOP_LOSS_PERCENT=0.10
MIN_PROFIT_USD=2.50
MIN_AI_CONFIDENCE=0.60
SCAN_INTERVAL_SECONDS=300
POSITION_CHECK_SECONDS=60
```

### Risk Management
```
DAILY_LOSS_LIMIT_PERCENT=0.10
MAX_CONSECUTIVE_LOSSES=3
```

### Feature Flags
```
USE_PAPER_TRADING=true
AUTO_START_LIVE_TRADING=true
ENABLE_DEBUG_LOGS=false
```

### AI Configuration
```
AI_CACHE_TTL_SECONDS=300
AI_TIMEOUT_SECONDS=30
```

### ⚠️ ÖNEMLI
- `DATABASE_URL` ve `REDIS_URL` otomatik eklendi (dokunma!)
- `DB_PASSWORD` gerekmiyor (Railway otomatik halletti)

## ✅ Deploy Tamamlandı!

### 6. Bot'u Başlat

1. **"Deployments"** sekmesinde en son deployment'ı gör
2. **"View Logs"** ile logları izle
3. Şu mesajları görmelisin:
   ```
   ✅ Database connected
   ✅ Telegram bot initialized
   ✅ Exchange connected
   🚀 AUTONOMOUS TRADING ENGINE STARTED
   ```

### 7. Test Et

Telegram'da bot'a yaz:
- `/status` - Bot durumunu gör
- `/scan` - Market taraması başlat
- `/help` - Tüm komutları gör

## 🎛️ Bot Kontrolü

### Bot'u Durdur
Railway Dashboard → **"Settings"** → **"Sleep"** veya **"Delete"**

### Bot'u Yeniden Başlat
Railway Dashboard → **"Deployments"** → **"Restart"**

### Logları İzle
Railway Dashboard → **"Deployments"** → **"View Logs"**

### Environment Variable Değiştir
Railway Dashboard → **"Variables"** → Değiştir → Otomatik restart

## 💰 Maliyet

- **Free Tier**: İlk $5 ücretsiz (her ay yenilenir)
- **Starter Plan**: $5/ay
  - PostgreSQL dahil
  - Redis dahil
  - 500 saat çalışma (7/24 için yeterli)
  - 8GB RAM + 8 vCPU

## 🔄 Güncellemeler

Bot'u güncellemek için:

```bash
# Yerel değişikliklerini yap
git add .
git commit -m "Update bot"
git push origin main
```

Railway otomatik olarak yeni versiyonu deploy eder (1-2 dakika)!

## 🆘 Sorun Giderme

### Bot Başlamıyor
1. **Logs'u kontrol et**: Railway Dashboard → "View Logs"
2. **Environment variables'ı kontrol et**: Hepsi doğru mu?
3. **Database bağlı mı**: PostgreSQL ve Redis çalışıyor mu?

### Bot Çalışıyor Ama Trade Yapmıyor
1. **Daily loss limit**: $-11.27 geçtiysen bugün kapandı (yarın reset)
2. **Paper trading mode**: `USE_PAPER_TRADING=true` ise gerçek para kullanmaz

### Telegram Bot Yanıt Vermiyor
1. **Bot token doğru mu**: `TELEGRAM_BOT_TOKEN` kontrol et
2. **Chat ID doğru mu**: `TELEGRAM_CHAT_ID` kontrol et
3. **Logs'da hata var mı**: Railway'de logları kontrol et

## 📊 Monitoring

Railway Dashboard'dan izle:
- **CPU Usage**: Bot ne kadar işlemci kullanıyor
- **Memory Usage**: RAM kullanımı
- **Logs**: Anlık bot aktivitesi
- **Metrics**: İstek sayısı, response time vb.

## 🎉 Tamamlandı!

Bot'un artık 7/24 Railway.app'te çalışıyor!

**Telefon Bildirimleri İçin:**
Railway mobil app yükle → Push notifications aç → Bot durumunu telefondan takip et

**Soru/Sorun:**
GitHub Issues: https://github.com/anthropics/claude-code/issues
