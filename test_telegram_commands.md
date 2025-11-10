# Telegram Bot Komutları Test Listesi

## ✅ TEMEL KOMUTLAR

### `/start`
- **İşlev:** Botu başlatır ve hoşgeldin mesajı gösterir
- **Butonlar:** Status, Positions, History, Scan, Help
- **Test:** Kullanıcıya inline keyboard gösteriliyor mu?

### `/help`
- **İşlev:** Tüm komutların listesini gösterir
- **Test:** Komut listesi eksiksiz mi?

### `/status`
- **İşlev:** Bot durumu, sermaye, günlük P&L gösterir
- **Butonlar:** Start Bot, Stop Bot, Positions, Scan
- **Test:**
  - Durum doğru gösteriliyor mu? (🟢 RUNNING / 🔴 STOPPED)
  - Sermaye doğru mu?
  - Günlük P&L hesaplanıyor mu?

## 📊 POZİSYON YÖNETİMİ

### `/positions`
- **İşlev:** Açık pozisyonların detaylarını gösterir
- **Test:**
  - Tüm pozisyonlar listeleniyor mu?
  - P&L değerleri güncel mi?
  - Entry/Current price doğru mu?

### `/history`
- **İşlev:** Son 20 trade'i gösterir
- **Test:**
  - Trade history doğru sıralanmış mı? (en yeni en üstte)
  - P&L değerleri doğru mu?
  - Exit reason gösteriliyor mu?

### `/closeall`
- **İşlev:** TÜM açık pozisyonları kapatır
- **Test:**
  - Tüm pozisyonlar kapanıyor mu?
  - P&L doğru hesaplanıyor mu?
  - Sermaye güncelleniy or mu?

## 🤖 BOT KONTROLÜ

### `/startbot`
- **İşlev:** Trading bot'u başlatır (yeni pozisyon açmaya izin verir)
- **Test:**
  - is_trading_enabled true oluyor mu?
  - Bot yeni pozisyon açabiliyor mu?

### `/stopbot`
- **İşlev:** Trading bot'u durdurur (YENİ pozisyon açmaz, mevcut pozisyonları izler)
- **Test:**
  - is_trading_enabled false oluyor mu?
  - Bot yeni pozisyon açmıyor mu? ✅ (Bu FIX edildi!)
  - Mevcut pozisyonlar izlenmeye devam ediyor mu?

### `/reset`
- **İşlev:** Circuit breaker'ları resetler (günlük loss limiti aşıldığında)
- **Test:**
  - Daily loss reset oluyor mu?
  - Bot tekrar trade açabiliyor mu?

## 💰 SERMAYE YÖNETİMİ

### `/setcapital <miktar>`
- **Örnek:** `/setcapital 1000`
- **İşlev:** Mevcut sermayeyi ayarlar
- **Test:**
  - Sermaye günceleniyor mu?
  - Max position count doğru hesaplanıyor mu? (capital / 100)
  - Database'e kaydediliyor mu?

## 📈 ANALİZ KOMUTLARI

### `/scan`
- **İşlev:** Market scan başlatır, fırsatları gösterir
- **Test:**
  - 35 symbol taranıyor mu?
  - AI analiz yapılıyor mu?
  - Opportunity scoring çalışıyor mu?
  - Market breadth gösteriliyor mu?

### `/daily`
- **İşlev:** Günlük performans raporu
- **Test:**
  - Bugünkü trade sayısı doğru mu?
  - Win rate doğru hesaplanıyor mu?
  - Timezone Turkey time (UTC+3) mi? ✅ (Fix edildi!)
  - Trade detayları gösteriliyor mu?

### `/chart <symbol>`
- **Örnek:** `/chart BTCUSDT`
- **İşlev:** Symbol için chart gösterir
- **Test:**
  - Chart generate ediliyor mu?
  - Indicators gösteriliyor mu?
  - Entry/exit noktaları işaretli mi?

### `/mlstats`
- **İşlev:** ML model performans istatistikleri
- **Test:**
  - Model accuracy gösteriliyor mu?
  - Symbol-specific performance var mı?
  - Pattern win rates gösteriliyor mu?

### `/mlinsights`
- **İşlev:** ML model'in öğrendikleri ve öneriler
- **Test:**
  - Winning patterns listeleniyor mu?
  - Losing patterns gösteriliyor mu?
  - Recommendations var mı?

## 🔘 INLINE BUTONLAR

### Status Button
- `/start` mesajındaki "Status" butonu
- `handle_status_button()` çağrılıyor mu?

### Positions Button
- `/start` mesajındaki "Positions" butonu
- Özet bilgi gösteriliyor mu?

### History Button
- Son 5 trade gösteriliyor mu?

### Scan Button
- Market scan başlatılıyor mu?

### Start Bot / Stop Bot Buttons
- Komut çalıştırılıyor mu?
- Durum mesajı gösteriliyor mu?

### Chart Button
- Symbol seçimi için keyboard gösteriliyor mu?

## 🎯 ÖNCELİKLİ TEST ALANLAR

1. ✅ **FIXED: /stopbot** - Bot durduğunda yeni pozisyon açmıyor mu?
2. ✅ **FIXED: /daily** - Timezone Turkey time gösteriyor mu?
3. **ML Exit** - Fallback logic çalışıyor mu? (Test edilecek)
4. **Historical Boost** - LONG boost uygulanıyor mu? (Test edilecek)
5. **/setcapital** - Dynamic position count doğru hesaplanıyor mu?
6. **/closeall** - Tüm pozisyonlar düzgün kapanıyor mu?

## 📝 TEST PROSEDÜRÜ

1. Bot'u Railway'de deploy et
2. Telegram'dan her komutu sırayla test et
3. Logları kontrol et (Railway Deploy Logs)
4. Database değişikliklerini kontrol et
5. Hataları not et ve fix'le

## ⚠️ BİLİNEN SORUNLAR

- ✅ **FIXED:** /stopbot ignored - Bot yeni pozisyon açmaya devam ediyordu
- ✅ **FIXED:** ML Exit never triggers - Confidence threshold ve fallback logic iyileştirildi
- ✅ **FIXED:** SHORT bias - Historical performance boost eklendi
- ✅ **FIXED:** Timezone UTC - Turkey time (UTC+3) uygulandı
