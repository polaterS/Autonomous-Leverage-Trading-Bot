# BTC CORRELATION FILTER - TEMPORARILY DISABLED

## 🧪 Geçici Test Modu

**Tarih**: 2025-11-20 23:15
**Sebep**: Bot hiç pozisyon açmıyor - tüm LONG trade'ler BTC bearish momentum nedeniyle bloke ediliyor
**Durum**: ⚠️ DISABLED (Geçici)

---

## 📊 Sorun Analizi

### BEFORE (Filter Aktif):
```
🔍 108 coin tarandı
❌ 0 fırsat bulundu

Rejection nedenleri:
- LONG: "BTC bearish momentum (0.5%) conflicts with LONG - wait for BTC recovery"
- SHORT: "Price too far from resistance (5-12% away)"
```

**Sonuç**: BTC %0.5 düşüyor → TÜM LONG trade'ler bloke ediliyor (108/108 coin)

### AFTER (Filter Disabled):
```
⏳ Test ediliyor...
```

---

## ⚙️ Değişiklik Detayları

### **Dosya**: `src/price_action_analyzer.py`
**Lines**: 988-1011 (commented out)

**BEFORE:**
```python
if btc_ohlcv and len(btc_ohlcv) >= 3 and symbol != 'BTC/USDT:USDT':
    # Calculate BTC trend
    btc_trend_direction = 'UP' if btc_close > btc_open else 'DOWN'
    btc_momentum_pct = abs(btc_close - btc_open) / btc_open * 100

    # LONG check: If BTC bearish, skip LONG
    if ml_signal == 'BUY' and btc_trend_direction == 'DOWN' and btc_momentum_pct > 0.5:
        return result  # ❌ BLOCKED

    # SHORT check: If BTC bullish, skip SHORT
    if ml_signal == 'SELL' and btc_trend_direction == 'UP' and btc_momentum_pct > 0.5:
        return result  # ❌ BLOCKED
```

**AFTER:**
```python
# ⚠️ TEMPORARILY DISABLED: Testing without BTC correlation filter
# (all lines commented out)
```

---

## 📈 Beklenen Sonuçlar

### ✅ İYİ SENARYO:
- Bot daha fazla trade açar
- Win rate 70-80% kalır (kabul edilebilir)
- Günlük trade sayısı 10-20'ye çıkar
- **Karar**: Filter'i kapalı tut

### ⚠️ KÖTÜ SENARYO:
- Win rate %60'ın altına düşer
- Counter-trend pozisyonlar açılır (BTC ters gidince loss)
- Loss streak artar
- **Karar**: Filter'i tekrar aç

---

## 🎯 İzlenecek Metrikler

### **1. Win Rate**
- **Hedef**: 70%+ (kabul edilebilir)
- **Alarm**: <65% (filter'i geri aç)
- **Önceki**: 80% (filter aktifken, bugün 4W/1L)

### **2. Trade Sıklığı**
- **Hedef**: 10-20 trade/gün
- **Önceki**: 5 trade/gün (2 saat market aktif, 2 saat 0 trade)

### **3. Average P&L**
- **Hedef**: $2-4 profit per win
- **Önceki**: $2.30 avg win, $2.17 avg loss

### **4. Loss Streak**
- **Alarm**: 3+ consecutive losses
- **Önceki**: 1 loss out of 5 trades

---

## 🔄 Geri Açma Talimatları

Eğer win rate düşerse veya loss streak uzarsa:

### **Option 1: Tamamen Geri Aç (Recommended)**
```python
# src/price_action_analyzer.py - Lines 988-1011
# Uncomment all lines (remove # from start of lines)
```

### **Option 2: Daha Esnek Threshold**
```python
# Line 999: Change threshold from 0.5% to 1.0%
if ml_signal == 'BUY' and btc_trend_direction == 'DOWN' and btc_momentum_pct > 1.0:  # Was 0.5
```

### **Option 3: Sadece Güçlü BTC Hareketlerinde Bloke Et**
```python
# Line 999: Change threshold from 0.5% to 2.0%
if ml_signal == 'BUY' and btc_trend_direction == 'DOWN' and btc_momentum_pct > 2.0:  # Was 0.5
```

---

## 📝 Test Logları

### **Test 1: 2025-11-20 23:15 - 23:30 (15 dakika)**
- ⏳ Bekleniyor...
- Hedef: En az 1-2 LONG fırsat bulması

### **Test 2: 2025-11-20 23:30 - 00:00 (30 dakika)**
- ⏳ Bekleniyor...
- Hedef: Trade açıp kapatması, P&L kaydı

### **Test 3: 2025-11-21 00:00 - 08:00 (Gece)**
- ⏳ Bekleniyor...
- Not: Time Filter gece 00:00-01:00 ve 05:00-06:00 bloke eder

### **Test 4: 2025-11-21 08:00 - 14:00 (Sabah)**
- ⏳ Bekleniyor...
- Hedef: En iyi trading saatleri, 5-10 trade expected

---

## ⚠️ RISK UYARISI

**BTC Correlation filter'i neden vardı:**
- BTC tüm altcoin piyasasını yönlendiriyor
- BTC düşerken LONG açmak = counter-trend = riskli
- BTC yükselirken SHORT açmak = counter-trend = riskli

**Filter kapalıyken riskler:**
- Counter-trend pozisyonlar açılabilir
- BTC ters giderse tüm altcoinler etkilenir
- Loss rate artabilir

**Güvenlik önlemleri (hala aktif):**
- ✅ Stop loss: 8-12% (değişmedi)
- ✅ Daily loss limit: 10% ($20 max loss/day)
- ✅ Max consecutive losses: 5 (circuit breaker)
- ✅ Time Filter: Toxic hours bloke (değişmedi)
- ✅ S/R distance checks: Hala çalışıyor

---

## 📊 İlk Sonuçlar (Updates)

### **2025-11-20 23:15 - Deploy edildi**
- Railway auto-deploy başladı
- ~2 dakika içinde aktif olacak
- İlk scan 23:17'de expected

### **2025-11-20 23:17 - İlk Scan**
- ⏳ Waiting for first scan results...

---

## 🎯 Karar Kriteri

**24 saat sonra değerlendirme:**

| Metrik | Filter Aktif (Bugün) | Filter Kapalı (Test) | Karar |
|--------|---------------------|---------------------|--------|
| Win Rate | 80% (4W/1L) | ⏳ Test | >70% ise kapat |
| Trades/Day | 5 (2 saat market) | ⏳ Test | 10-20 ise kapat |
| Avg P&L | +$2.30 | ⏳ Test | Pozitif ise kapat |
| Loss Streak | Max 1 | ⏳ Test | <3 ise kapat |

**Sonuç:**
- ✅ **Kapalı tut** if: Win rate >70% + More trades + Positive P&L
- ❌ **Geri aç** if: Win rate <65% + Loss streak >3

---

## 📝 Notlar

1. Bu GEÇİCİ bir test - kalıcı değil
2. Railway'de otomatik deploy edildi (commit: da88241)
3. Kod comment'li - kolayca geri açılabilir
4. 24-48 saat test edip karar vereceğiz
5. Win rate düşerse HEMEN geri açacağız

---

**Status**: ⚠️ TESTING (Geçici olarak kapalı)
**Revert Ready**: ✅ Yes (uncommenting 24 lines)
**Risk Level**: 🟡 Medium (monitored closely)

**Güncellenecek...**
