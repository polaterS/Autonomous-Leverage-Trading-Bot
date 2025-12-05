# 🎯 LEVEL-BASED TRADING v5.0.3 - Closed Candle + Stop Hunt Detection
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V503_CLOSED_CANDLE_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.3: CLOSED CANDLE + STOP HUNT DETECTION!" && \
    echo "   🛡️ CRITICAL FIX: Professional Entry Timing" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.3 FEATURES:" && \
    echo "      ✅ Pattern detection on CLOSED candle (not forming)" && \
    echo "      ✅ Stop Hunt Reversal pattern detection" && \
    echo "      ✅ False breakout protection" && \
    echo "      ✅ Professional candle close waiting" && \
    echo "      ✅ Uses iloc[-2] for last CLOSED candle" && \
    echo "      ✅ Prevents false signals from forming candles" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ WHY THIS MATTERS:" && \
    echo "      ❌ OLD: Used forming candle (could change!)" && \
    echo "      ✅ NEW: Uses closed candle (final, reliable)" && \
    echo "      🎯 Stop hunts now trigger entries (not stops)" && \
    echo "      📈 Reduced false signals significantly"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 🎯 CACHE BUST MARKER: v5.0.3 - Closed Candle + Stop Hunt
# Current deployment: 20251205_V503_CLOSED_CANDLE_FIX
# Changes: Pattern detection now uses CLOSED candle (not forming)
#   🛡️ v5.0.3: Closed Candle + Stop Hunt Detection
#      ✅ Pattern detection on CLOSED candle (iloc[-2])
#      ✅ Stop Hunt Reversal pattern detection
#      ✅ False breakout protection
#      ✅ Professional candle close waiting
#      ✅ Prevents false signals from forming candles
#   📊 Previous versions:
#      ✅ v5.0.2: Partial TP + Breakeven System
#      ✅ v5.0.1: /analyze BTC Command
#      ✅ v5.0.0: Level-Based Trading System (complete redesign)
COPY . .

# 🔥 NUCLEAR OPTION: Delete ALL Python cache IMMEDIATELY after copy
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPYCACHEPREFIX=/dev/null

# Expose health check port
EXPOSE 8000

# Health check to ensure bot is running
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the trading bot (FORCE clear ALL Python cache to ensure fresh code)
CMD echo "🔥 DELETING PYTHON CACHE BEFORE STARTUP..." && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app -type f -name "*.pyo" -delete 2>/dev/null || true && \
    echo "✅ Cache deletion complete, verifying..." && \
    echo "Remaining .pyc files: $(find /app -type f -name '*.pyc' | wc -l)" && \
    echo "🚀 Starting bot with -B flag (bypass bytecode)..." && \
    python -u -B main.py
