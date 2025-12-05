# 🎯 LEVEL-BASED TRADING v5.0.5 - RSI Direction Filter
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V505_RSI_DIRECTION_FILTER
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.5: RSI DIRECTION FILTER!" && \
    echo "   🛡️ CRITICAL FIX: Prevent counter-trend entries!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.5 FEATURES:" && \
    echo "      ✅ RSI Direction Filter added" && \
    echo "      ✅ RSI < 30 (oversold) → Block SHORT" && \
    echo "      ✅ RSI > 70 (overbought) → Block LONG" && \
    echo "      ✅ Prevents APT-like losses (SHORT at RSI 19)" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ WHY THIS MATTERS:" && \
    echo "      ❌ APT SHORT at RSI 19 = Counter-trend = LOSS" && \
    echo "      ✅ Now: RSI oversold blocks SHORT entries" && \
    echo "      ✅ Now: RSI overbought blocks LONG entries" && \
    echo "      📈 Trade WITH the momentum, not AGAINST it"

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

# 🎯 CACHE BUST MARKER: v5.0.5 - RSI Direction Filter
# Current deployment: 20251205_V505_RSI_DIRECTION_FILTER
# Changes: Prevent counter-trend entries (APT SHORT at RSI 19 = LOSS)
#   🛡️ v5.0.5: RSI Direction Filter
#      ✅ RSI < 30 (oversold) → Block SHORT entries
#      ✅ RSI > 70 (overbought) → Block LONG entries
#      ✅ Trade WITH momentum, not AGAINST it
#   📊 Previous versions:
#      ✅ v5.0.4b: Quality Filter Fix (Tech Advanced 40%→10%)
#      ✅ v5.0.4: 2-of-3 Confirmation Fix
#      ✅ v5.0.3: Closed Candle + Stop Hunt Detection
#      ✅ v5.0.2: Partial TP + Breakeven System
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
