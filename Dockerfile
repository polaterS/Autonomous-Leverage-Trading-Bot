# 🔧 v6.5 PA-ONLY + News Sentiment + Dynamic TP
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251226_V65_NEWS_DYNAMIC_TP
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🔧 v6.5: NEWS SENTIMENT + DYNAMIC TP!" && \
    echo "   📊 Major Updates:" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v6.5 FEATURES:" && \
    echo "      🗞️ /news command - Crypto news sentiment analysis" && \
    echo "      🎯 Dynamic TP based on S/R levels" && \
    echo "      🔧 Fixed hardcoded 20% stop-loss bug" && \
    echo "      📊 Now uses config: MAX_STOP_LOSS_PERCENT" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v6.4: Professional Entry at S/R levels" && \
    echo "      ✅ v6.3: Multi-TF S/R analysis"

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

# 🔧 CACHE BUST MARKER: v5.0.17 - Indicator Fix
# Current deployment: 20251209_V5017_INDICATOR_FIX
# Changes: CRITICAL fix - Level-Based Trading indicators were MISSING!
#   🔧 v5.0.17: Indicator Fix
#      🔧 Added ema_20, ema_50 (trend direction filter was BROKEN!)
#      🔧 Added adx, plus_di, minus_di (trend strength filter BROKEN!)
#      🔧 /analyze now shows ADX + EMA values for debugging
#   📊 Previous versions:
#      ✅ v5.0.16: Low Liquidity Blacklist
#      ✅ v5.0.15: Balanced Settings
#      ✅ v5.0.14: Ghost Exit Price Fix
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
