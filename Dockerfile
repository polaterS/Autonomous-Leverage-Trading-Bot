# 🎯 LEVEL-BASED TRADING v5.0.1 - /analyze BTC Command
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V501_ANALYZE_COMMAND
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.1: /analyze BTC TELEGRAM COMMAND!" && \
    echo "   📱 NEW: Enhanced Telegram Analysis Command" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.1 FEATURES:" && \
    echo "      ✅ /analyze BTC - Show S/R levels, volume, RSI" && \
    echo "      ✅ All 5 timeframe S/R levels displayed" && \
    echo "      ✅ Trend line count shown" && \
    echo "      ✅ Volume spike status (1.5x threshold)" && \
    echo "      ✅ RSI zone (oversold/overbought)" && \
    echo "      ✅ Candlestick patterns detected" && \
    echo "      ✅ Entry confirmation checklist" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 ANALYZE COMMAND SHOWS:" && \
    echo "      • Top 5 Support levels (distance %)" && \
    echo "      • Top 5 Resistance levels (distance %)" && \
    echo "      • Trend lines (ascending/descending)" && \
    echo "      • Volume: Current vs 1.5x threshold" && \
    echo "      • RSI: Value and zone" && \
    echo "      • Entry status: At level or waiting" && \
    echo "      • Confirmation checklist: ✅/❌"

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

# 🎯 CACHE BUST MARKER: v5.0.1 - /analyze BTC Command
# Current deployment: 20251205_V501_ANALYZE_COMMAND
# Changes: Enhanced /analyze Telegram command for Level-Based System
#   📱 v5.0.1: /analyze BTC Command
#      ✅ /analyze BTC - Show all S/R levels from 5 timeframes
#      ✅ Volume spike detection (1.5x threshold)
#      ✅ RSI zone display (oversold/overbought)
#      ✅ Candlestick pattern detection
#      ✅ Entry confirmation checklist
#      ✅ Trend line count display
#   📊 Previous versions:
#      ✅ v5.0.0: Level-Based Trading System (complete redesign)
#      ✅ v4.7.13: ADX filter source fix
#      ✅ v4.7.12: Derivatives filter removed
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
