# 🎯 LEVEL-BASED TRADING v5.0.2 - Partial TP + Breakeven
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V502_PARTIAL_TP_BREAKEVEN
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.2: PARTIAL TP + BREAKEVEN SYSTEM!" && \
    echo "   💰 NEW: Level-Based Exit Strategy" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.2 FEATURES:" && \
    echo "      ✅ TP1: Close 50% at first S/R level" && \
    echo "      ✅ BREAKEVEN: Move stop to entry after TP1" && \
    echo "      ✅ TP2: Close remaining 50% at second S/R" && \
    echo "      ✅ Risk-free trade after TP1 hit!" && \
    echo "      ✅ Binance SL order auto-updated to breakeven" && \
    echo "      ✅ Database tracks breakeven_active status" && \
    echo "      ✅ Telegram notifications for each stage" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 EXECUTION STRATEGY:" && \
    echo "      1️⃣ Price hits TP1 → Close 50%" && \
    echo "      2️⃣ Stop moves to Entry (Breakeven)" && \
    echo "      3️⃣ Trade becomes RISK-FREE!" && \
    echo "      4️⃣ Price hits TP2 → Close remaining 50%" && \
    echo "      📈 Maximum profit potential with protected gains"

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

# 🎯 CACHE BUST MARKER: v5.0.2 - Partial TP + Breakeven
# Current deployment: 20251205_V502_PARTIAL_TP_BREAKEVEN
# Changes: Level-Based Exit Strategy with Partial TP + Breakeven
#   💰 v5.0.2: Partial TP + Breakeven System
#      ✅ TP1: Close 50% at first S/R level (profit_target_1)
#      ✅ BREAKEVEN: Move stop-loss to entry price after TP1
#      ✅ TP2: Close remaining 50% at second S/R level
#      ✅ Risk-free trading after TP1 hit!
#      ✅ Binance SL order auto-updated to breakeven
#      ✅ Database tracks breakeven_active status
#   📊 Previous versions:
#      ✅ v5.0.1: /analyze BTC Command
#      ✅ v5.0.0: Level-Based Trading System (complete redesign)
#      ✅ v4.7.13: ADX filter source fix
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
