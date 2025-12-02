# 🚀 PA-ONLY v4.7.1 - ULTRA PROFESSIONAL ANALYSIS (FIXED)
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251202_V471_ULTRA_PRO_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🚀 v4.7.0: ULTRA PROFESSIONAL ANALYSIS!" && \
    echo "   🆕 TIER 1 - Derivatives Analysis:" && \
    echo "      💰 Funding Rate (contrarian sentiment)" && \
    echo "      📊 Open Interest (trend strength)" && \
    echo "      📈 Long/Short Ratio (crowd positioning)" && \
    echo "      😱 Fear & Greed Index (market sentiment)" && \
    echo "   🆕 TIER 2 - Advanced Technical:" && \
    echo "      📉 CVD - Cumulative Volume Delta (buy/sell pressure)" && \
    echo "      ⛩️ Ichimoku Cloud (complete trading system)" && \
    echo "      🔥 Liquidation Levels (liquidity hunting zones)" && \
    echo "   🆕 TIER 3 - Harmonic Patterns:" && \
    echo "      🦋 Gartley, Butterfly, Bat, Crab patterns" && \
    echo "      🎯 PRZ (Potential Reversal Zone) detection" && \
    echo "   ✅ Scoring: 100 points (13 categories)" && \
    echo "   - Instant Trading still DISABLED"

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

# 🔥 CACHE BUST MARKER: PA-ONLY v4.7.0
# Current deployment: 20251202_V470_ULTRA_PROFESSIONAL
# Changes: Ultra Professional Analysis System
#   🆕 TIER 1 - Derivatives Analysis:
#      💰 Funding Rate (contrarian sentiment indicator)
#      📊 Open Interest (trend strength confirmation)
#      📈 Long/Short Ratio (crowd positioning)
#      😱 Fear & Greed Index (composite market sentiment)
#   🆕 TIER 2 - Advanced Technical:
#      📉 CVD - Cumulative Volume Delta (buy/sell pressure divergence)
#      ⛩️ Ichimoku Cloud (complete Japanese trading system)
#      🔥 Liquidation Levels (liquidity hunting zones estimation)
#   🆕 TIER 3 - Harmonic Patterns:
#      🦋 Gartley, Butterfly, Bat, Crab pattern detection
#      🎯 PRZ (Potential Reversal Zone) identification
#   ✅ Scoring: 100 points across 13 categories
#   🏛️ All previous indicators (v4.4, v4.5, v4.6) still active
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
