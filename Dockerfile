# 🛡️ LEVEL-BASED TRADING v5.0.13 - Critical Protection Filters
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251208_V5013_CRITICAL_PROTECTION
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🛡️ v5.0.13: CRITICAL PROTECTION FILTERS!" && \
    echo "   🔥 FIX: 6 consecutive losing trades!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.13 FEATURES:" && \
    echo "      ✅ TREND DIRECTION FILTER (no counter-trend!)" && \
    echo "      ✅ ADX MOMENTUM FILTER (skip if ADX > 50)" && \
    echo "      ✅ ALL 3 confirmations required (was 2/3)" && \
    echo "      ✅ Tighter stop-loss: 0.8% = ~$8 max loss" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v5.0.12: Bulletproof Exit Price" && \
    echo "      ✅ v5.0.11: Reliable Exit Price" && \
    echo "      ✅ v5.0.10: Async Portfolio Updates"

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

# 🛡️ CACHE BUST MARKER: v5.0.13 - Critical Protection Filters
# Current deployment: 20251208_V5013_CRITICAL_PROTECTION
# Changes: 6 consecutive losing trades - need protection filters!
#   🛡️ v5.0.13: Critical Protection Filters
#      ✅ TREND DIRECTION FILTER (no counter-trend trades!)
#      ✅ ADX MOMENTUM FILTER (skip if ADX > 50)
#      ✅ ALL 3 confirmations required (was 2/3)
#      ✅ Tighter stop-loss: 0.8% = ~$8 max loss
#   📊 Previous versions:
#      ✅ v5.0.12: Bulletproof Exit Price
#      ✅ v5.0.11: Reliable Exit Price
#      ✅ v5.0.10: Async Portfolio Updates
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
