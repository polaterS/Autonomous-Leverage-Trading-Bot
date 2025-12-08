# 🎯 LEVEL-BASED TRADING v5.0.15 - Balanced Settings
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251208_V5015_BALANCED_SETTINGS
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.15: BALANCED STOP-LOSS + 2/3 CONFIRMATION!" && \
    echo "   📊 Based on user feedback: 85% win rate before tight SL" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.15 FEATURES:" && \
    echo "      ✅ Stop-loss: 1.2% = ~\$12 max loss (was 0.8%)" && \
    echo "      ✅ 2/3 confirmations (was 3/3 - too strict!)" && \
    echo "      ✅ TREND + ADX filters still active!" && \
    echo "      ✅ + v5.0.14 ghost exit price fix" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   📊 Previous versions:" && \
    echo "      ✅ v5.0.14: Ghost Exit Price Fix" && \
    echo "      ✅ v5.0.13: Protection Filters"

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

# 🎯 CACHE BUST MARKER: v5.0.15 - Balanced Settings
# Current deployment: 20251208_V5015_BALANCED_SETTINGS
# Changes: User feedback - 85% win rate before tight stop-loss!
#   🎯 v5.0.15: Balanced Stop-Loss + 2/3 Confirmations
#      ✅ Stop-loss: 0.8% → 1.2% = ~$12 max loss (room to breathe!)
#      ✅ Confirmations: 3/3 → 2/3 (3/3 too strict to find trades)
#      ✅ TREND + ADX filters still active for quality entries
#      ✅ + v5.0.14 ghost exit price fix included
#   📊 Previous versions:
#      ✅ v5.0.14: Ghost Exit Price Fix
#      ✅ v5.0.13: Protection Filters
#      ✅ v5.0.12: Bulletproof Exit Price
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
