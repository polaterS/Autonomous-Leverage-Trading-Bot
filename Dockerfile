# 🎯 LEVEL-BASED TRADING v5.0.8 - Level Proximity Re-Validation
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V508_LEVEL_PROXIMITY_REVALIDATION
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.8: LEVEL PROXIMITY RE-VALIDATION!" && \
    echo "   🛡️ CRITICAL FIX: Re-check S/R level proximity at execution!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.8 FEATURES:" && \
    echo "      ✅ Level proximity re-validation at execution time" && \
    echo "      ✅ Prevents trade when price drifts from S/R level" && \
    echo "      ✅ 0.5% threshold enforced at BOTH scan AND execution" && \
    echo "      ✅ Telegram notification when trade skipped" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ BUG FIXED:" && \
    echo "      ❌ OLD: BCH SHORT at 0.75% from resistance (scan=0.4%)" && \
    echo "      ✅ NEW: Check proximity at execution, skip if drifted" && \
    echo "      📈 No more mid-range entries after price drift"

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

# 🎯 CACHE BUST MARKER: v5.0.8 - Level Proximity Re-Validation
# Current deployment: 20251205_V508_LEVEL_PROXIMITY_REVALIDATION
# Changes: Re-check S/R level proximity before trade execution
#   🛡️ v5.0.8: Level Proximity Re-Validation
#      ✅ Check distance to S/R level at execution time
#      ✅ Skip trade if price drifted away (>0.5% from level)
#      ✅ Telegram notification for skipped trades
#   📊 Previous versions:
#      ✅ v5.0.7: Level Side Check Fix
#      ✅ v5.0.6: Skip Balance Check (live = paper behavior)
#      ✅ v5.0.5: RSI Direction Filter (block counter-trend)
#      ✅ v5.0.4b: Quality Filter Fix
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
