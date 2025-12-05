# 🎯 LEVEL-BASED TRADING v5.0.7 - Level Side Check Fix
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V507_LEVEL_SIDE_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.7: LEVEL SIDE CHECK FIX!" && \
    echo "   🛡️ CRITICAL FIX: Check correct side of S/R level!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.7 FEATURES:" && \
    echo "      ✅ Level side validation added" && \
    echo "      ✅ Resistance: Price must be AT or BELOW (not above)" && \
    echo "      ✅ Support: Price must be AT or ABOVE (not below)" && \
    echo "      ✅ Fixed confirmation display (correct ✓/✗ marks)" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ WHY THIS MATTERS:" && \
    echo "      ❌ OLD: BCH SHORT at broken resistance (price above)" && \
    echo "      ✅ NEW: Only trade when price is ON CORRECT SIDE" && \
    echo "      ✅ No more SHORT when resistance already broken" && \
    echo "      📈 Prevents wrong-side entries like BCH $578>$576"

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

# 🎯 CACHE BUST MARKER: v5.0.7 - Level Side Check Fix
# Current deployment: 20251205_V507_LEVEL_SIDE_FIX
# Changes: Check correct side of S/R level before entry
#   🛡️ v5.0.7: Level Side Check Fix
#      ✅ Resistance: Price must be AT or BELOW (not above)
#      ✅ Support: Price must be AT or ABOVE (not below)
#      ✅ Fixed confirmation display (correct ✓/✗ marks)
#   📊 Previous versions:
#      ✅ v5.0.6: Skip Balance Check (live = paper behavior)
#      ✅ v5.0.5: RSI Direction Filter (block counter-trend)
#      ✅ v5.0.4b: Quality Filter Fix (Tech Advanced 40%→10%)
#      ✅ v5.0.4: 2-of-3 Confirmation Fix
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
