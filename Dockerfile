# 🎯 LEVEL-BASED TRADING v5.0.9 - Trendline Price Position Validation
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251205_V509_TRENDLINE_VALIDATION
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🎯 v5.0.9: TRENDLINE PRICE POSITION VALIDATION!" && \
    echo "   🛡️ CRITICAL FIX: Validate price position vs trendline!" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🆕 v5.0.9 FEATURES:" && \
    echo "      ✅ Trendline price position validation" && \
    echo "      ✅ Ascending: Price must be AT or ABOVE line" && \
    echo "      ✅ Descending: Price must be AT or BELOW line" && \
    echo "      ✅ Broken trendlines now skipped" && \
    echo "   ═══════════════════════════════════════════════════" && \
    echo "   🛡️ BUG FIXED:" && \
    echo "      ❌ OLD: BLUR LONG when price BELOW ascending line" && \
    echo "      ✅ NEW: Skip broken trendlines, prevent wrong entries" && \
    echo "      📈 Trade logic now matches /analyze classification"

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

# 🎯 CACHE BUST MARKER: v5.0.9 - Trendline Price Position Validation
# Current deployment: 20251205_V509_TRENDLINE_VALIDATION
# Changes: Validate price position relative to trendline before classifying
#   🛡️ v5.0.9: Trendline Price Position Validation
#      ✅ Ascending line: Price must be AT or ABOVE (not below)
#      ✅ Descending line: Price must be AT or BELOW (not above)
#      ✅ Broken trendlines are now skipped
#   📊 Previous versions:
#      ✅ v5.0.8: Level Proximity Re-Validation
#      ✅ v5.0.7: Level Side Check Fix
#      ✅ v5.0.6: Skip Balance Check
#      ✅ v5.0.5: RSI Direction Filter
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
