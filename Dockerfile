# 🔧 PA-ONLY v4.7.9 - CONFLUENCE SCORING FIX
FROM python:3.11-slim

# Cache bust argument to force rebuild when needed
ARG CACHE_BUST=20251204_V479_CONFLUENCE_FIX
RUN echo "🔥🔥🔥 CACHE BUST: ${CACHE_BUST}" && \
    echo "Build timestamp: $(date)" && \
    echo "🔧 v4.7.9: CONFLUENCE SCORING FIX!" && \
    echo "   🔧 BUG FIX:" && \
    echo "      ✅ market_scanner.py now calculates ALL enhanced indicators" && \
    echo "      ✅ Passes enhanced_data, advanced_data, institutional_data" && \
    echo "      ✅ Passes derivatives_data, technical_advanced_data, harmonic_data" && \
    echo "      ✅ Categories no longer fallback to 60% defaults" && \
    echo "   🛡️ Previous fixes:" && \
    echo "      ✅ v4.7.8: Protection filters" && \
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

# 🔧 CACHE BUST MARKER: PA-ONLY v4.7.9
# Current deployment: 20251204_V479_CONFLUENCE_FIX
# Changes: Fixed confluence scoring in market_scanner.py
#   🔧 v4.7.9 CONFLUENCE SCORING FIX:
#      ✅ BUG: score_opportunity() was called WITHOUT enhanced indicator params
#      ✅ RESULT: All categories showed 60% fallback instead of real values
#      ✅ FIX: Now calculates and passes ALL enhanced indicator parameters
#      ✅ Categories: enhanced_data, advanced_data, institutional_data (v4.4-v4.6)
#      ✅ Categories: derivatives_data, technical_advanced_data, harmonic_data (v4.7.0)
#   🛡️ Previous fixes:
#      ✅ v4.7.8: Three protection filters
#      ✅ v4.7.7: Order book methods
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
