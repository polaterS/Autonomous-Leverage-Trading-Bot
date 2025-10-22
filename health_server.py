"""
Simple health check HTTP server.
Runs alongside the trading bot to provide health status.
"""

from fastapi import FastAPI
from datetime import datetime
import asyncio

app = FastAPI()

startup_time = datetime.now()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - startup_time).total_seconds()

    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Autonomous Leverage Trading Bot",
        "status": "running",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
