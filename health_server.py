"""
Comprehensive health check HTTP server.
Provides detailed system status, database health, and trading metrics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, date
from decimal import Decimal
import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Trading Bot Health API", version="1.0.0")

startup_time = datetime.now()


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns detailed status of all bot components.
    """
    try:
        from src.database import get_db_client
        from src.config import get_settings

        settings = get_settings()
        uptime_seconds = (datetime.now() - startup_time).total_seconds()

        # Database health check
        db_healthy = False
        db_error = None
        try:
            db = await get_db_client()
            await db.get_trading_config()
            db_healthy = True
        except Exception as e:
            db_error = str(e)

        # Build response
        health_status = {
            "status": "healthy" if db_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(uptime_seconds),
            "uptime_formatted": format_uptime(uptime_seconds),
            "components": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "error": db_error
                },
                "paper_trading": settings.use_paper_trading,
                "ai_models": {
                    "qwen3_max": bool(settings.qwen_api_key),
                    "deepseek_v32": bool(settings.deepseek_api_key)
                }
            }
        }

        if not db_healthy:
            return JSONResponse(status_code=503, content=health_status)

        return health_status

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/status")
async def trading_status():
    """
    Get detailed trading bot status including positions and performance.
    """
    try:
        from src.database import get_db_client

        db = await get_db_client()

        # Get current config
        config = await db.get_trading_config()

        # Get active position
        active_position = await db.get_active_position()

        # Get today's performance
        today_pnl = await db.get_daily_pnl(date.today())

        # Get recent trades
        recent_trades = await db.get_recent_trades(limit=5)

        return {
            "status": "trading" if active_position else "scanning",
            "trading_enabled": config.get('is_trading_enabled', False),
            "capital": {
                "current": float(config.get('current_capital', 0)),
                "initial": float(config.get('initial_capital', 0))
            },
            "daily_pnl": float(today_pnl),
            "active_position": {
                "symbol": active_position['symbol'],
                "side": active_position['side'],
                "leverage": active_position['leverage'],
                "entry_price": float(active_position['entry_price']),
                "current_pnl": float(active_position.get('unrealized_pnl_usd', 0))
            } if active_position else None,
            "recent_trades": len(recent_trades),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get trading performance metrics.
    """
    try:
        from src.database import get_db_client

        db = await get_db_client()

        # Get recent trades for stats
        recent_trades = await db.get_recent_trades(limit=50)

        if not recent_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "message": "No trades executed yet"
            }

        total_trades = len(recent_trades)
        winners = sum(1 for t in recent_trades if t['is_winner'])
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(Decimal(str(t['realized_pnl_usd'])) for t in recent_trades)
        avg_pnl = float(total_pnl / total_trades)

        return {
            "total_trades": total_trades,
            "winning_trades": winners,
            "losing_trades": total_trades - winners,
            "win_rate": round(win_rate, 2),
            "total_pnl": float(total_pnl),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Autonomous Leverage Trading Bot",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/health": "System health check",
            "/status": "Trading bot status",
            "/metrics": "Performance metrics"
        },
        "uptime": format_uptime((datetime.now() - startup_time).total_seconds())
    }


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
