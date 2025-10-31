"""
Simple chart server to serve interactive HTML charts.
Stores charts on filesystem and serves via FastAPI endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import json
from src.utils import setup_logging

logger = setup_logging()

# Filesystem storage for charts
CHARTS_DIR = Path("/app/data/charts") if Path("/app/data").exists() else Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for metadata
_charts_storage: Dict[str, Dict[str, any]] = {}

# FastAPI app instance
chart_app = FastAPI()


def store_chart(html_content: str, symbol: str) -> str:
    """
    Store chart HTML and return unique chart ID.

    Args:
        html_content: HTML content of the chart
        symbol: Trading symbol

    Returns:
        Unique chart ID
    """
    chart_id = str(uuid.uuid4())[:8]  # Short ID

    # Store HTML to filesystem
    html_file = CHARTS_DIR / f"{chart_id}.html"
    html_file.write_text(html_content, encoding='utf-8')

    # Store metadata
    metadata = {
        'symbol': symbol,
        'created_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
    }
    metadata_file = CHARTS_DIR / f"{chart_id}.json"
    metadata_file.write_text(json.dumps(metadata), encoding='utf-8')

    # Cache in memory
    _charts_storage[chart_id] = {
        'symbol': symbol,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(hours=24)
    }

    # Clean up expired charts
    _cleanup_expired_charts()

    logger.info(f"📊 Stored chart {chart_id} for {symbol} (filesystem)")
    return chart_id


def _cleanup_expired_charts():
    """Remove expired charts from storage."""
    now = datetime.now()

    # Clean from filesystem
    for html_file in CHARTS_DIR.glob("*.html"):
        chart_id = html_file.stem
        metadata_file = CHARTS_DIR / f"{chart_id}.json"

        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                expires_at = datetime.fromisoformat(metadata['expires_at'])
                if expires_at < now:
                    html_file.unlink()
                    metadata_file.unlink()
                    logger.debug(f"🗑️ Removed expired chart {chart_id}")
            except Exception as e:
                logger.error(f"Error cleaning up chart {chart_id}: {e}")

    # Clean from memory cache
    expired = [
        chart_id for chart_id, data in _charts_storage.items()
        if data['expires_at'] < now
    ]
    for chart_id in expired:
        del _charts_storage[chart_id]


@chart_app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chart Server API",
        "endpoints": {
            "/chart/{chart_id}": "Get interactive chart",
            "/charts": "List all charts"
        },
        "active_charts": len(_charts_storage)
    }


@chart_app.get("/chart/{chart_id}", response_class=HTMLResponse)
async def get_chart(chart_id: str):
    """
    Serve interactive chart HTML.

    Args:
        chart_id: Unique chart identifier

    Returns:
        HTML content of the chart
    """
    # Check filesystem
    html_file = CHARTS_DIR / f"{chart_id}.html"
    metadata_file = CHARTS_DIR / f"{chart_id}.json"

    if not html_file.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Chart not found or expired")

    # Read metadata
    try:
        metadata = json.loads(metadata_file.read_text())
        expires_at = datetime.fromisoformat(metadata['expires_at'])

        # Check if expired
        if expires_at < datetime.now():
            html_file.unlink()
            metadata_file.unlink()
            raise HTTPException(status_code=410, detail="Chart expired")

        # Read and serve HTML
        html_content = html_file.read_text(encoding='utf-8')
        logger.info(f"📊 Serving chart {chart_id} ({metadata['symbol']})")
        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error serving chart {chart_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@chart_app.get("/charts")
async def list_charts():
    """List all active charts."""
    _cleanup_expired_charts()

    # List from filesystem
    charts = []
    for metadata_file in CHARTS_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_file.read_text())
            chart_id = metadata_file.stem
            charts.append({
                "id": chart_id,
                "symbol": metadata['symbol'],
                "created_at": metadata['created_at'],
                "expires_at": metadata['expires_at']
            })
        except Exception as e:
            logger.error(f"Error reading chart metadata: {e}")

    return {
        "total_charts": len(charts),
        "charts": charts
    }


# Singleton chart app
_chart_app_instance = None


def get_chart_app() -> FastAPI:
    """Get FastAPI chart app instance."""
    global _chart_app_instance
    if _chart_app_instance is None:
        _chart_app_instance = chart_app
    return _chart_app_instance
