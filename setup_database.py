"""
Database setup script for the Autonomous Leverage Trading Bot.
Creates all necessary tables and initializes configuration.
"""

import asyncio
import asyncpg
from src.config import get_settings
from pathlib import Path


async def setup_database():
    """Create database tables and initial configuration."""
    settings = get_settings()

    print("Connecting to database...")

    try:
        # Connect to database
        conn = await asyncpg.connect(settings.database_url)

        print("Connected successfully!")
        print("Creating tables...")

        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        await conn.execute(schema_sql)

        print("Tables created successfully!")

        # Update initial configuration with environment values
        await conn.execute("""
            UPDATE trading_config
            SET initial_capital = $1,
                current_capital = $1,
                position_size_percent = $2,
                min_stop_loss_percent = $3,
                max_stop_loss_percent = $4,
                min_profit_usd = $5,
                max_leverage = $6,
                min_ai_confidence = $7,
                daily_loss_limit_percent = $8,
                max_consecutive_losses = $9,
                last_updated = NOW()
            WHERE id = 1
        """,
            settings.initial_capital,
            settings.position_size_percent,
            settings.min_stop_loss_percent,
            settings.max_stop_loss_percent,
            settings.min_profit_usd,
            settings.max_leverage,
            settings.min_ai_confidence,
            settings.daily_loss_limit_percent,
            settings.max_consecutive_losses
        )

        print("Configuration updated successfully!")

        # Verify setup
        config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")
        print("\nInitial Trading Configuration:")
        print(f"  Initial Capital: ${config['initial_capital']}")
        print(f"  Max Leverage: {config['max_leverage']}x")
        print(f"  Position Size: {float(config['position_size_percent'])*100}%")
        print(f"  Stop Loss Range: {float(config['min_stop_loss_percent'])*100}% - {float(config['max_stop_loss_percent'])*100}%")
        print(f"  Min Profit: ${config['min_profit_usd']}")
        print(f"  Min AI Confidence: {float(config['min_ai_confidence'])*100}%")
        print(f"  Daily Loss Limit: {float(config['daily_loss_limit_percent'])*100}%")
        print(f"  Max Consecutive Losses: {config['max_consecutive_losses']}")

        await conn.close()
        print("\nDatabase setup complete!")

    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(setup_database())
