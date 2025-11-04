"""
Database maintenance utilities
One-time cleanup functions for fixing database issues
"""

import asyncio
from decimal import Decimal
from datetime import date
from src.database import get_db_client
from src.utils import setup_logging

logger = setup_logging()


async def cleanup_duplicate_configs():
    """
    Fix duplicate trading_config rows
    Keep only id=1 and update capital from daily_performance
    """
    logger.info("=" * 60)
    logger.info("DATABASE MAINTENANCE: Cleaning duplicate configs")
    logger.info("=" * 60)

    db = await get_db_client()

    try:
        # Step 1: Count current configs
        async with db.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM trading_config")
            logger.info(f"Current trading_config rows: {count}")

            if count <= 1:
                logger.info("No duplicates found! Database is clean.")
                return

            # Step 2: Show all configs
            rows = await conn.fetch("""
                SELECT id, current_capital, last_updated
                FROM trading_config
                ORDER BY id
            """)

            logger.info("\nCurrent configs:")
            for row in rows:
                logger.info(f"  ID: {row['id']:3d} | Capital: ${float(row['current_capital']):12.2f}")

            # Step 3: Delete duplicates (keep only id=1)
            logger.info("\nDeleting duplicate configs (keeping id=1)...")
            delete_result = await conn.execute("DELETE FROM trading_config WHERE id != 1")
            deleted_count = int(delete_result.split()[-1])
            logger.info(f"Deleted {deleted_count} duplicate rows")

            # Step 4: Get correct capital from daily_performance
            logger.info("\nFetching correct capital from daily_performance...")
            today = date.today()
            daily_perf = await conn.fetchrow("""
                SELECT ending_capital
                FROM daily_performance
                WHERE date = $1
                LIMIT 1
            """, today)

            if daily_perf:
                correct_capital = daily_perf['ending_capital']
                logger.info(f"Today's ending capital: ${float(correct_capital):.2f}")

                # Update trading_config
                logger.info("Updating trading_config capital...")
                await conn.execute("""
                    UPDATE trading_config
                    SET current_capital = $1,
                        last_updated = NOW()
                    WHERE id = 1
                """, correct_capital)

                logger.info(f"Capital updated to ${float(correct_capital):.2f}")
            else:
                logger.warning("No daily_performance for today, keeping current capital")

            # Step 5: Verify final state
            final_config = await conn.fetchrow("SELECT * FROM trading_config WHERE id = 1")

            logger.info("\n" + "=" * 60)
            logger.info("CLEANUP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("\nFinal config:")
            logger.info(f"  ID: {final_config['id']}")
            logger.info(f"  Initial Capital: ${float(final_config['initial_capital']):.2f}")
            logger.info(f"  Current Capital: ${float(final_config['current_capital']):.2f}")
            logger.info(f"  Last Updated: {final_config['last_updated']}")
            logger.info(f"  Trading Enabled: {final_config['is_trading_enabled']}")

            return True

    except Exception as e:
        logger.error(f"Database cleanup failed: {e}", exc_info=True)
        return False


async def verify_database_health():
    """
    Verify database is in good state
    Returns dict with health status
    """
    logger.info("Checking database health...")

    db = await get_db_client()
    health = {
        'config_count': 0,
        'active_positions': 0,
        'trade_history_count': 0,
        'daily_performance_exists': False,
        'capital_synced': False,
        'issues': []
    }

    try:
        async with db.pool.acquire() as conn:
            # Check config count
            health['config_count'] = await conn.fetchval("SELECT COUNT(*) FROM trading_config")
            if health['config_count'] > 1:
                health['issues'].append(f"Multiple config rows: {health['config_count']}")

            # Check active positions
            health['active_positions'] = await conn.fetchval("SELECT COUNT(*) FROM active_position")

            # Check trade history
            health['trade_history_count'] = await conn.fetchval("SELECT COUNT(*) FROM trade_history")

            # Check daily performance
            today = date.today()
            daily_perf = await conn.fetchrow("""
                SELECT ending_capital
                FROM daily_performance
                WHERE date = $1
            """, today)
            health['daily_performance_exists'] = daily_perf is not None

            # Check capital sync
            if daily_perf:
                config = await conn.fetchrow("SELECT current_capital FROM trading_config WHERE id = 1")
                if config:
                    capital_diff = abs(float(daily_perf['ending_capital']) - float(config['current_capital']))
                    health['capital_synced'] = capital_diff < 0.01  # Within 1 cent

                    if not health['capital_synced']:
                        health['issues'].append(
                            f"Capital mismatch: config=${float(config['current_capital']):.2f}, "
                            f"daily=${float(daily_perf['ending_capital']):.2f}"
                        )

        # Log health status
        logger.info("=" * 60)
        logger.info("DATABASE HEALTH CHECK")
        logger.info("=" * 60)
        logger.info(f"Config rows: {health['config_count']}")
        logger.info(f"Active positions: {health['active_positions']}")
        logger.info(f"Trade history: {health['trade_history_count']}")
        logger.info(f"Daily performance: {'Yes' if health['daily_performance_exists'] else 'No'}")
        logger.info(f"Capital synced: {'Yes' if health['capital_synced'] else 'No'}")

        if health['issues']:
            logger.warning("\nISSUES DETECTED:")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\nNo issues detected!")

        logger.info("=" * 60)

        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        health['issues'].append(f"Health check error: {e}")
        return health


if __name__ == "__main__":
    # For manual testing
    async def main():
        # Run health check
        health = await verify_database_health()

        # If issues detected, offer cleanup
        if health['issues']:
            print("\nIssues detected. Running cleanup...")
            await cleanup_duplicate_configs()

            # Verify again
            print("\nVerifying after cleanup...")
            await verify_database_health()

    asyncio.run(main())
