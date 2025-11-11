import asyncio
import asyncpg
import os
from datetime import datetime, timedelta

async def deep_analysis():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])

    print('=' * 80)
    print('🔍 DEEP SYSTEM ANALYSIS - Trading Bot Performance')
    print('=' * 80)

    # 1. OVERALL STATISTICS
    query1 = '''
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
            AVG(realized_pnl_usd) as avg_pnl,
            MAX(realized_pnl_usd) as max_win,
            MIN(realized_pnl_usd) as min_loss,
            SUM(realized_pnl_usd) as total_pnl,
            AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/60) as avg_duration_minutes
        FROM trade_history
    '''
    stats = await conn.fetchrow(query1)

    total = stats['total_trades']
    wins = stats['wins']
    losses = stats['losses']
    win_rate = (wins / total * 100) if total > 0 else 0

    print('\n📊 OVERALL PERFORMANCE:')
    print(f'Total Trades: {total}')
    print(f'Wins: {wins} | Losses: {losses}')
    print(f'Win Rate: {win_rate:.1f}%')
    print(f'Average P&L: ${float(stats["avg_pnl"]):.2f}')
    print(f'Best Win: ${float(stats["max_win"]):.2f}')
    print(f'Worst Loss: ${float(stats["min_loss"]):.2f}')
    print(f'Total P&L: ${float(stats["total_pnl"]):.2f}')
    print(f'Avg Duration: {float(stats["avg_duration_minutes"]):.1f} minutes')

    # 2. SIDE PERFORMANCE (LONG vs SHORT)
    query2 = '''
        SELECT
            side,
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            AVG(realized_pnl_usd) as avg_pnl,
            AVG(confidence) as avg_confidence
        FROM trade_history
        GROUP BY side
    '''
    sides = await conn.fetch(query2)

    print('\n📈 PERFORMANCE BY SIDE:')
    for row in sides:
        side_total = row['total']
        side_wins = row['wins']
        side_wr = (side_wins / side_total * 100) if side_total > 0 else 0
        print(f'{row["side"]:5} | Trades: {side_total:4} | WR: {side_wr:5.1f}% | Avg P&L: ${float(row["avg_pnl"]):+.2f} | Avg Conf: {float(row["avg_confidence"]):.1f}%')

    # 3. TOP PERFORMING SYMBOLS
    query3 = '''
        SELECT
            symbol,
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(realized_pnl_usd) as total_pnl,
            AVG(confidence) as avg_confidence
        FROM trade_history
        GROUP BY symbol
        HAVING COUNT(*) >= 5
        ORDER BY SUM(realized_pnl_usd) DESC
        LIMIT 10
    '''
    symbols = await conn.fetch(query3)

    print('\n🏆 TOP 10 PROFITABLE SYMBOLS (min 5 trades):')
    for i, row in enumerate(symbols, 1):
        sym_total = row['total']
        sym_wins = row['wins']
        sym_wr = (sym_wins / sym_total * 100) if sym_total > 0 else 0
        print(f'{i:2}. {row["symbol"]:20} | {sym_total:3} trades | WR: {sym_wr:5.1f}% | P&L: ${float(row["total_pnl"]):+7.2f} | Conf: {float(row["avg_confidence"]):.1f}%')

    # 4. WORST PERFORMING SYMBOLS
    query4 = '''
        SELECT
            symbol,
            COUNT(*) as total,
            SUM(realized_pnl_usd) as total_pnl
        FROM trade_history
        GROUP BY symbol
        HAVING COUNT(*) >= 5
        ORDER BY SUM(realized_pnl_usd) ASC
        LIMIT 5
    '''
    worst = await conn.fetch(query4)

    print('\n💔 WORST 5 SYMBOLS (min 5 trades):')
    for i, row in enumerate(worst, 1):
        print(f'{i}. {row["symbol"]:20} | {row["total"]:3} trades | P&L: ${float(row["total_pnl"]):+7.2f}')

    # 5. CLOSE REASONS
    query5 = '''
        SELECT
            close_reason,
            COUNT(*) as count,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            AVG(realized_pnl_usd) as avg_pnl
        FROM trade_history
        GROUP BY close_reason
        ORDER BY count DESC
    '''
    reasons = await conn.fetch(query5)

    print('\n🎯 EXIT REASONS:')
    for row in reasons:
        reason_wins = row['wins']
        reason_total = row['count']
        reason_wr = (reason_wins / reason_total * 100) if reason_total > 0 else 0
        print(f'{row["close_reason"]:40} | {reason_total:4} trades | WR: {reason_wr:5.1f}% | Avg: ${float(row["avg_pnl"]):+.2f}')

    # 6. LEVERAGE USAGE
    query6 = '''
        SELECT
            leverage,
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            AVG(realized_pnl_usd) as avg_pnl
        FROM trade_history
        GROUP BY leverage
        ORDER BY leverage
    '''
    leverages = await conn.fetch(query6)

    print('\n⚡ LEVERAGE PERFORMANCE:')
    for row in leverages:
        lev_wins = row['wins']
        lev_total = row['total']
        lev_wr = (lev_wins / lev_total * 100) if lev_total > 0 else 0
        print(f'{row["leverage"]:2}x | {lev_total:4} trades | WR: {lev_wr:5.1f}% | Avg P&L: ${float(row["avg_pnl"]):+.2f}')

    # 7. ML MODEL USAGE
    query7 = '''
        SELECT
            ai_model,
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            AVG(realized_pnl_usd) as avg_pnl,
            AVG(confidence) as avg_confidence
        FROM trade_history
        WHERE ai_model IS NOT NULL
        GROUP BY ai_model
        ORDER BY total DESC
    '''
    models = await conn.fetch(query7)

    print('\n🤖 ML MODEL PERFORMANCE:')
    for row in models:
        mod_wins = row['wins']
        mod_total = row['total']
        mod_wr = (mod_wins / mod_total * 100) if mod_total > 0 else 0
        print(f'{row["ai_model"]:30} | {mod_total:4} trades | WR: {mod_wr:5.1f}% | Avg: ${float(row["avg_pnl"]):+.2f} | Conf: {float(row["avg_confidence"]):.1f}%')

    # 8. RECENT TREND (last 7 days)
    query8 = '''
        SELECT
            DATE(exit_time) as date,
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(realized_pnl_usd) as daily_pnl
        FROM trade_history
        WHERE exit_time >= NOW() - INTERVAL '7 days'
        GROUP BY DATE(exit_time)
        ORDER BY date DESC
    '''
    recent = await conn.fetch(query8)

    print('\n📅 LAST 7 DAYS TREND:')
    for row in recent:
        day_wins = row['wins']
        day_total = row['total']
        day_wr = (day_wins / day_total * 100) if day_total > 0 else 0
        print(f'{row["date"]} | {day_total:3} trades | WR: {day_wr:5.1f}% | P&L: ${float(row["daily_pnl"]):+7.2f}')

    await conn.close()

asyncio.run(deep_analysis())
