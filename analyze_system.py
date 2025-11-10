import asyncio
import asyncpg
import os
from decimal import Decimal

async def full_analysis():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])

    print('='*60)
    print('SISTEM ANALIZI - TAM RAPOR')
    print('='*60)

    # 1. SON 50 TRADE ANALIZI
    print('\n1. SON 50 TRADE:')
    query = '''
        SELECT side, realized_pnl_usd, close_reason
        FROM trade_history
        ORDER BY exit_time DESC
        LIMIT 50
    '''
    trades = await conn.fetch(query)

    long_win = sum(1 for t in trades if t['side']=='LONG' and float(t['realized_pnl_usd'])>0)
    long_loss = sum(1 for t in trades if t['side']=='LONG' and float(t['realized_pnl_usd'])<=0)
    short_win = sum(1 for t in trades if t['side']=='SHORT' and float(t['realized_pnl_usd'])>0)
    short_loss = sum(1 for t in trades if t['side']=='SHORT' and float(t['realized_pnl_usd'])<=0)

    total_long = long_win + long_loss
    total_short = short_win + short_loss

    long_wr = (long_win/total_long*100) if total_long>0 else 0
    short_wr = (short_win/total_short*100) if total_short>0 else 0

    print(f'  LONG:  {long_win}W/{long_loss}L = {long_wr:.1f}% WR ({total_long} trades)')
    print(f'  SHORT: {short_win}W/{short_loss}L = {short_wr:.1f}% WR ({total_short} trades)')

    # 2. ML EXIT PERFORMANSI
    print('\n2. ML EXIT SISTEMI:')
    ml_exits = sum(1 for t in trades if 'ML exit' in t['close_reason'])
    stop_losses = sum(1 for t in trades if 'Stop-loss' in t['close_reason'])
    take_profits = sum(1 for t in trades if 'Take-profit' in t['close_reason'])

    print(f'  ML Exit: {ml_exits} trades')
    print(f'  Stop-Loss: {stop_losses} trades')
    print(f'  Take-Profit: {take_profits} trades')

    # 3. ORTALAMA KAYIP ANALIZI
    print('\n3. KAYIP ANALIZI:')
    losses = [float(t['realized_pnl_usd']) for t in trades if float(t['realized_pnl_usd'])<0]
    if losses:
        avg_loss = sum(losses)/len(losses)
        max_loss = min(losses)
        print(f'  Ortalama kayip: ${avg_loss:.2f}')
        print(f'  Max kayip: ${max_loss:.2f}')
        print(f'  -$10 stop-loss hitleri: {sum(1 for l in losses if l<=-9.5)}')

    # 4. TRADE DAGILIMI
    print('\n4. TRADE DAGILIMI (Son 50):')
    long_pct = (total_long/50*100) if trades else 0
    short_pct = (total_short/50*100) if trades else 0
    print(f'  LONG: {long_pct:.0f}%')
    print(f'  SHORT: {short_pct:.0f}%')

    # 5. PATTERN PERFORMANCE
    print('\n5. PATTERN ANALIZI:')
    pattern_query = '''
        SELECT pattern_name, total_trades,
               wins
        FROM pattern_performance
        WHERE total_trades >= 3
        ORDER BY (CAST(wins AS FLOAT) / total_trades) DESC
        LIMIT 10
    '''
    patterns = await conn.fetch(pattern_query)
    if patterns:
        print('  En iyi 10 pattern:')
        for p in patterns:
            wr = (p['wins']/p['total_trades']*100) if p['total_trades']>0 else 0
            print(f'    {p["pattern_name"][:20]:20} {p["wins"]}W/{p["total_trades"]}T = {wr:.0f}%')
    else:
        print('  Pattern verisi yok')

    # 6. RISK YONETIMI
    print('\n6. RISK YONETIMI:')
    config = await conn.fetchrow('SELECT * FROM trading_config LIMIT 1')
    if config:
        print(f'  Sermaye: ${float(config["current_capital"]):.2f}')
        print(f'  Max positions: {int(float(config["current_capital"])/100)}')
        print(f'  Daily max loss: ${float(config["daily_max_loss_usd"]):.2f}')

    # 7. AI CACHE DURUMU
    print('\n7. ML MODEL CACHE:')
    cache_query = '''
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active
        FROM ai_analysis_cache
    '''
    cache = await conn.fetchrow(cache_query)
    if cache:
        print(f'  Total cache entries: {cache["total"]}')
        print(f'  Active (not expired): {cache["active"]}')

    await conn.close()
    print('\n' + '='*60)

asyncio.run(full_analysis())
