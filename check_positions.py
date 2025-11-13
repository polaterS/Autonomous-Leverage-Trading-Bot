import asyncio
import asyncpg
import os

async def check_positions():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])

    print('='*70)
    print('DATABASE KONTROL - AKTIF POZISYONLAR')
    print('='*70)
    print()

    # Aktif pozisyonlar (table name: active_position, singular!)
    positions = await conn.fetch('SELECT * FROM active_position ORDER BY entry_time DESC')

    print(f'Aktif pozisyon sayisi: {len(positions)}')
    print()

    if positions:
        print('AKTIF POZISYONLAR:')
        print('-'*70)
        total_pnl = 0
        for i, p in enumerate(positions, 1):
            symbol = p['symbol']
            side = p['side']
            entry = float(p['entry_price'])
            current = float(p['current_price']) if p['current_price'] else 0
            pnl = float(p['unrealized_pnl_usd'] or 0)
            total_pnl += pnl
            time = p['entry_time'].strftime('%H:%M:%S')
            print(f'{i}. {time} {symbol:20} {side:5} Entry: ${entry:8.4f} Current: ${current:8.4f} P&L: ${pnl:7.2f}')
        print()
        print(f'TOPLAM UNREALIZED P&L: ${total_pnl:.2f}')
        print()
    else:
        print('HICH AKTIF POZISYON YOK!')
        print()

    # Bugun kapanan
    closed = await conn.fetch('''
        SELECT symbol, side, realized_pnl_usd, close_reason, exit_time
        FROM trade_history
        WHERE exit_time >= CURRENT_DATE
        ORDER BY exit_time DESC
    ''')

    print(f'Bugun kapanan trade sayisi: {len(closed)}')
    print()

    if closed:
        print('BUGUN KAPANAN TRADELER:')
        print('-'*70)
        total_realized = 0
        for t in closed:
            symbol = t['symbol']
            side = t['side']
            pnl = float(t['realized_pnl_usd'])
            total_realized += pnl
            reason = t['close_reason']
            time = t['exit_time'].strftime('%H:%M:%S')
            print(f'{time} {symbol:20} {side:5} ${pnl:7.2f} ({reason})')
        print()
        print(f'BUGUN TOPLAM REALIZED P&L: ${total_realized:.2f}')

    await conn.close()

if __name__ == '__main__':
    asyncio.run(check_positions())
