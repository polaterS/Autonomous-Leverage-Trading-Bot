"""
Quick test to check Railway database connectivity.
"""
import asyncpg
import asyncio
import ssl

async def test_connection():
    print("=" * 60)
    print("RAILWAY DATABASE CONNECTION TEST")
    print("=" * 60)

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    database_url = "postgresql://postgres:CgvMUhtnggPLOanphEShtbZREfdRbuQF@shuttle.proxy.rlwy.net:23396/railway"

    print(f"\n>> Connecting to: shuttle.proxy.rlwy.net:23396")
    print(f">> Database: railway")
    print(f">> User: postgres\n")

    try:
        print(">> Attempting connection...")
        conn = await asyncpg.connect(database_url, ssl=ssl_context, timeout=10)
        print("[SUCCESS] CONNECTION SUCCESSFUL!")

        # Test query
        version = await conn.fetchval("SELECT version()")
        print(f"\n>> PostgreSQL Version:")
        print(f"   {version[:80]}...")

        await conn.close()
        print("\n[SUCCESS] Connection closed successfully")

    except asyncio.TimeoutError:
        print("[ERROR] CONNECTION TIMEOUT (10 seconds)")
        print("\nPossible causes:")
        print("  - Railway database is down")
        print("  - Firewall blocking port 23396")
        print("  - Network connectivity issue")

    except ConnectionRefusedError as e:
        print(f"[ERROR] CONNECTION REFUSED: {e}")
        print("\nPossible causes:")
        print("  - Railway database service is stopped")
        print("  - Wrong host/port (check Railway dashboard)")
        print("  - Port 23396 is blocked by firewall/antivirus")

    except Exception as e:
        print(f"[ERROR] CONNECTION FAILED: {e}")
        print(f"\nError type: {type(e).__name__}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_connection())
