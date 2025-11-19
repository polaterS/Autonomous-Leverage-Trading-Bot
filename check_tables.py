"""Check database tables"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# List all tables
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
""")

tables = cur.fetchall()
print("\nDATABASE TABLES:")
print("="*50)
for table in tables:
    print(f"  - {table[0]}")

# Check if trading_positions table exists
if any('position' in str(t[0]).lower() for t in tables):
    position_table = [t[0] for t in tables if 'position' in str(t[0]).lower()][0]
    print(f"\nFound position table: {position_table}")

    # Count records
    cur.execute(f"SELECT COUNT(*) FROM {position_table}")
    count = cur.fetchone()[0]
    print(f"   Total records: {count}")

    # Get column names
    cur.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{position_table}'
    """)
    columns = cur.fetchall()
    print(f"\n   Columns:")
    for col in columns:
        print(f"     - {col[0]}")
else:
    print("\nNo position table found!")

conn.close()
