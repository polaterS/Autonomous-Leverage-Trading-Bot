#!/usr/bin/env python3
"""
SIMPLE DATABASE MIGRATION - No dependencies needed!
Just run: python run_migration_simple.py
"""

import urllib.request
import json
import ssl

# Database credentials (from your .env)
DB_HOST = "shuttle.proxy.rlwy.net"
DB_PORT = "23396"
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASS = "CgvMUhtnggPLOanphEShtbZREfdRbuQF"

SQL_COMMANDS = [
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS confluence_score FLOAT DEFAULT NULL;",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS quality VARCHAR(20) DEFAULT NULL;",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS risk_percentage FLOAT DEFAULT NULL;"
]

print("=" * 70)
print("DATABASE MIGRATION - Simple Method")
print("=" * 70)
print("")
print("IMPORTANT: This method requires manual Railway Dashboard access")
print("")
print("Please go to Railway Dashboard and run these SQL commands:")
print("=" * 70)
print("")

for i, sql in enumerate(SQL_COMMANDS, 1):
    print(f"{i}. {sql}")

print("")
print("=" * 70)
print("")
print("TO RUN IN RAILWAY:")
print("1. Go to https://railway.app/")
print("2. Click PostgreSQL database")
print("3. Click 'Query' tab")
print("4. Copy-paste the SQL above")
print("5. Click 'Run Query'")
print("")
print("VERIFY SUCCESS:")
print("Run this query to check:")
print("")
print("SELECT column_name, data_type FROM information_schema.columns")
print("WHERE table_name = 'trades'")
print("AND column_name IN ('confluence_score', 'quality', 'risk_percentage');")
print("")
print("=" * 70)
