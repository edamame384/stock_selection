import sys
import os

# Avoid Shift-JIS decode errors on Japanese Windows
os.environ.setdefault("PGCLIENTENCODING", "utf8")

import psycopg2

passwords = ["postgres", "password", "1234", "admin", "root", ""]

for pwd in passwords:
    try:
        dsn = f"host=localhost port=5432 dbname=stock_selection user=postgres password={pwd} connect_timeout=3"
        conn = psycopg2.connect(dsn)
        print(f"Connected! password={repr(pwd)}")
        conn.close()
        sys.exit(0)
    except Exception as e:
        try:
            msg = str(e)
        except Exception:
            msg = repr(e)
        print(f"NG password={repr(pwd)}: {msg[:80]}")

print("All failed")
