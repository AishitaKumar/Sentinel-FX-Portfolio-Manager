import sqlite3
import os

def init_corporate_ledger():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/exposure_ledger.db")
    cursor = conn.cursor()
    
    # Wipe the old table completely
    cursor.execute("DROP TABLE IF EXISTS corporate_invoices")
    
    # Recreate the table completely empty
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS corporate_invoices (
        invoice_id TEXT PRIMARY KEY,
        counterparty TEXT,
        country TEXT,
        currency TEXT,
        amount REAL,
        due_days INTEGER,
        hedged_amount REAL DEFAULT 0.0
    )
    """)
    
    conn.commit()
    conn.close()
    print("Your corporate ledger is completely empty and waiting for dynamic entries.")

if __name__ == "__main__":
    init_corporate_ledger()