import sqlite3
import sys

db_path = "/home/garrett/VSCode/EECE7150-final2/snell_lib.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open("db_schema.txt", "w") as f:
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        f.write(f"Tables: {[t[0] for t in tables]}\n")
        
        # Inspect Node or Data table if they exist
        for table_name in ['Node', 'Data', 'Images']:
            if any(table_name == t[0] for t in tables):
                f.write(f"\nSchema for {table_name}:\n")
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                for col in columns:
                    f.write(f"{col}\n")
                    
                # Check first row to see data types
                f.write(f"\nFirst row of {table_name}:\n")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                row = cursor.fetchone()
                f.write(f"{row}\n")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
