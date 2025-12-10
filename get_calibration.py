import sqlite3
import cv2
import numpy as np
import sys

db_path = "/home/garrett/VSCode/EECE7150-final2/snell_lib.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get one row with calibration data
    cursor.execute("SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        calib_data = row[0]
        print(f"Calibration data size: {len(calib_data)} bytes")
        
        # Try reading as int32
        ints = np.frombuffer(calib_data, dtype=np.int32)
        print(f"As int32 (first 10): {ints[:10]}")
        for i, val in enumerate(ints[:10]):
            print(f"Index {i}: {val}")
        
        # Try reading as float64 (skipping potential header)
        # Header seems to be ~4-16 bytes?
        # Let's try multiple offsets
        for offset in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
            try:
                floats = np.frombuffer(calib_data, dtype=np.float64, offset=offset)
                print(f"As float64 (offset {offset}, first 10): {floats[:10]}")
            except Exception:
                pass
                
        # Try reading as float32
        for offset in [0, 4]:
            try:
                floats = np.frombuffer(calib_data, dtype=np.float32, offset=offset)
                print(f"As float32 (offset {offset}, first 10): {floats[:10]}")
            except Exception:
                pass
            
    else:
        print("No calibration data found in Data table.")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
