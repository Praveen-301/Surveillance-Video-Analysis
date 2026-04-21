import sqlite3
import json
import os

def verify():
    # Resolve DB path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path  = os.path.join(base_dir, 'surveillance.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("--- 5 Latest High-Level Alerts ---")
    cursor.execute("SELECT alert_type, severity, frame_idx, timestamp, details FROM alerts WHERE severity != 'LOW' ORDER BY id DESC LIMIT 5")
    for row in cursor.fetchall():
        print(f"[{row[3]}] {row[0]} ({row[1]}) at Frame {row[2]}")
        print(f"   Details: {row[4]}")
    
    print("\n--- Any Violence Alerts? ---")
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type = 'VIOLENCE'")
    print(f"Total Violence alerts: {cursor.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    verify()
