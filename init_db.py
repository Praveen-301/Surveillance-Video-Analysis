import sqlite3, os

conn = sqlite3.connect("surveillance.db")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.executescript("""
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type  TEXT,
    severity    TEXT,
    frame_idx   INTEGER,
    timestamp   TEXT DEFAULT (datetime('now')),
    details     TEXT,
    acknowledged INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS watchlist (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    risk_level  TEXT,
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS heatmap_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path  TEXT,
    frame_idx   INTEGER,
    timestamp   TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS zone_config (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_name       TEXT,
    polygon_json    TEXT,
    loitering_sec   INTEGER,
    crowd_max       INTEGER,
    severity        TEXT
);
""")
conn.commit(); conn.close()
os.makedirs("saved_frames", exist_ok=True)
print("Database initialized: surveillance.db")
print("saved_frames/ directory created")
