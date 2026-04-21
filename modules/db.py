import sqlite3, json

class DBManager:
    def __init__(self, db_path="surveillance.db"):
        self.db_path = db_path

    def log(self, severity, frame_idx, signals):
        alert_type = self._type(signals)
        details    = json.dumps(signals)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO alerts (alert_type,severity,frame_idx,details)"
            " VALUES (?,?,?,?)",
            (alert_type, severity, frame_idx, details)
        )
        conn.commit(); conn.close()

    def _type(self, signals):
        if "weapon"     in signals: return "WEAPON"
        if "action"     in signals: return "VIOLENCE"
        if "face_match" in signals: return "WATCHLIST"
        if "loitering"  in signals: return "LOITERING"
        if "crowd"      in signals: return "CROWD"
        return "GENERAL"

    def get_all_alerts(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY frame_idx"
        ).fetchall()
        conn.close()
        return rows
