import insightface, sqlite3, numpy as np

class WatchlistMatcher:
    def __init__(self, threshold=0.65):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1)
        self.threshold = threshold
        self.watchlist = self._load()
        print(f"Watchlist loaded: {len(self.watchlist)} persons")

    def _load(self):
        conn = sqlite3.connect("surveillance.db")
        rows = conn.execute(
            "SELECT name,risk_level,embedding FROM watchlist"
        ).fetchall()
        conn.close()
        result = []
        for name, risk, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            result.append({"name":name,"risk":risk,"emb":emb})
        return result

    def match(self, frame):
        faces = self.app.get(frame)
        matches = []
        for face in faces:
            emb = face.normed_embedding
            for entry in self.watchlist:
                sim = float(np.dot(emb, entry["emb"]))
                if sim >= self.threshold:
                    matches.append({
                        "name": entry["name"],
                        "risk": entry["risk"],
                        "conf": sim,
                        "bbox": face.bbox.astype(int),
                    })
        return matches
