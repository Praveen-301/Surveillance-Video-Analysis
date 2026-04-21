import json
from shapely.geometry import Point, Polygon
from collections import defaultdict

class ZoneRulesEngine:
    def __init__(self, zones_file="zones.json"):
        with open(zones_file) as f:
            self.zones = json.load(f)
        self.entry_frames = defaultdict(dict)

    def check(self, track_id, cx, cy, frame_idx, fps):
        alerts = []
        pt = Point(cx, cy)
        for zone in self.zones:
            poly  = Polygon(zone["polygon"])
            name  = zone["name"]
            limit = zone["loitering_sec"] * fps
            if poly.contains(pt):
                if track_id not in self.entry_frames[name]:
                    self.entry_frames[name][track_id] = frame_idx
                elapsed = frame_idx - self.entry_frames[name][track_id]
                if elapsed > limit:
                    alerts.append({
                        "zone"   : name,
                        "track"  : track_id,
                        "seconds": round(elapsed/fps, 1),
                        "severity": zone["severity"],
                    })
            else:
                self.entry_frames[name].pop(track_id, None)
        return alerts
