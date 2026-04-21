WEIGHTS = {
    "weapon"    : 0.40,
    "action"    : 0.30,
    "face_match": 0.35,
    "loitering" : 0.15,
    "crowd"     : 0.20,
    "surge"     : 0.10,  # Supporting signal only — not a primary threat indicator
}

def fuse(signals):
    score = 0.0
    total = 0.0
    for key, weight in WEIGHTS.items():
        if key in signals:
            score += float(signals[key]) * weight
            total += weight
    if total == 0: return 0.0, "NORMAL"
    score = score / total
    if   score >= 0.85: severity = "CRITICAL"
    elif score >= 0.70: severity = "HIGH"
    elif score >= 0.50: severity = "MEDIUM"
    else:               severity = "LOW"
    return round(score, 3), severity
