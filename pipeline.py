import cv2, argparse, os, numpy as np
from typing import Dict
from modules.pose      import PoseEstimator
from modules.action    import ActionRecognizer
from modules.weapon    import WeaponDetector
from modules.zones     import ZoneRulesEngine
from modules.surge     import SurgeDetector
from modules.fusion    import fuse
from modules.heatmap   import HeatmapEngine
from modules.db        import DBManager

SAVE_EVERY = 25   # save annotated frame every N frames

# COCO 17-keypoint skeleton bone pairs
BONES = [
    (5,7),(7,9),(6,8),(8,10),           # arms
    (5,6),(5,11),(6,12),(11,12),        # torso
    (11,13),(13,15),(12,14),(14,16),    # legs
    (0,1),(0,2),(1,3),(2,4),            # head
]

def draw_skeleton(frame, kp, W, H, dot_color=(255, 165, 0), line_color=(0, 200, 255)):
    """
    Draw a 17-keypoint COCO skeleton on *frame* in-place.

    Args:
        frame     : BGR image (np.ndarray)
        kp        : (17, 3) array — x_norm, y_norm, confidence per keypoint
        W, H      : frame width / height in pixels
        dot_color : BGR colour for joint dots
        line_color: BGR colour for bone lines
    """
    pts = []
    for kx, ky, kc in kp:
        # Low threshold (0.15) keeps background persons visible
        pts.append((int(kx * W), int(ky * H)) if kc > 0.15 else None)
    for i, j in BONES:
        if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
            cv2.line(frame, pts[i], pts[j], line_color, 1)
    for pt in pts:
        if pt:
            cv2.circle(frame, pt, 3, dot_color, -1)

def main(video_path: str, output_path: str, do_index: bool = False) -> None:
    # Capture absolute paths BEFORE changing directory
    video_path  = os.path.abspath(video_path)
    output_path = os.path.abspath(output_path)

    # Ensure relative paths (to configs/models) are resolved from the script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)

    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Please check the path.")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out   = cv2.VideoWriter(output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    os.makedirs("saved_frames", exist_ok=True)

    # Single pose model is now the tracker — no separate Detector needed
    pose  = PoseEstimator()
    act   = ActionRecognizer()
    wpn   = WeaponDetector()
    zone  = ZoneRulesEngine("zones.json")
    surge = SurgeDetector()
    hmap  = HeatmapEngine(H, W)
    db    = DBManager("surveillance.db")

    frame_idx         = 0
    prev_gray         = None
    physical_signals: Dict[int, Dict] = {}   # frame_idx -> signal snapshot for FCVAR

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        signals      = {}
        violent_tids = set()
        tid_labels   = {}   # track_id -> (label, conf) for overlay

        # pose.extract() returns BOTH tracked dict and all_persons list
        kpts, all_persons = pose.extract(frame)

        for person in all_persons:
            tid  = person["tid"]
            x1, y1, x2, y2 = person["bbox"]
            cx   = int((x1 + x2) / 2)
            cy   = int((y1 + y2) / 2)

            if tid is not None:
                act.update(tid, person["kp"])
                label, conf, alert = act.predict(tid)
                tid_labels[tid] = (label, conf)
                if alert:
                    hmap.update(cx, cy)
                    signals["action"]       = conf
                    signals["action_label"] = label
                    violent_tids.add(tid)
                zone_alerts = zone.check(tid, cx, cy, frame_idx, fps)
                if zone_alerts:
                    signals["loitering"] = 1.0
            else:
                hmap.update(cx, cy)

        threats = wpn.detect(frame)
        if threats:
            signals["weapon"]       = max(t["conf"] for t in threats)
            signals["weapon_class"] = threats[0]["class"]
            for t in threats:
                wcx = int((t["bbox"][0]+t["bbox"][2])/2)
                wcy = int((t["bbox"][1]+t["bbox"][3])/2)
                hmap.update(wcx, wcy)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            sv = surge.detect(prev_gray, gray)
            if sv > 0.5: signals["surge"] = sv
        prev_gray = gray

        # ----------------------------------------------------------------
        # Compute fused score & log HIGH/CRITICAL alerts
        # ----------------------------------------------------------------
        fusion_score = 0.0
        if signals:
            fusion_score, severity = fuse(signals)
            if severity in ("HIGH", "CRITICAL"):
                db.log(severity, frame_idx, signals)
                print(f"Frame {frame_idx:05d} | {severity:8s} | {signals}")

        # Collect physical snapshot for FCVAR indexing (every frame)
        physical_signals[frame_idx] = {
            "weapon_score"  : float(signals.get("weapon",    0.0)),
            "action_score"  : float(signals.get("action",    0.0)),
            "zone_violation": int("loitering" in signals),
            "surge_score"   : float(signals.get("surge",     0.0)),
            "fusion_score"  : round(fusion_score, 4),
            "track_ids"     : [p["tid"] for p in all_persons if p["tid"] is not None],
        }

        # --- Visual Overlay ---
        frame = hmap.apply_overlay(frame, alpha=0.35)

        # Draw every detected person: skeleton + action label + bounding box
        for person in all_persons:
            tid  = person["tid"]
            x1, y1, x2, y2 = person["bbox"]
            is_violent = (tid is not None and tid in violent_tids)
            color = (0, 0, 255) if is_violent else (0, 255, 0)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Action label above box
            if tid is not None and tid in tid_labels:
                lbl, cf = tid_labels[tid]
                txt = f"T{tid}:{lbl}({cf:.2f})"
            elif tid is not None:
                txt = f"T{tid}:..."
            else:
                txt = "T?:..."
            cv2.putText(frame, txt, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Skeleton keypoints & bones
            draw_skeleton(frame, person["kp"], W, H)

        # Weapon boxes
        for t in threats:
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"WEAPON: {t['class']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Frame info HUD
        cv2.putText(frame,
            f"Frame {frame_idx} | Det:{len(all_persons)} | Pose:{len(kpts)} | Action:{len(tid_labels)}",
            (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if frame_idx % SAVE_EVERY == 0:
            cv2.imwrite(f"saved_frames/frame_{frame_idx:06d}.jpg", frame)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{total}] frames")

    cap.release(); out.release()
    hmap.snapshot("heatmap_final.png")
    print(f"\nDone: {output_path}")

    # ------------------------------------------------------------------
    # Optional FCVAR indexing (triggered by --index flag)
    # ------------------------------------------------------------------
    if do_index:
        print("\nStarting FCVAR indexing…")
        try:
            from modules.retrieval import FCVARManager
            mgr = FCVARManager()
            mgr.index_video(video_path, physical_signals, sample_fps=1.0)
            print("FCVAR indexing complete.")
        except Exception as exc:
            print(f"FCVAR indexing failed: {exc}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True,       help="Path to input video")
    ap.add_argument("--output",  default="output_annotated.mp4", help="Output annotated video")
    ap.add_argument("--index",   action="store_true", help="Run FCVAR indexing after processing")
    args = ap.parse_args()
    main(args.video, args.output, args.index)
