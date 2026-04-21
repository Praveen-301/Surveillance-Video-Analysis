"""
HyViS Core Module Test Script
===============================
Loads a sample video and runs the 4 core modules:
  1. Detector    — person detection + tracking (YOLOv11)
  2. PoseEstimator — keypoint extraction     (YOLOv11-pose)
  3. ActionRecognizer — ST-GCN action class  (requires trained weights)
  4. WeaponDetector   — threat detection     (requires custom weights)

Usage:
    python test_core.py --video <path_to_video.mp4>
    python test_core.py --video <path>  --weapon-weights models/weapon_yolo11n_best.pt
    python test_core.py --video <path>  --skip-action  --skip-weapon   # fastest test

Notes:
    - ActionRecognizer and WeaponDetector will gracefully skip (with a warning)
      if their model weights are not present.
    - ST-GCN requires configs/stgcn_model.py and configs/ntu60_labels.pkl
"""

import sys, os, argparse, time, cv2, numpy as np

# ── resolve imports whether run from HyViS/ or AIML Project/ ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── helpers ───────────────────────────────────────────────────────────────────

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{d['track_id']} {d['conf']:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return frame


def draw_pose(frame, pose_map, color=(255, 165, 0)):
    """Draw keypoint dots for every tracked person."""
    # COCO-17 skeleton connections
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),          # head
        (5,6),(5,7),(7,9),(6,8),(8,10),    # arms
        (5,11),(6,12),(11,12),             # torso
        (11,13),(13,15),(12,14),(14,16),   # legs
    ]
    h, w = frame.shape[:2]
    for tid, kps in pose_map.items():
        pts = []
        for kp in kps:
            x, y, c = kp
            px, py = int(x * w), int(y * h)
            pts.append((px, py, c))
            if c > 0.3:
                cv2.circle(frame, (px, py), 4, color, -1)
        for i, j in SKELETON:
            if i < len(pts) and j < len(pts):
                if pts[i][2] > 0.3 and pts[j][2] > 0.3:
                    cv2.line(frame, pts[i][:2], pts[j][:2], color, 1)
    return frame


def draw_weapons(frame, threats):
    for t in threats:
        x1, y1, x2, y2 = t["bbox"]
        label = f"{t['class']} [{t['severity']}] {t['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    return frame


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HyViS Core Module Tester")
    p.add_argument("--video",           required=True,             help="Path to input video file")
    p.add_argument("--det-weights",     default="yolo11n.pt",      help="Detector weights")
    p.add_argument("--pose-weights",    default="yolo11n-pose.pt", help="Pose weights")
    p.add_argument("--weapon-weights",  default="models/weapon_yolo11n_best.pt")
    p.add_argument("--det-conf",        type=float, default=0.40)
    p.add_argument("--max-frames",      type=int,   default=300,   help="Max frames to process (0=all)")
    p.add_argument("--skip-action",     action="store_true",       help="Skip ActionRecognizer")
    p.add_argument("--skip-weapon",     action="store_true",       help="Skip WeaponDetector")
    p.add_argument("--show",            action="store_true",       help="Show live preview window")
    p.add_argument("--save-preview",    default="",                help="Save annotated preview video to this path")
    return p.parse_args()


def load_modules(args):
    modules = {}

    # ── 1. Detector ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading Detector …")
    from modules.detector import Detector
    modules["detector"] = Detector(weights=args.det_weights, conf=args.det_conf)
    print("  ✓ Detector ready")

    # ── 2. PoseEstimator ─────────────────────────────────────────────────────
    print("[2/4] Loading PoseEstimator …")
    from modules.pose import PoseEstimator
    modules["pose"] = PoseEstimator(weights=args.pose_weights)
    print("  ✓ PoseEstimator ready")

    # ── 3. ActionRecognizer ──────────────────────────────────────────────────
    if args.skip_action:
        print("[3/4] ActionRecognizer — SKIPPED (--skip-action flag)")
        modules["action"] = None
    else:
        stgcn_cfg    = os.path.join(SCRIPT_DIR, "configs", "stgcn_model.py")
        stgcn_labels = os.path.join(SCRIPT_DIR, "configs", "ntu60_labels.pkl")
        stgcn_wts    = os.path.join(SCRIPT_DIR, "models",  "stgcn_ntu60_final.pth")
        missing = [p for p in [stgcn_cfg, stgcn_labels, stgcn_wts] if not os.path.exists(p)]
        if missing:
            print(f"[3/4] ActionRecognizer — SKIPPED (missing files: {missing})")
            modules["action"] = None
        else:
            print("[3/4] Loading ActionRecognizer …")
            os.chdir(SCRIPT_DIR)           # action.py uses relative open() calls
            from modules.action import ActionRecognizer
            modules["action"] = ActionRecognizer()
            print("  ✓ ActionRecognizer ready")

    # ── 4. WeaponDetector ────────────────────────────────────────────────────
    if args.skip_weapon:
        print("[4/4] WeaponDetector — SKIPPED (--skip-weapon flag)")
        modules["weapon"] = None
    else:
        wpt = os.path.join(SCRIPT_DIR, args.weapon_weights)
        if not os.path.exists(wpt):
            print(f"[4/4] WeaponDetector — SKIPPED (weights not found: {wpt})")
            modules["weapon"] = None
        else:
            print("[4/4] Loading WeaponDetector …")
            from modules.weapon import WeaponDetector
            modules["weapon"] = WeaponDetector(weights=wpt)
            print("  ✓ WeaponDetector ready")

    return modules


def run_pipeline(video_path, modules, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_fr  = args.max_frames if args.max_frames > 0 else total

    print(f"\n{'='*60}")
    print(f"  Video : {video_path}")
    print(f"  Size  : {w}×{h}  |  FPS: {fps_in:.1f}  |  Total frames: {total}")
    print(f"  Processing up to {max_fr} frames …")
    print(f"{'='*60}\n")

    # optional output writer
    writer = None
    if args.save_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_preview, fourcc, fps_in, (w, h))

    # stats accumulators
    total_detections = 0
    total_poses      = 0
    action_alerts    = 0
    weapon_alerts    = 0
    frame_times      = []

    frame_idx = 0
    while frame_idx < max_fr:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Detection ────────────────────────────────────────────────────────
        detections = modules["detector"].detect(frame)
        total_detections += len(detections)

        # ── Pose ─────────────────────────────────────────────────────────────
        pose_map = modules["pose"].extract(frame)
        total_poses += len(pose_map)

        # ── Action ───────────────────────────────────────────────────────────
        action_results = {}
        if modules["action"] is not None:
            for tid, kps in pose_map.items():
                modules["action"].update(tid, kps)
            for tid in pose_map:
                label, conf, alert = modules["action"].predict(tid)
                action_results[tid] = (label, conf, alert)
                if alert:
                    action_alerts += 1
                    print(f"  [ACTION ALERT] Frame {frame_idx:05d} | "
                          f"Track {tid} | {label} ({conf:.2f})")

        # ── Weapon ───────────────────────────────────────────────────────────
        threats = []
        if modules["weapon"] is not None:
            threats = modules["weapon"].detect(frame)
            weapon_alerts += len(threats)
            for t in threats:
                print(f"  [WEAPON ALERT] Frame {frame_idx:05d} | "
                      f"{t['class']} [{t['severity']}] conf={t['conf']:.2f} "
                      f"bbox={t['bbox'].tolist()}")

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)

        # ── Annotate & display ───────────────────────────────────────────────
        vis = frame.copy()
        vis = draw_detections(vis, detections)
        vis = draw_pose(vis, pose_map)
        vis = draw_weapons(vis, threats)

        # Action labels overlay
        for tid, (label, conf, alert) in action_results.items():
            color = (0, 0, 255) if alert else (200, 200, 200)
            cv2.putText(vis, f"T{tid}:{label}({conf:.2f})", (10, 20 + tid * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # HUD
        fps_live = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"Frame {frame_idx} | Det:{len(detections)} | "
                         f"Pose:{len(pose_map)} | {fps_live:.1f} FPS",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if writer:
            writer.write(vis)
        if args.show:
            cv2.imshow("HyViS Core Test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ── Per-frame console log (every 30 frames) ──────────────────────────
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:05d} | "
                  f"persons={len(detections)} | "
                  f"poses={len(pose_map)} | "
                  f"threats={len(threats)} | "
                  f"time={elapsed*1000:.1f}ms")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ── Final Summary ─────────────────────────────────────────────────────────
    avg_ms = np.mean(frame_times) * 1000 if frame_times else 0
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {frame_idx} frames processed")
    print(f"  Avg latency  : {avg_ms:.1f} ms/frame  ({1000/avg_ms:.1f} FPS)")
    print(f"  Total det    : {total_detections}  (avg {total_detections/max(frame_idx,1):.1f}/frame)")
    print(f"  Total poses  : {total_poses}")
    print(f"  Action alerts: {action_alerts}  "
          f"{'(module inactive)' if modules['action'] is None else ''}")
    print(f"  Weapon alerts: {weapon_alerts}  "
          f"{'(module inactive)' if modules['weapon'] is None else ''}")
    print(f"{'='*60}\n")

    # ── Quick pass/fail gate ─────────────────────────────────────────────────
    print("── VERIFICATION ──────────────────────────────────────────")
    if total_detections > 0:
        print("  [PASS] Detector fired — persons detected")
    else:
        print("  [WARN] Detector returned 0 detections (check video / conf threshold)")

    if total_poses > 0:
        print("  [PASS] PoseEstimator fired — keypoints extracted")
    else:
        print("  [WARN] PoseEstimator returned 0 poses")

    if modules["action"] is not None:
        print("  [PASS] ActionRecognizer loaded — "
              f"{action_alerts} alert(s) triggered")
    else:
        print("  [INFO] ActionRecognizer was not loaded (weights missing or skipped)")

    if modules["weapon"] is not None:
        print("  [PASS] WeaponDetector loaded — "
              f"{weapon_alerts} threat(s) detected")
    else:
        print("  [INFO] WeaponDetector was not loaded (weights missing or skipped)")

    print("──────────────────────────────────────────────────────────\n")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args    = parse_args()
    modules = load_modules(args)
    run_pipeline(args.video, modules, args)
