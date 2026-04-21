from ultralytics import YOLO
import numpy as np


class PoseEstimator:
    """
    Wraps YOLOv11-pose for person detection + 17-keypoint extraction.
    Acts as the single person tracker for the entire pipeline,
    eliminating the need for a separate detection model.
    """

    def __init__(self, weights="yolo11n-pose.pt", det_conf=0.25):
        """
        Args:
            weights  : path to YOLOv11-pose weights file.
            det_conf : detection confidence threshold.
                       Lower values detect smaller/background persons.
        """
        self.model    = YOLO(weights)
        self.det_conf = det_conf

    def extract(self, frame):
        """
        Run pose estimation + tracking on a single frame.

        Returns
        -------
        tracked : dict  {track_id -> np.ndarray (17, 3)}
            Keypoints (x_norm, y_norm, conf) keyed by stable track ID.
            Used by ActionRecognizer for ST-GCN inference.

        all_persons : list of dicts
            One entry per detected person (including untracked ones):
              - 'tid'  : int track_id, or None if tracking was lost this frame
              - 'bbox' : np.ndarray [x1, y1, x2, y2]
              - 'kp'   : np.ndarray (17, 3) — x_norm, y_norm, conf per keypoint
        """
        results = self.model.track(
            frame, persist=True, conf=self.det_conf, verbose=False
        )[0]

        tracked     = {}
        all_persons = []

        if results.keypoints is None:
            return tracked, all_persons

        ids = results.boxes.id   # tensor of track IDs; may be None

        for i in range(len(results.keypoints.xyn)):
            xy   = results.keypoints.xyn[i].cpu().numpy()     # (17, 2) normalized
            conf = results.keypoints.conf[i].cpu().numpy()    # (17,)
            kp   = np.concatenate([xy, conf[:, np.newaxis]], axis=1)  # (17, 3)
            bbox = results.boxes.xyxy[i].cpu().numpy().astype(int)

            tid = None
            if ids is not None and i < len(ids) and ids[i] is not None:
                tid = int(ids[i])
                tracked[tid] = kp

            all_persons.append({"tid": tid, "bbox": bbox, "kp": kp})

        return tracked, all_persons
