from ultralytics import YOLO

class Detector:
    def __init__(self, weights="yolo11n.pt", conf=0.40):
        self.model = YOLO(weights)
        self.conf  = conf

    def detect(self, frame):
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)[0]
        detections = []
        if results.boxes.id is None:
            return detections
        for box, tid in zip(results.boxes, results.boxes.id):
            detections.append({
                'track_id': int(tid),
                'bbox'    : box.xyxy[0].cpu().numpy().astype(int),
                'conf'    : float(box.conf),
            })
        return detections
