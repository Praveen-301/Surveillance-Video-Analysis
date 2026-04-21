from ultralytics import YOLO

class WeaponDetector:
    SEVERITY = {
        "Handgun":"CRITICAL","Rifle":"CRITICAL","Shotgun":"CRITICAL",
        "Knife":"HIGH","Sword":"HIGH","Axe":"HIGH",
    }
    def __init__(self, weights="models/weapon_yolo11n_best.pt", conf=0.75):
        self.model = YOLO(weights)
        self.conf  = conf
        print(f"Weapon detector loaded: {self.model.names}")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        threats = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)]
            threats.append({
                "class"   : cls_name,
                "severity": self.SEVERITY.get(cls_name,"HIGH"),
                "conf"    : float(box.conf),
                "bbox"    : box.xyxy[0].cpu().numpy().astype(int),
            })
        return threats
