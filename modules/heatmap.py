import cv2, numpy as np

class HeatmapEngine:
    def __init__(self, H, W):
        self.acc = np.zeros((H, W), dtype=np.float32)

    def update(self, cx, cy, radius=30):
        if 0<=cy<self.acc.shape[0] and 0<=cx<self.acc.shape[1]:
            mask = np.zeros_like(self.acc)
            cv2.circle(mask, (cx,cy), radius, 1.0, -1)
            self.acc += mask

    def get_colored_heatmap(self):
        if self.acc.max() == 0:
            return np.zeros((*self.acc.shape, 3), dtype=np.uint8)
        acc = self.acc / self.acc.max()
        return cv2.applyColorMap((acc*255).astype(np.uint8), cv2.COLORMAP_JET)

    def apply_overlay(self, frame, alpha=0.4):
        if self.acc.max() == 0:
            return frame
        hm = self.get_colored_heatmap()
        mask = self.acc > 0
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(frame[mask], 1 - alpha, hm[mask], alpha, 0)
        return blended

    def snapshot(self, path="heatmap_final.png"):
        if self.acc.size == 0 or self.acc.max() == 0:
            print(f"Heatmap ignored: empty frame buffer (size 0).")
            return
        hm = self.get_colored_heatmap()
        cv2.imwrite(path, hm)
        print(f"Heatmap saved: {path}")
