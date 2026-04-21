import cv2, numpy as np

class SurgeDetector:
    def detect(self, prev_gray, curr_gray):
        p = cv2.resize(prev_gray, (320,240))
        c = cv2.resize(curr_gray, (320,240))
        flow = cv2.calcOpticalFlowFarneback(
            p, c, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mean_mag = float(np.mean(mag))
        # Gate: ignore ambient/camera motion below 2.0 pixels/frame
        # Normalize against 15.0 instead of 5.0 to be less hair-trigger
        if mean_mag < 2.0:
            return 0.0
        return min((mean_mag - 2.0) / 15.0, 1.0)
