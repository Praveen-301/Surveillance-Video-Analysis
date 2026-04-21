import torch, pickle, numpy as np, os
from collections import deque, Counter

# Resolve paths relative to this file
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(_BASE_DIR, "configs")
_MODEL_DIR  = os.path.join(_BASE_DIR, "models")

exec(open(os.path.join(_CONFIG_DIR, "stgcn_model.py")).read())

with open(os.path.join(_CONFIG_DIR, "ntu60_labels.pkl"),"rb") as f:
    _d = pickle.load(f)
    NTU60_LABELS = _d["labels"]
    VIOLENCE_IDS = _d["violence_ids"]

class ActionRecognizer:
    def __init__(self, weights=None, buf_size=30, vote_win=10, vote_thresh=5,
                 display_conf_min=0.50):
        if weights is None:
            weights = os.path.join(_MODEL_DIR, "stgcn_ntu60_final.pth")
        self.model = STGCN(num_classes=60, in_ch=3)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self.buffers          = {}
        self.votes            = {}
        self.label_votes      = {}   # smoothed display label votes
        self.buf_size         = buf_size
        self.vote_win         = vote_win
        self.vote_thresh      = vote_thresh
        self.display_conf_min = display_conf_min   # hide label if model is unsure
        print("Action recognizer ready — 60 NTU classes")

    def update(self, track_id, keypoints_17x3):
        if track_id not in self.buffers:
            self.buffers[track_id]     = deque(maxlen=self.buf_size)
            self.votes[track_id]       = deque(maxlen=self.vote_win)
            self.label_votes[track_id] = deque(maxlen=self.vote_win)
        self.buffers[track_id].append(keypoints_17x3)

    def predict(self, track_id):
        buf = self.buffers.get(track_id)
        if buf is None or len(buf) < self.buf_size:
            return "Normal", 0.0, False
        seq   = np.stack(list(buf))
        seq_t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out       = self.model(seq_t)
            probs     = torch.softmax(out, 1)[0]
            pred_id   = int(probs.argmax())
            pred_conf = float(probs[pred_id])
            pred_name = NTU60_LABELS.get(pred_id, f"class_{pred_id}")
            is_violent = pred_id in VIOLENCE_IDS

            # --- Violence alert vote (unchanged) ---
            vote = "Violence" if is_violent else "Normal"
            self.votes[track_id].append(vote)
            top_label, top_count = Counter(self.votes[track_id]).most_common(1)[0]
            alert = (top_label == "Violence" and top_count >= self.vote_thresh
                     and pred_conf >= 0.25)

            # --- Display label: smoothed + confidence gated ---
            # If model is below display threshold, treat as Normal for display
            display_name = pred_name if pred_conf >= self.display_conf_min else "Normal"
            self.label_votes[track_id].append(display_name)
            smooth_label = Counter(self.label_votes[track_id]).most_common(1)[0][0]

            return smooth_label, pred_conf, alert
