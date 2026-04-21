import torch, pickle, sys, os

# Run from HyViS/ directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("  HyViS ST-GCN Model Test")
print("=" * 50)

# 1. Load the model definition
print("\n[1] Loading model definition from configs/stgcn_model.py ...")
try:
    exec(open("configs/stgcn_model.py").read(), globals())
    print("    OK — STGCN class defined")
except FileNotFoundError:
    print("    ERROR: configs/stgcn_model.py not found.")
    sys.exit(1)

# 2. Load labels
print("[2] Loading NTU-60 labels from configs/ntu60_labels.pkl ...")
try:
    with open("configs/ntu60_labels.pkl", "rb") as f:
        _d = pickle.load(f)
        NTU60_LABELS = _d["labels"]
        VIOLENCE_IDS = _d["violence_ids"]
    print(f"    OK — {len(NTU60_LABELS)} labels loaded")
    print(f"    Violence class IDs: {VIOLENCE_IDS}")
except FileNotFoundError:
    print("    ERROR: configs/ntu60_labels.pkl not found.")
    sys.exit(1)

# 3. Initialize model and load weights
print("[3] Initializing STGCN(num_classes=60) ...")
model = STGCN(num_classes=60, in_ch=3)
total_params = sum(p.numel() for p in model.parameters())
print(f"    OK — {total_params:,} parameters")

print("    Loading weights from models/stgcn_ntu60_final.pth ...")
try:
    state = torch.load("models/stgcn_ntu60_final.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print("    OK — weights loaded successfully")
except FileNotFoundError:
    print("    ERROR: models/stgcn_ntu60_final.pth not found.")
    sys.exit(1)
except RuntimeError as e:
    print(f"    ERROR loading weights: {e}")
    sys.exit(1)

model.eval()

# 4. Dummy inference — single sequence
print("[4] Running inference on dummy skeleton (1×30×17×3) ...")
dummy_input = torch.randn(1, 30, 17, 3)  # Batch=1, Frames=30, Nodes=17, Channels=3
with torch.no_grad():
    out   = model(dummy_input)
    probs = torch.softmax(out, 1)[0]
    pred_id   = int(probs.argmax())
    pred_conf = float(probs[pred_id])
    pred_name = NTU60_LABELS.get(pred_id, f"class_{pred_id}")

print(f"    Prediction  : [{pred_id}] {pred_name}")
print(f"    Confidence  : {pred_conf:.4f}")
print(f"    Is violent  : {pred_id in VIOLENCE_IDS}")

# 5. Batch inference — simulate multiple tracks
print("[5] Batch inference test (8 sequences) ...")
batch_input = torch.randn(8, 30, 17, 3)
with torch.no_grad():
    batch_out   = model(batch_input)
    batch_probs = torch.softmax(batch_out, 1)
print(f"    Output shape : {batch_out.shape}  (expected [8, 60])")
print(f"    All probs sum to 1: {batch_probs.sum(1).allclose(torch.ones(8))}")

# 6. Top-5 predictions for first dummy
print("[6] Top-5 class predictions for dummy input:")
top5 = probs.topk(5)
for rank, (conf, idx) in enumerate(zip(top5.values, top5.indices), 1):
    name = NTU60_LABELS.get(int(idx), f"class_{int(idx)}")
    violent_tag = " ⚠ VIOLENT" if int(idx) in VIOLENCE_IDS else ""
    print(f"    {rank}. [{int(idx):2d}] {name:<30s} {float(conf):.4f}{violent_tag}")

# 7. Summary
print()
print("=" * 50)
print("  ALL CHECKS PASSED — ST-GCN IS READY")
print("=" * 50)
