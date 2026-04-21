# HyViS — Hybrid Video Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![YOLO](https://img.shields.io/badge/Detection-YOLOv11-green)
![CLIP](https://img.shields.io/badge/Search-OpenAI%20CLIP-blueviolet)

**Multi-modal AI surveillance pipeline for real-time threat detection, action recognition, and semantic video search.**

</div>

---

## Features

| Module | Description |
|--------|-------------|
| 🦴 **Pose Estimation** | YOLOv11-Pose — 17-keypoint COCO skeleton tracking for every person |
| 🥋 **Action Recognition** | ST-GCN — skeleton-based violence detection (NTU-60 classes) |
| 🔫 **Weapon Detection** | Custom YOLO model for Handgun / Rifle / Knife / Sword classification |
| 🔥 **Surge Detection** | Optical-flow motion surge for crowd panic events |
| 🗺️ **Activity Heatmap** | Cumulative person-density heatmap overlaid on video |
| 📐 **Zone Rules Engine** | Shapely-based virtual fence loitering alerts |
| 👁️ **Face Watchlist** | InsightFace ReID matching against a SQLite watchlist |
| 🔍 **CLIP Semantic Search** | Zero-shot text or image-based frame retrieval |
| 🛡️ **Streamlit Dashboard** | Live alert feed, heatmap viewer, zone overview, settings |

---

## Architecture

```
Video Input
    │
    ▼
PoseEstimator (YOLOv11-pose)          ← single tracker for all persons
    ├─► ActionRecognizer (ST-GCN)      ← violence / action labels
    ├─► ZoneRulesEngine               ← virtual fence loitering
    └─► HeatmapEngine                 ← cumulative density map

WeaponDetector (Custom YOLO)
WatchlistMatcher (InsightFace)
CrowdDensityEstimator
SurgeDetector (Optical Flow)

    │ all signals
    ▼
FusionEngine ──► severity score ──► DBManager (SQLite)
    │
    ▼
Annotated Video Output + Saved Frames ──► CLIP Index ──► Dashboard Search
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialise the database
```bash
python init_db.py
```

### 3. Run the pipeline on a video
```bash
python pipeline.py --video footage.mp4 --output result.mp4
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Project Structure

```
HyViS/
├── pipeline.py            # Main video processing pipeline
├── app.py                 # Streamlit web dashboard
├── init_db.py             # Database schema initialisation
├── verify_db.py           # Quick database inspection utility
├── test_core.py           # Core module diagnostic tests
├── input/                 # Place your raw video files here
├── output/                # Processed/Annotated videos go here
│
├── modules/
│   ├── action.py          # ST-GCN action recognition
│   ├── clip_search.py     # CLIP semantic frame search
│   ├── crowd.py           # Crowd density estimation
│   ├── db.py              # SQLite alert logging
│   ├── fusion.py          # Weighted signal fusion / severity scoring
│   ├── heatmap.py         # Cumulative activity heatmap
│   ├── pose.py            # YOLOv11-pose tracker (single source of truth)
│   ├── surge.py           # Optical-flow motion surge detector
│   ├── watchlist.py       # InsightFace face-match watchlist
│   ├── weapon.py          # YOLO weapon detector
│   └── zones.py           # Shapely virtual-fence zone engine
│
├── configs/
│   ├── stgcn_model.py     # ST-GCN model class definition
│   └── ntu60_labels.pkl   # NTU-60 action class labels + violence IDs
│
├── models/                # Model weights (not tracked in git)
│   ├── weapon_yolo11n_best.pt
│   └── stgcn_ntu60_final.pth
│
├── zones.json             # Virtual fence zone configuration
└── requirements.txt
```

---

## Dashboard Pages

| Page | Contents |
|------|----------|
| 📋 **Alert Feed** | Colour-coded table, severity metrics, timeline scatter, alert type pie |
| 🗺️ **Heatmap Viewer** | Person-density heatmap with download button |
| 🔍 **CLIP Search** | **Text tab** — natural language query · **Person tab** — upload photo to find person |
| 📊 **Zone Overview** | Loitering incidents per zone, JSON zone config viewer |
| ⚙️ **Settings** | Clear alerts / frames / CLIP index, show database stats |

---

## Configuration

### zones.json
```json
[
  {
    "name": "Restricted Area",
    "polygon": [[100,100],[400,100],[400,400],[100,400]],
    "loitering_sec": 30,
    "severity": "HIGH"
  }
]
```

### Tunable Thresholds

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Weapon confidence | `modules/weapon.py` | `0.75` | Min YOLO confidence for weapon alert |
| Surge gate | `modules/surge.py` | `2.0 px` | Min optical-flow magnitude before surge fires |
| Action display confidence | `modules/action.py` | `0.50` | Below this, label shows as "Normal" |
| Keypoint draw threshold | `pipeline.py` | `0.15` | Min keypoint confidence to draw skeleton dot |

---

## Known Limitations

- **Domain gap**: ST-GCN is trained on NTU-60 (indoor lab). Real outdoor footage may produce noisy action labels for normal walking.
- **Weapon model**: May produce false positives on phone/remote-like objects. Threshold set to `0.75` to mitigate.
- **Face watchlist**: Empty by default — populate `watchlist` table in `surveillance.db` with InsightFace embeddings to enable.

---

## License

MIT License — see [LICENSE](LICENSE)
