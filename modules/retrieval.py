"""
modules/retrieval.py — FCVARManager
Fusion-Conditioned Video Anomaly Retrieval.

Bridges CLIP semantic similarity (S_semantic) with HyViS physical fusion
scores (F_physical) via a weighted reranking formula:

    R_final = (alpha * S_semantic) + (beta * F_physical)

Data layout
-----------
* FAISS IndexFlatIP  : stores L2-normalised 512-dim CLIP visual embeddings.
* hyvis_metadata.db  : stores per-frame physical metadata (weapon_score,
                       action_score, zone_violation, surge_score,
                       fusion_score, track_ids).  A `faiss_row` column
                       provides O(1) reverse-lookup from a FAISS result
                       back to its physical metadata row.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Literal, Optional

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
EMBED_DIM:     int = 512
TOP_N_DEFAULT: int = 100          # FAISS candidates before reranking
TOP_K_DEFAULT: int = 20           # final results returned to caller


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def _get_db(db_path: str) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with Row factory."""
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db(db_path: str) -> None:
    """Create the `frames` table and indexes if they don't exist."""
    conn = _get_db(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS frames (
            frame_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path      TEXT    NOT NULL,
            timestamp_sec   REAL    NOT NULL,
            faiss_row       INTEGER NOT NULL,
            weapon_score    REAL    DEFAULT 0.0,
            action_score    REAL    DEFAULT 0.0,
            zone_violation  INTEGER DEFAULT 0,
            surge_score     REAL    DEFAULT 0.0,
            fusion_score    REAL    DEFAULT 0.0,
            track_ids       TEXT    DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_frame_id  ON frames(frame_id);
        CREATE INDEX IF NOT EXISTS idx_faiss_row ON frames(faiss_row);
        CREATE INDEX IF NOT EXISTS idx_video     ON frames(video_path);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# FCVARManager
# ---------------------------------------------------------------------------
class FCVARManager:
    """
    Fusion-Conditioned Video Anomaly Retrieval manager.

    Parameters
    ----------
    db_path    : path to hyvis_metadata.db (created on first run)
    index_path : path to FAISS index file   (created on first index_video call)
    """

    def __init__(
        self,
        db_path:    str = "hyvis_metadata.db",
        index_path: str = "fcvar.index",
    ) -> None:
        self.db_path    = db_path
        self.index_path = index_path

        print("Loading CLIP (openai/clip-vit-base-patch32)...")
        self.clip      = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.clip.eval()
        print("  OK CLIP ready.")

        _init_db(db_path)
        self.index = self._load_or_create_index()

    # ------------------------------------------------------------------
    # FAISS helpers
    # ------------------------------------------------------------------
    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        if os.path.exists(self.index_path):
            idx = faiss.read_index(self.index_path)
            print(f"  OK FAISS index loaded — {idx.ntotal} vectors from '{self.index_path}'.")
        else:
            idx = faiss.IndexFlatIP(EMBED_DIM)
            print(f"  OK New FAISS IndexFlatIP({EMBED_DIM}) created.")
        return idx

    def _save_index(self) -> None:
        faiss.write_index(self.index, self.index_path)

    # ------------------------------------------------------------------
    # CLIP embedding helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _embed_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Embed a BGR OpenCV frame → (1, 512) normalised float32."""
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        feat   = self.clip.get_image_features(**inputs)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a text query → (1, 512) normalised float32."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        feat   = self.clip.get_text_features(**inputs)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _embed_image_path(self, image_path: str) -> np.ndarray:
        """Embed a query image file → (1, 512) normalised float32."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Query image not found: '{image_path}'")
        pil    = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt")
        feat   = self.clip.get_image_features(**inputs)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_video(
        self,
        video_path:       str,
        physical_signals: Optional[Dict[int, Dict[str, Any]]] = None,
        sample_fps:       float = 1.0,
    ) -> int:
        """
        Sample `video_path` at `sample_fps`, generate CLIP embeddings,
        write them to FAISS, and populate the SQLite `frames` table with
        the corresponding physical signal data.

        Parameters
        ----------
        video_path :
            Absolute or relative path to the source video file.
        physical_signals :
            Dict keyed by frame_idx (int).  Each value is a dict with keys:
              weapon_score, action_score, zone_violation, surge_score,
              fusion_score, track_ids.
            Produced by pipeline.py's frame loop and injected here so the
            FCVARManager itself stays stateless w.r.t. the live pipeline.
        sample_fps :
            Frames per second to sample from the video.  Default: 1.0.

        Returns
        -------
        Number of frames indexed.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: '{video_path}'")

        cap     = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: '{video_path}'")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step    = max(1, int(round(vid_fps / sample_fps)))

        physical_signals = physical_signals or {}
        embeddings: List[np.ndarray] = []
        rows:       List[tuple]       = []
        frame_idx = 0
        indexed   = 0

        print(f"Indexing '{os.path.basename(video_path)}' at {sample_fps} FPS ...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                try:
                    emb     = self._embed_frame(frame)
                    faiss_r = self.index.ntotal + len(embeddings)
                    sig     = physical_signals.get(frame_idx, {})

                    rows.append((
                        video_path,
                        round(frame_idx / vid_fps, 3),
                        faiss_r,
                        float(sig.get("weapon_score",   0.0)),
                        float(sig.get("action_score",   0.0)),
                        int(  sig.get("zone_violation", 0)),
                        float(sig.get("surge_score",    0.0)),
                        float(sig.get("fusion_score",   0.0)),
                        json.dumps(sig.get("track_ids", [])),
                    ))
                    embeddings.append(emb)
                    indexed += 1

                    if indexed % 50 == 0:
                        print(f"  ... {indexed} frames embedded")

                except Exception as exc:
                    print(f"  Warning: skipping frame {frame_idx} — {exc}")

            frame_idx += 1

        cap.release()

        if not embeddings:
            print("Warning: No frames were indexed.")
            return 0

        # Batch-add to FAISS
        mat = np.vstack(embeddings).astype(np.float32)
        self.index.add(mat)
        self._save_index()

        # Batch-insert into SQLite
        conn = _get_db(self.db_path)
        try:
            conn.executemany(
                """INSERT INTO frames
                   (video_path, timestamp_sec, faiss_row,
                    weapon_score, action_score, zone_violation,
                    surge_score, fusion_score, track_ids)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            print(f"DB write error: {exc}")
        finally:
            conn.close()

        print(f"  OK Indexed {indexed} frames. FAISS total: {self.index.ntotal}.")
        return indexed

    # ------------------------------------------------------------------
    # Search & Reranking
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        mode:  Literal["text", "image"] = "text",
        alpha: float = 0.4,
        beta:  float = 0.6,
        top_n: int   = TOP_N_DEFAULT,
        top_k: int   = TOP_K_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """
        Semantic + physical reranking search.

        R_final = (alpha * S_semantic) + (beta * F_physical)

        Parameters
        ----------
        query : Text string (mode='text') or path to an image file (mode='image').
        mode  : 'text' or 'image'.
        alpha : Weight for CLIP semantic similarity  (default 0.4).
        beta  : Weight for physical fusion score     (default 0.6).
        top_n : Number of FAISS candidates to retrieve before reranking.
        top_k : Number of final results returned after reranking.

        Returns
        -------
        List of result dicts sorted by R_final descending.
        """
        if self.index.ntotal == 0:
            raise RuntimeError(
                "FAISS index is empty — run index_video() first."
            )

        # 1. Encode query
        if mode == "text":
            q_emb = self._embed_text(query)
        elif mode == "image":
            q_emb = self._embed_image_path(query)
        else:
            raise ValueError(f"mode must be 'text' or 'image', got '{mode}'.")

        # 2. FAISS top-N retrieval (cosine sim via inner product on L2-norm vecs)
        n = min(top_n, self.index.ntotal)
        sem_scores, faiss_rows = self.index.search(q_emb, n)
        sem_scores = sem_scores[0].tolist()
        faiss_rows = faiss_rows[0].tolist()

        # 3. Fetch physical metadata & compute R_final
        conn    = _get_db(self.db_path)
        results: List[Dict[str, Any]] = []

        for s_sem, f_row in zip(sem_scores, faiss_rows):
            if f_row < 0:        # FAISS returns -1 for padding
                continue
            try:
                row = conn.execute(
                    "SELECT * FROM frames WHERE faiss_row = ?", (int(f_row),)
                ).fetchone()
            except sqlite3.OperationalError as exc:
                print(f"  DB read error for faiss_row={f_row}: {exc}")
                continue

            if row is None:
                continue

            f_physical = float(row["fusion_score"])
            r_final    = (alpha * float(s_sem)) + (beta * f_physical)

            results.append({
                "frame_id"      : row["frame_id"],
                "video_path"    : row["video_path"],
                "timestamp_sec" : row["timestamp_sec"],
                "faiss_row"     : f_row,
                "s_semantic"    : round(float(s_sem),   4),
                "f_physical"    : round(f_physical,      4),
                "r_final"       : round(r_final,         4),
                "weapon_score"  : row["weapon_score"],
                "action_score"  : row["action_score"],
                "zone_violation": row["zone_violation"],
                "surge_score"   : row["surge_score"],
                "track_ids"     : json.loads(row["track_ids"]),
            })

        conn.close()

        results.sort(key=lambda x: x["r_final"], reverse=True)
        return results[:top_k]
