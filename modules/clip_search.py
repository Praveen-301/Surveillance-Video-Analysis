import torch, os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPSearch:
    MODEL_NAME = "openai/clip-vit-base-patch32"  # CPU-friendly

    def __init__(self, frames_dir="saved_frames"):
        # Resolve path relative to project root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.frames_dir = os.path.join(self.base_dir, frames_dir)
        self.INDEX_FILE = os.path.join(self.base_dir, "clip_index.pt")

        print(f"Loading CLIP ({self.MODEL_NAME})...")
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self.proc  = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model.eval()

        self.frame_paths   = []
        self.frame_features= None

        # Load saved index if exists, else build fresh
        if os.path.exists(self.INDEX_FILE):
            self._load_index()
        else:
            self._build_index()
            if self.frame_features is not None:
                self._save_index()

    def _build_index(self):
        if not os.path.exists(self.frames_dir):
            print(f"Directory {self.frames_dir} not found.")
            return

        paths = sorted([
            os.path.join(self.frames_dir, f)
            for f in os.listdir(self.frames_dir)
            if f.endswith(".jpg")
        ])
        if not paths:
            print("No frames found in saved_frames/")
            print("Run pipeline.py first to generate frames.")
            return

        print(f"Indexing {len(paths)} frames... (first time only)")
        features   = []
        batch_size = 8

        for i in range(0, len(paths), batch_size):
            batch  = paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB").resize((224,224))
                      for p in batch]
            inputs = self.proc(
                images=images,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Ensure we have a tensor (some versions return a dict or object)
                if not isinstance(outputs, torch.Tensor):
                    feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
                else:
                    feats = outputs
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats)

            done = min(i+batch_size, len(paths))
            if done % 50 == 0 or done == len(paths):
                print(f"  {done}/{len(paths)} frames indexed")

        self.frame_paths    = paths
        self.frame_features = torch.cat(features, dim=0)
        print(f"Indexing complete — {len(paths)} frames ready")

    def _save_index(self):
        torch.save({
            "features"    : self.frame_features,
            "frame_paths" : self.frame_paths,
        }, self.INDEX_FILE)
        print(f"Index saved to {self.INDEX_FILE}")

    def _load_index(self):
        data = torch.load(self.INDEX_FILE, map_location="cpu")
        self.frame_features = data["features"]
        self.frame_paths    = data["frame_paths"]
        print(f"Index loaded — {len(self.frame_paths)} frames")

    def rebuild_index(self):
        """Call this after running pipeline on a new video"""
        if os.path.exists(self.INDEX_FILE):
            os.remove(self.INDEX_FILE)
        self._build_index()
        if self.frame_features is not None:
            self._save_index()

    def search(self, query_text, top_k=5, fps=25):
        if self.frame_features is None or len(self.frame_paths) == 0:
            return []

        # Encode text
        inputs = self.proc(
            text=[query_text],
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if not isinstance(outputs, torch.Tensor):
                text_feat = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
            else:
                text_feat = outputs
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        return self._rank_frames(text_feat, top_k, fps)

    def search_by_image(self, query_image, top_k=5, fps=25):
        """
        Find frames visually similar to a query image (e.g. a person's photo).

        Args:
            query_image : PIL.Image or str path to an image file
            top_k       : number of results to return
            fps         : video fps for timestamp calculation

        Returns:
            list of dicts with frame_path, frame_num, timestamp, similarity
        """
        if self.frame_features is None or len(self.frame_paths) == 0:
            return []

        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert("RGB")
        else:
            query_image = query_image.convert("RGB")

        inputs = self.proc(images=query_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if not isinstance(outputs, torch.Tensor):
                img_feat = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
            else:
                img_feat = outputs
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        return self._rank_frames(img_feat, top_k, fps)

    def _rank_frames(self, query_feat, top_k, fps):
        """Shared cosine-similarity ranking used by both text and image search."""
        sims = (self.frame_features @ query_feat.T).squeeze()

        if len(self.frame_paths) == 1:
            top_idx    = torch.tensor([0])
            sim_scores = torch.tensor([sims.item()])
        else:
            top_k = min(top_k, len(self.frame_paths))
            res        = torch.topk(sims, top_k)
            top_idx    = res.indices
            sim_scores = res.values

        results = []
        for i, idx in enumerate(top_idx):
            path      = self.frame_paths[int(idx)]
            frame_num = int(
                os.path.basename(path)
                .replace("frame_", "")
                .replace(".jpg", "")
            )
            secs = frame_num / fps
            results.append({
                "frame_path" : path,
                "frame_num"  : frame_num,
                "timestamp"  : f"{int(secs//60):02d}:{int(secs%60):02d}",
                "similarity" : round(float(sim_scores[i]), 4),
            })

        return results
