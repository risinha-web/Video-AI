"""
Video Search Implementation using CLIP for multimodal scene search.

Approach:
  - Model: OpenAI CLIP ViT-B/32 via HuggingFace Transformers
  - Indexing: sample 1 frame/second, encode with CLIP vision encoder in batches
  - Search: encode text query with CLIP text encoder, rank frames by cosine similarity
  - Deduplication: non-maximum suppression with a 2-second minimum gap between results
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Add parent directory to path for evaluation imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.interface import VideoSearchInterface, SearchResult


class VideoSearch(VideoSearchInterface):
    """
    CLIP-based video search.

    Frames are sampled at 1 FPS, embedded with CLIP's vision encoder,
    and stored as normalised vectors for fast cosine-similarity search
    against natural-language text queries.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"
    SAMPLE_INTERVAL_SECONDS = 1.0  # one frame per second
    BATCH_SIZE = 64                 # frames per CLIP inference call
    MIN_RESULT_GAP_MS = 1000       # minimum gap between returned results (ms)

    def __init__(self):
        """Load CLIP model and initialise state."""
        self.video_path: Optional[str] = None
        self.video_fps: float = 0.0
        self.frame_numbers: List[int] = []
        self.timestamps_ms: List[int] = []
        self.embeddings: Optional[np.ndarray] = None  # shape [N, 512]
        self.thumbnail_dir: Optional[str] = None

        self.stats = {
            "fps": 0.0,
            "memory_mb": 0.0,
            "model_info": {},
            "index_time_seconds": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.half()  # fp16 on GPU for ~2x throughput

        self.stats["model_info"] = {
            "name": "CLIP",
            "version": "ViT-B/32",
            "embedding_dim": 512,
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return L2-normalised image embeddings for a batch of PIL images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            feats = self.model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def _embed_text(self, text: str) -> np.ndarray:
        """Return L2-normalised text embedding for a query string.

        Uses the "a photo of X" prompt template, which matches CLIP's training
        distribution and improves matching on hard/abstract queries.
        """
        prompt = f"a photo of {text}"
        inputs = self.processor(
            text=[prompt], return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            text_out = self.model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            feats = self.model.text_projection(text_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()[0]

    def _memory_mb(self) -> float:
        """Return current process RSS in MB."""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # ------------------------------------------------------------------
    # VideoSearchInterface implementation
    # ------------------------------------------------------------------

    def load_video(self, video_path: str) -> None:
        """Load, sample, and index a video file.

        Samples one frame per second, encodes each through CLIP's vision
        encoder (in batches of BATCH_SIZE), and stores the resulting
        normalised embeddings for later search.

        Args:
            video_path: Path to the video file (mp4, avi, mov, etc.)

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the file cannot be opened as a video.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # How many raw frames to skip between samples
        sample_every = max(1, int(round(self.video_fps * self.SAMPLE_INTERVAL_SECONDS)))

        # Temp directory persists between searches so thumbnails remain valid
        self.thumbnail_dir = tempfile.mkdtemp(prefix="video_search_")

        # Reset index state
        self.frame_numbers = []
        self.timestamps_ms = []
        embeddings_list: List[np.ndarray] = []

        batch_images: List[Image.Image] = []
        batch_frame_idxs: List[int] = []

        frame_idx = 0
        while True:
            # grab() advances the capture without decoding — much faster than read()
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                ret, bgr = cap.retrieve()
                if not ret:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                # Save a small thumbnail for the UI
                thumb = pil.copy()
                thumb.thumbnail((320, 240), Image.LANCZOS)
                thumb_path = os.path.join(
                    self.thumbnail_dir, f"frame_{frame_idx:07d}.jpg"
                )
                thumb.save(thumb_path, quality=75, optimize=True)

                batch_images.append(pil)
                batch_frame_idxs.append(frame_idx)

                # Flush batch when full
                if len(batch_images) == self.BATCH_SIZE:
                    embeddings_list.append(self._embed_images(batch_images))
                    for fi in batch_frame_idxs:
                        self.frame_numbers.append(fi)
                        self.timestamps_ms.append(int(fi * 1000 / self.video_fps))
                    batch_images = []
                    batch_frame_idxs = []

            frame_idx += 1

        # Flush remaining partial batch
        if batch_images:
            embeddings_list.append(self._embed_images(batch_images))
            for fi in batch_frame_idxs:
                self.frame_numbers.append(fi)
                self.timestamps_ms.append(int(fi * 1000 / self.video_fps))

        cap.release()

        self.embeddings = (
            np.vstack(embeddings_list) if embeddings_list else np.empty((0, 512))
        )
        self.video_path = video_path

        elapsed = time.time() - start_time
        sampled = len(self.frame_numbers)

        self.stats.update(
            {
                "fps": sampled / elapsed if elapsed > 0 else 0.0,
                "memory_mb": self._memory_mb(),
                "index_time_seconds": elapsed,
                "total_frames": total_frames,
                "sampled_frames": sampled,
            }
        )

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the indexed video for scenes matching the query.

        Encodes the query with CLIP's text encoder and ranks all indexed
        frames by cosine similarity. Applies non-maximum suppression so
        returned results are spread at least MIN_RESULT_GAP_MS apart.

        Args:
            query: Natural language description of the scene to find.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult sorted by confidence (descending).

        Raises:
            RuntimeError: If load_video() has not been called.
            ValueError: If the query is empty.
        """
        if self.video_path is None:
            raise RuntimeError("Call load_video() before search().")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Encode query and compute cosine similarity against all frame embeddings
        # (embeddings are already L2-normalised, so dot product == cosine similarity)
        query_vec = self._embed_text(query.strip())     # [512]
        sims = self.embeddings @ query_vec              # [N], range ≈ [-1, 1]

        sorted_idxs = np.argsort(sims)[::-1]

        results: List[SearchResult] = []
        used_ts: List[int] = []

        for idx in sorted_idxs:
            if len(results) >= top_k:
                break

            ts = self.timestamps_ms[idx]

            # Non-maximum suppression: skip frames too close to an accepted result
            if any(abs(ts - u) < self.MIN_RESULT_GAP_MS for u in used_ts):
                continue

            # Clip cosine similarity to [0, 1] for the confidence field
            confidence = float(np.clip(sims[idx], 0.0, 1.0))

            thumb_path = os.path.join(
                self.thumbnail_dir, f"frame_{self.frame_numbers[idx]:07d}.jpg"
            )
            results.append(
                SearchResult(
                    timestamp_ms=ts,
                    confidence=confidence,
                    frame_number=self.frame_numbers[idx],
                    thumbnail_path=thumb_path if os.path.exists(thumb_path) else None,
                )
            )
            used_ts.append(ts)

        return results

    def get_processing_stats(self) -> dict:
        """Return performance statistics populated during load_video()."""
        return self.stats
