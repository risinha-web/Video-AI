# API Contract

This document specifies the interface your implementation must follow.

## Overview

Your implementation must:
1. Create a class that inherits from `VideoSearchInterface`
2. Implement all three required methods
3. Return properly formatted `SearchResult` objects

---

## VideoSearchInterface

Location: `evaluation/interface.py`

```python
from abc import ABC, abstractmethod
from typing import List

class VideoSearchInterface(ABC):
    @abstractmethod
    def load_video(self, video_path: str) -> None:
        """Load and optionally preprocess a video."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for scenes matching the natural language query."""
        pass

    @abstractmethod
    def get_processing_stats(self) -> dict:
        """Return stats: fps, memory_mb, model_info."""
        pass
```

---

## Method Specifications

### load_video(video_path: str) -> None

**Purpose:** Load and preprocess a video for searching.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| video_path | str | Absolute or relative path to video file |

**Behavior:**
- Load the video file
- Extract frames (all or sampled)
- Generate embeddings or features
- Build index structures for efficient search
- Store state for subsequent `search()` calls

**Exceptions:**
| Exception | When |
|-----------|------|
| FileNotFoundError | Video file doesn't exist |
| ValueError | Unsupported video format |
| MemoryError | Not enough memory to process |

**Example:**
```python
searcher = VideoSearch()
searcher.load_video("/path/to/video.mp4")
```

---

### search(query: str, top_k: int = 10) -> List[SearchResult]

**Purpose:** Find video scenes matching a natural language query.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| query | str | required | Natural language description |
| top_k | int | 10 | Maximum results to return |

**Returns:** `List[SearchResult]` sorted by confidence (descending)

**Behavior:**
- Encode the query text
- Compare against preprocessed video frames
- Return up to `top_k` most relevant results
- Results should not overlap significantly in time

**Exceptions:**
| Exception | When |
|-----------|------|
| RuntimeError | `load_video()` not called yet |
| ValueError | Empty query string |

**Example:**
```python
results = searcher.search("person wearing red dress", top_k=5)
for r in results:
    print(f"Time: {r.timestamp_ms}ms, Confidence: {r.confidence:.2f}")
```

---

### get_processing_stats() -> dict

**Purpose:** Return performance statistics about video processing.

**Returns:** Dictionary with the following keys (at minimum):

| Key | Type | Description |
|-----|------|-------------|
| fps | float | Frames processed per second during indexing |
| memory_mb | float | Peak memory usage in megabytes |
| model_info | dict | Information about model(s) used |

**Optional keys:**
| Key | Type | Description |
|-----|------|-------------|
| index_time_seconds | float | Time to index the video |
| total_frames | int | Number of frames in video |
| sampled_frames | int | Number of frames actually processed |
| embedding_dim | int | Dimension of embeddings |

**Example:**
```python
stats = searcher.get_processing_stats()
print(f"Processing speed: {stats['fps']:.1f} FPS")
print(f"Memory usage: {stats['memory_mb']:.0f} MB")
print(f"Model: {stats['model_info'].get('name', 'unknown')}")
```

---

## SearchResult

Location: `evaluation/interface.py`

```python
@dataclass
class SearchResult:
    timestamp_ms: int          # Millisecond offset in video
    confidence: float          # 0.0 to 1.0
    frame_number: int          # Frame index
    thumbnail_path: Optional[str] = None  # Path to extracted frame
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| timestamp_ms | int | Yes | Millisecond offset from video start |
| confidence | float | Yes | Confidence score (0.0 to 1.0) |
| frame_number | int | Yes | Zero-indexed frame number |
| thumbnail_path | str | No | Path to saved frame image |

### Validation

- `timestamp_ms` must be ≥ 0
- `confidence` must be between 0.0 and 1.0
- `frame_number` must be ≥ 0

**Example:**
```python
result = SearchResult(
    timestamp_ms=12500,      # 12.5 seconds into video
    confidence=0.87,
    frame_number=300,        # 300th frame
    thumbnail_path="/tmp/frame_300.jpg"
)
```

---

## Implementation Template

```python
from typing import List
from evaluation.interface import VideoSearchInterface, SearchResult

class VideoSearch(VideoSearchInterface):
    def __init__(self):
        self.video_path = None
        self.embeddings = None
        self.frame_timestamps = []
        self.stats = {}

    def load_video(self, video_path: str) -> None:
        import cv2
        import time

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = time.time()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract frames
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            self.frame_timestamps.append(int(frame_idx * 1000 / fps))
            frame_idx += 1

        cap.release()

        # Generate embeddings (your model here)
        self.embeddings = self._generate_embeddings(frames)

        self.video_path = video_path
        self.stats = {
            "fps": frame_count / (time.time() - start_time),
            "memory_mb": self._get_memory_usage(),
            "model_info": {"name": "your_model_name"},
            "total_frames": frame_count,
            "index_time_seconds": time.time() - start_time,
        }

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Must call load_video() first")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Encode query and find similar frames
        query_embedding = self._encode_query(query)
        similarities = self._compute_similarities(query_embedding)

        # Get top-k indices
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )[:top_k]

        return [
            SearchResult(
                timestamp_ms=self.frame_timestamps[i],
                confidence=float(similarities[i]),
                frame_number=i,
            )
            for i in top_indices
        ]

    def get_processing_stats(self) -> dict:
        return self.stats

    def _generate_embeddings(self, frames):
        # Implement with your chosen model
        raise NotImplementedError

    def _encode_query(self, query: str):
        # Implement with your chosen model
        raise NotImplementedError

    def _compute_similarities(self, query_embedding):
        # Implement similarity computation
        raise NotImplementedError

    def _get_memory_usage(self):
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
```

---

## Best Practices

### 1. Handle Edge Cases

```python
def search(self, query: str, top_k: int = 10):
    # Handle edge cases
    if top_k <= 0:
        return []

    results = self._do_search(query, top_k)

    # Ensure we don't return more than available
    return results[:min(top_k, len(results))]
```

### 2. Avoid Overlapping Results

If consecutive frames match, consolidate:
```python
def _deduplicate_results(self, results, min_gap_ms=1000):
    if not results:
        return []

    deduplicated = [results[0]]
    for result in results[1:]:
        if result.timestamp_ms - deduplicated[-1].timestamp_ms >= min_gap_ms:
            deduplicated.append(result)

    return deduplicated
```

### 3. Generate Thumbnails

```python
def _save_thumbnail(self, frame, frame_number, output_dir="/tmp/thumbnails"):
    import cv2
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True)
    path = f"{output_dir}/frame_{frame_number}.jpg"
    cv2.imwrite(path, frame)
    return path
```

### 4. Track Memory Usage

```python
def _get_memory_usage(self):
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
```

---

## Validation

The evaluation framework validates your implementation:

```python
from evaluation.interface import validate_interface

errors = validate_interface(your_implementation)
if errors:
    print("Validation failed:")
    for error in errors:
        print(f"  - {error}")
```

Validation checks:
- Class inherits from `VideoSearchInterface`
- All required methods are implemented
- Methods are callable

---

## Questions?

Review the example code in `submission/main.py` and the evaluation tests in `evaluation/evaluate.py`.
