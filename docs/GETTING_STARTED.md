# Getting Started Guide

This guide walks you through setting up your development environment and understanding the assignment.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- ~10GB disk space (for models and video data)
- GPU recommended but not required

## Setup

### 1. Clone Your Repository

```bash
git clone <your-assignment-repo-url>
cd ai-interview
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install evaluation framework
pip install -r requirements.txt

# Install submission dependencies (you'll update these)
pip install -r submission/requirements.txt
```

### 4. Download Test Video

```bash
# Big Buck Bunny - open source animated short
wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 \
    -O data/big_buck_bunny.mp4
```

### 5. Verify Setup

```bash
# Check interface compliance
python -m evaluation.evaluate --check-interface --submission ./submission
```

You should see: `Interface Check PASSED` (or `NotImplementedError` if methods aren't implemented yet).

---

## Project Structure

```
ai-interview/
├── README.md                  # Assignment overview
├── EVALUATION.md              # Scoring details
├── requirements.txt           # Evaluation dependencies
│
├── evaluation/                # DO NOT MODIFY
│   ├── interface.py           # API you must implement
│   ├── metrics.py             # Scoring functions
│   ├── evaluate.py            # CLI evaluation tool
│   └── report.py              # Report generation
│
├── submission/                # YOUR CODE GOES HERE
│   ├── main.py                # Implement VideoSearch class
│   ├── app.py                 # Web interface (can modify)
│   └── requirements.txt       # Your dependencies
│
├── data/
│   ├── sample_queries.json    # Test queries
│   └── ground_truth/          # Expected results
│
└── docs/
    ├── GETTING_STARTED.md     # This file
    └── API_CONTRACT.md        # Interface documentation
```

---

## Understanding the Task

### What You're Building

A video search system that:
1. Loads a video file
2. Preprocesses/indexes the video (extract frames, generate embeddings)
3. Accepts natural language queries
4. Returns timestamps of matching scenes

### The Interface

You must implement three methods in `VideoSearch`:

```python
class VideoSearch(VideoSearchInterface):
    def load_video(self, video_path: str) -> None:
        """
        Called once per video. This is where you:
        - Load the video file
        - Extract frames (all or sampled)
        - Generate embeddings for each frame
        - Build index structures
        """
        pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Called for each user query. This is where you:
        - Encode the query text
        - Compare against frame embeddings
        - Return top-k most similar frames
        """
        pass

    def get_processing_stats(self) -> dict:
        """
        Return performance metrics. Called after processing.
        Must return at least: fps, memory_mb, model_info
        """
        pass
```

### SearchResult Object

```python
@dataclass
class SearchResult:
    timestamp_ms: int          # Millisecond offset in video
    confidence: float          # 0.0 to 1.0
    frame_number: int          # Frame index
    thumbnail_path: Optional[str]  # Path to extracted frame image
```

---

## Development Workflow

### 1. Start Simple

Get a basic working solution first:
```python
def search(self, query: str, top_k: int = 10):
    # Even a random result is better than crashing
    return [SearchResult(
        timestamp_ms=0,
        confidence=0.5,
        frame_number=0,
    )]
```

### 2. Add Core Functionality

Implement actual video loading and search:
- Extract frames with OpenCV
- Generate embeddings with your chosen model
- Compare query against frame embeddings

### 3. Test Frequently

```bash
# Quick interface check
python -m evaluation.evaluate --check-interface

# Run against sample data
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/big_buck_bunny.mp4 \
    --queries ./data/sample_queries.json \
    --quick
```

### 4. Optimize

Once accuracy is good, focus on:
- Processing speed
- Memory efficiency
- Latency

### 5. Polish

- Handle edge cases
- Add documentation
- Clean up code

---

## Running the Web Interface

```bash
# Start the Gradio interface
python -m submission.app
```

Open http://localhost:7860 in your browser.

---

## FAQ

### Q: What models should I use?

Research and choose what you think is best. Consider:
- Vision-language models (CLIP, BLIP, etc.)
- Image captioning + text matching
- Object detection + semantic matching

### Q: Can I use GPU?

Yes, GPU is supported and recommended. Ensure your requirements.txt includes GPU-enabled packages if needed.

### Q: How do I handle long videos?

Consider:
- Frame sampling (e.g., 1 frame per second)
- Chunked processing
- Lazy loading

### Q: What video formats are supported?

Any format OpenCV can read: MP4, AVI, MOV, MKV, etc.

### Q: Can I modify the web interface?

Yes! The Gradio template is a starting point. You can:
- Extend it with new features
- Replace it with Flask, FastAPI, Streamlit
- Build a custom frontend

### Q: How strict is the time limit?

The evaluation will timeout after 5 minutes per video. Aim for:
- Video loading: < 2 minutes for 10-minute video
- Search query: < 10 seconds per query

### Q: Can I add additional files?

Yes, add any modules, utilities, or resources you need. Just ensure:
- `main.py` is the entry point
- `app.py` launches the web interface
- All dependencies are in `requirements.txt`

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'evaluation'"

Run from the project root directory:
```bash
cd ai-interview
python -m evaluation.evaluate ...
```

### "No VideoSearchInterface implementation found"

Ensure your class:
1. Is named `VideoSearch`
2. Inherits from `VideoSearchInterface`
3. Is defined in `submission/main.py`

### "Error loading video"

Check:
1. Video file exists at the path
2. OpenCV is installed: `pip install opencv-python`
3. Video format is supported

### Out of memory

Try:
1. Reduce batch size
2. Sample fewer frames
3. Use a smaller model
4. Process in chunks

---

## Next Steps

1. Read [API_CONTRACT.md](API_CONTRACT.md) for interface details
2. Review [EVALUATION.md](../EVALUATION.md) for scoring criteria
3. Start implementing `submission/main.py`

Good luck!
