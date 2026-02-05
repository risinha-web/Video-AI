# AI/ML Engineer Interview Assignment

## Multimodal Video Search System

Build a web application that finds specific scenes in videos using natural language queries.

**Example:** Given a video and the query "person wearing a red dress", your system should return timestamps and thumbnails of all scenes containing someone in a red dress.

---

## The Challenge

Create a video search system that:
1. Accepts a video file
2. Takes natural language queries from users
3. Returns relevant video timestamps with confidence scores
4. Displays results through a web interface

### What We're Evaluating

| Criteria | Weight | Description |
|----------|--------|-------------|
| Search Accuracy | 50% | Precision@K and Recall scores |
| Performance | 30% | Latency, throughput, memory efficiency |
| Code Quality | 15% | Architecture, documentation, testing |
| Innovation | 5% | Creative approaches, UX improvements |

---

## Requirements

### Must Have
- [ ] Implement `VideoSearchInterface` in `submission/main.py`
- [ ] Provide a working web interface (Gradio template provided)
- [ ] Handle videos up to 10 minutes in length
- [ ] Return results within reasonable time (<30s for first result)

### Should Have
- [ ] Support common video formats (MP4, MOV, AVI)
- [ ] Return confidence scores with results
- [ ] Display thumbnails for matched scenes
- [ ] Handle edge cases gracefully

### Nice to Have
- [ ] Real-time search as video loads
- [ ] Batch query support
- [ ] Export results to file
- [ ] Custom UI beyond the template

---

## Getting Started

### 1. Set Up Environment

```bash
# Clone your assignment repository
git clone <your-repo-url>
cd ai-interview

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install evaluation dependencies
pip install -r requirements.txt

# Install your submission dependencies
pip install -r submission/requirements.txt
```

### 2. Get Test Videos

See [data/README.md](data/README.md) for instructions. Quick start:

```bash
# Download Big Buck Bunny (open source test video)
wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 \
    -O data/big_buck_bunny.mp4
```

### 3. Implement Your Solution

Edit `submission/main.py` to implement the `VideoSearch` class:

```python
from evaluation.interface import VideoSearchInterface, SearchResult

class VideoSearch(VideoSearchInterface):
    def load_video(self, video_path: str) -> None:
        # Load and preprocess the video
        pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Find scenes matching the query
        pass

    def get_processing_stats(self) -> dict:
        # Return performance statistics
        pass
```

### 4. Test Your Solution

```bash
# Check interface compliance
python -m evaluation.evaluate --check-interface

# Run evaluation
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/big_buck_bunny.mp4 \
    --queries ./data/sample_queries.json \
    --ground-truth ./data/ground_truth/sample_labels.json

# Launch web interface
python -m submission.app
```

---

## Submission Structure

```
submission/
├── main.py           # Your VideoSearch implementation (required)
├── app.py            # Web interface (required, can modify)
├── requirements.txt  # Your dependencies (required)
└── README.md         # Your documentation (optional but recommended)
```

You may add additional files and modules as needed.

---

## Evaluation

### Automated Testing (GitHub Classroom)

When you push, automated tests will:
1. Check interface compliance
2. Run sample evaluation
3. Report scores

### Metrics

| Metric | Weight | Excellent | Good | Needs Work |
|--------|--------|-----------|------|------------|
| Precision@10 | 25% | >0.8 | 0.5-0.8 | <0.5 |
| Recall | 25% | >0.8 | 0.5-0.8 | <0.5 |
| Time to First Result | 15% | <2s | 2-10s | >10s |
| Throughput | 15% | >15 FPS | 5-15 FPS | <5 FPS |
| Memory | 10% | <1GB | 1-2GB | >2GB |
| Robustness | 10% | Handles all cases | Minor issues | Crashes |

See [EVALUATION.md](EVALUATION.md) for detailed scoring rubric.

---

## Rules

1. **Work independently** - This is an individual assessment
2. **Use any resources** - Documentation, papers, existing libraries welcome
3. **Document your approach** - Explain your model choices and trade-offs
4. **No pre-trained end-to-end solutions** - You must implement the search logic
5. **Keep it runnable** - Evaluators must be able to run your code

---

## Timeline

This is designed as a weekend project (1-2 days of focused work).

**Suggested approach:**
- Day 1: Set up environment, choose model approach, implement core search
- Day 2: Optimize performance, handle edge cases, polish UI

---

## Resources

- [API Contract Documentation](docs/API_CONTRACT.md)
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Evaluation Details](EVALUATION.md)

---

## Questions?

If you encounter issues:
1. Check the documentation first
2. Review the FAQ in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. Contact your recruiter for clarification

Good luck!
