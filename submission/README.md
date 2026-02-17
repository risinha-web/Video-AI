# Submission Instructions

This directory contains your video search implementation.

## Required Files

- **main.py** - Your implementation of `VideoSearch` class (required)
- **app.py** - Web interface for your solution (required, can be modified/replaced)
- **requirements.txt** - Your dependencies (update with packages you use)

## Implementation Checklist

### 1. Implement `VideoSearch` class in `main.py`

Your class must inherit from `VideoSearchInterface` and implement:

```python
class VideoSearch(VideoSearchInterface):
    def load_video(self, video_path: str) -> None:
        """Load and preprocess the video."""
        pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Find scenes matching the query."""
        pass

    def get_processing_stats(self) -> dict:
        """Return performance statistics."""
        pass
```

### 2. Update requirements.txt

Add all packages your solution needs. The evaluator will run:
```bash
pip install -r submission/requirements.txt
```

### 3. Wire up the web interface

The provided `app.py` is a Gradio template. You can:
- Extend it with additional features
- Replace it with Flask, FastAPI, or another framework
- Ensure it launches on port 7860

## Testing Your Submission

1. **Interface check:**
   ```bash
   python -m evaluation.evaluate --check-interface --submission ./submission
   ```

2. **Local evaluation:**
   ```bash
   python -m evaluation.evaluate \
       --submission ./submission \
       --video ./data/your_video.mp4 \
       --queries ./data/sample_queries.json \
       --ground-truth ./data/ground_truth/sample_labels.json
   ```

3. **Run the web interface:**
   ```bash
   python -m submission.app
   ```
   Then open http://localhost:7860

## What We're Looking For

See [EVALUATION.md](../EVALUATION.md) for detailed scoring criteria.

**Key areas:**
- Search accuracy (Precision@K, Recall)
- Performance (latency, throughput, memory)
- Code quality and architecture
- Documentation

## Tips

- Start with a working baseline, then optimize
- Consider the trade-off between accuracy and speed
- Document your model choices and why you made them
- Include error handling for edge cases
- Test with various query types (objects, actions, attributes)

## Submission

When you're done:
1. Ensure all tests pass locally
2. Commit and push to your private repository
3. Add `vlt-ai` as a collaborator so we can review:
   ```bash
   gh repo edit --add-collaborator vlt-ai
   ```
   Or go to your repo → Settings → Collaborators → Add `vlt-ai`
4. Send your repo link to your recruiter

Good luck!
