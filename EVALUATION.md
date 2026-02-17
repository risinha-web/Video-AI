# Evaluation Criteria

This document details how submissions are scored.

## Overview

Your submission is evaluated across four main areas:

| Area | Weight | Description |
|------|--------|-------------|
| Search Accuracy | 50% | How well your system finds relevant scenes |
| Performance | 30% | Speed and resource efficiency |
| Code Quality | 15% | Architecture, documentation, testing |
| Innovation | 5% | Creative solutions and UX improvements |

---

## Search Accuracy (50%)

### Precision@K (25%)

Measures what fraction of your top K results are correct.

```
Precision@K = (correct results in top K) / K
```

| Score | Precision@10 |
|-------|-------------|
| 100%  | ≥ 0.90 |
| 75%   | 0.70 - 0.89 |
| 50%   | 0.50 - 0.69 |
| 25%   | 0.30 - 0.49 |
| 0%    | < 0.30 |

### Recall (25%)

Measures what fraction of all relevant scenes you found.

```
Recall = (relevant scenes found) / (total relevant scenes)
```

| Score | Recall |
|-------|--------|
| 100%  | ≥ 0.90 |
| 75%   | 0.70 - 0.89 |
| 50%   | 0.50 - 0.69 |
| 25%   | 0.30 - 0.49 |
| 0%    | < 0.30 |

### Matching Tolerance

A result is considered correct if it falls within **2 seconds** of a ground truth segment.

---

## Performance (30%)

### Time to First Result (15%)

How quickly you return the first correct result after a query.

| Score | Time |
|-------|------|
| 100%  | < 1s |
| 75%   | 1 - 3s |
| 50%   | 3 - 10s |
| 25%   | 10 - 30s |
| 0%    | > 30s |

### Throughput (15%)

Frames processed per second during video indexing.

| Score | FPS |
|-------|-----|
| 100%  | ≥ 30 (real-time) |
| 75%   | 15 - 29 |
| 50%   | 5 - 14 |
| 25%   | 1 - 4 |
| 0%    | < 1 |

---

## Resource Efficiency (10%)

### Memory Usage

Peak memory during video processing.

| Score | Memory |
|-------|--------|
| 100%  | < 512 MB |
| 75%   | 512 MB - 1 GB |
| 50%   | 1 - 2 GB |
| 25%   | 2 - 4 GB |
| 0%    | > 4 GB |

---

## Robustness (10%)

How well your system handles different query types.

### Query Difficulty Levels

| Level | Description | Examples |
|-------|-------------|----------|
| Easy | Simple objects or actions | "a car", "someone walking" |
| Medium | Combined attributes | "red car", "person running fast" |
| Hard | Abstract or emotional | "happy moment", "tense scene" |

Your score is weighted by difficulty:
- Easy queries: 20% of robustness score
- Medium queries: 30% of robustness score
- Hard queries: 50% of robustness score

---

## Code Quality (15%)

Evaluated manually by reviewers.

### Architecture (5%)

- Clean separation of concerns
- Appropriate use of abstractions
- Sensible module organization
- Error handling

### Documentation (5%)

- Clear README explaining your approach
- Comments on non-obvious code
- Docstrings for public methods
- Model choice justification

### Testing (5%)

- Unit tests for core logic
- Integration tests
- Edge case handling
- Reproducibility

---

## Innovation (5%)

Bonus points for going beyond requirements:

- Novel approach to the problem
- Significant UX improvements
- Exceptional performance optimization
- Creative feature additions
- Elegant technical solutions

---

## Scoring Formula

```
Total Score =
    (Precision * 0.25) +
    (Recall * 0.25) +
    (TTFR * 0.15) +
    (Throughput * 0.15) +
    (Memory * 0.10) +
    (Robustness * 0.10) +
    (Code Quality * 0.15) +  # Manual evaluation
    (Innovation * 0.05)      # Manual evaluation
```

Note: Automated evaluation covers 70% of the score. Manual review adds 20%.

---

## Running Evaluation Locally

```bash
# Full evaluation
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/test_video.mp4 \
    --queries ./data/sample_queries.json \
    --ground-truth ./data/ground_truth/sample_labels.json \
    --output report.json

# Quick evaluation (subset of queries)
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/test_video.mp4 \
    --queries ./data/sample_queries.json \
    --quick

# Generate markdown report
python -m evaluation.evaluate \
    --submission ./submission \
    --video ./data/test_video.mp4 \
    --queries ./data/sample_queries.json \
    --ground-truth ./data/ground_truth/sample_labels.json \
    --output report.md \
    --output-format markdown
```

---

## Example Report

```
==================================================
EVALUATION SUMMARY
==================================================

Total Score: 72.50%

Metrics:
  Precision@K               [################----] 80.00%
  Recall                    [##############------] 70.00%
  Time-to-First-Result      [##################--] 90.00%
  Throughput                [############--------] 60.00%
  Memory Efficiency         [################----] 80.00%
  Robustness                [##########----------] 50.00%
==================================================
```

---

## Tips for High Scores

1. **Accuracy First**: Focus on search quality before optimizing speed
2. **Profile Early**: Identify bottlenecks before optimizing
3. **Test Edge Cases**: Empty queries, very short videos, unusual formats
4. **Document Choices**: Explain why you chose your approach
5. **Keep It Simple**: A working simple solution beats a broken complex one
