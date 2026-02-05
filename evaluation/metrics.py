"""
Evaluation metrics for video search submissions.

Metrics are weighted as follows:
- Precision@K: 25%
- Recall: 25%
- Time-to-First-Result: 15%
- Throughput (FPS): 15%
- Memory Efficiency: 10%
- Robustness: 10%
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics

from .interface import SearchResult


# Tolerance for matching timestamps (in milliseconds)
DEFAULT_TOLERANCE_MS = 2000  # 2 seconds


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    name: str
    score: float  # Normalized 0.0 to 1.0
    raw_value: Any  # Raw metric value
    weight: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for a submission."""

    metrics: List[MetricResult]
    total_score: float
    query_results: Dict[str, Dict[str, Any]]
    errors: List[str]

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "raw_value": m.raw_value,
                    "weight": m.weight,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "query_results": self.query_results,
            "errors": self.errors,
        }


def calculate_precision_at_k(
    results: List[SearchResult],
    ground_truth: List[Dict[str, int]],
    k: int = 10,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> float:
    """Calculate Precision@K for search results.

    Args:
        results: List of SearchResult from the candidate's implementation.
        ground_truth: List of dicts with "start_ms" and "end_ms" keys.
        k: Number of top results to consider.
        tolerance_ms: Tolerance for matching timestamps.

    Returns:
        Precision@K score between 0.0 and 1.0.
    """
    if not results or not ground_truth:
        return 0.0

    top_k_results = results[:k]
    correct = 0

    for result in top_k_results:
        if _matches_any_ground_truth(result.timestamp_ms, ground_truth, tolerance_ms):
            correct += 1

    return correct / len(top_k_results)


def calculate_recall(
    results: List[SearchResult],
    ground_truth: List[Dict[str, int]],
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> float:
    """Calculate recall for search results.

    Args:
        results: List of SearchResult from the candidate's implementation.
        ground_truth: List of dicts with "start_ms" and "end_ms" keys.
        tolerance_ms: Tolerance for matching timestamps.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not ground_truth:
        return 1.0  # No ground truth to find
    if not results:
        return 0.0

    found = 0
    for gt in ground_truth:
        gt_center = (gt["start_ms"] + gt["end_ms"]) // 2
        for result in results:
            if abs(result.timestamp_ms - gt_center) <= tolerance_ms + (gt["end_ms"] - gt["start_ms"]) // 2:
                found += 1
                break

    return found / len(ground_truth)


def calculate_time_to_first_result(
    results: List[SearchResult],
    ground_truth: List[Dict[str, int]],
    latency_ms: float,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> float:
    """Calculate time to first correct result.

    Args:
        results: List of SearchResult, sorted by confidence.
        ground_truth: List of dicts with "start_ms" and "end_ms" keys.
        latency_ms: Total time to return results in milliseconds.
        tolerance_ms: Tolerance for matching timestamps.

    Returns:
        Score between 0.0 and 1.0 based on ranking of first correct result.
    """
    if not results or not ground_truth:
        return 0.0

    for i, result in enumerate(results):
        if _matches_any_ground_truth(result.timestamp_ms, ground_truth, tolerance_ms):
            # Score based on position: 1.0 for first result, decreasing linearly
            return max(0.0, 1.0 - (i * 0.1))

    return 0.0  # No correct result found


def calculate_throughput_score(fps: float) -> float:
    """Calculate score based on processing throughput.

    Scoring:
    - 30+ FPS: 1.0 (real-time or better)
    - 15-30 FPS: 0.75-1.0
    - 5-15 FPS: 0.5-0.75
    - 1-5 FPS: 0.25-0.5
    - <1 FPS: 0.0-0.25

    Args:
        fps: Frames processed per second.

    Returns:
        Score between 0.0 and 1.0.
    """
    if fps >= 30:
        return 1.0
    elif fps >= 15:
        return 0.75 + 0.25 * (fps - 15) / 15
    elif fps >= 5:
        return 0.5 + 0.25 * (fps - 5) / 10
    elif fps >= 1:
        return 0.25 + 0.25 * (fps - 1) / 4
    else:
        return max(0.0, 0.25 * fps)


def calculate_memory_score(memory_mb: float) -> float:
    """Calculate score based on memory efficiency.

    Scoring:
    - <512MB: 1.0
    - 512MB-1GB: 0.75-1.0
    - 1GB-2GB: 0.5-0.75
    - 2GB-4GB: 0.25-0.5
    - >4GB: 0.0-0.25

    Args:
        memory_mb: Peak memory usage in megabytes.

    Returns:
        Score between 0.0 and 1.0.
    """
    if memory_mb <= 512:
        return 1.0
    elif memory_mb <= 1024:
        return 0.75 + 0.25 * (1024 - memory_mb) / 512
    elif memory_mb <= 2048:
        return 0.5 + 0.25 * (2048 - memory_mb) / 1024
    elif memory_mb <= 4096:
        return 0.25 + 0.25 * (4096 - memory_mb) / 2048
    else:
        return max(0.0, 0.25 * (8192 - memory_mb) / 4096)


def calculate_robustness_score(
    results_by_difficulty: Dict[str, List[Dict[str, Any]]]
) -> float:
    """Calculate robustness score based on performance across difficulties.

    Args:
        results_by_difficulty: Dict mapping difficulty to list of per-query results.
            Each result should have "precision" and "recall" keys.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not results_by_difficulty:
        return 0.0

    difficulty_scores = []
    difficulty_weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    for difficulty, results in results_by_difficulty.items():
        if not results:
            continue

        avg_precision = statistics.mean(r.get("precision", 0) for r in results)
        avg_recall = statistics.mean(r.get("recall", 0) for r in results)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)

        weight = difficulty_weights.get(difficulty, 0.33)
        difficulty_scores.append(f1 * weight)

    if not difficulty_scores:
        return 0.0

    # Normalize by total weight used
    total_weight = sum(
        difficulty_weights.get(d, 0.33)
        for d in results_by_difficulty.keys()
        if results_by_difficulty[d]
    )
    return sum(difficulty_scores) / total_weight if total_weight > 0 else 0.0


def evaluate_results(
    results_by_query: Dict[str, List[SearchResult]],
    ground_truth: Dict[str, List[Dict[str, int]]],
    query_difficulties: Dict[str, str],
    processing_stats: dict,
    latency_by_query: Dict[str, float],
    k: int = 10,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> EvaluationResult:
    """Run full evaluation on submission results.

    Args:
        results_by_query: Dict mapping query_id to list of SearchResult.
        ground_truth: Dict mapping query_id to list of ground truth matches.
        query_difficulties: Dict mapping query_id to difficulty level.
        processing_stats: Dict from get_processing_stats().
        latency_by_query: Dict mapping query_id to search latency in ms.
        k: Number of results for Precision@K.
        tolerance_ms: Timestamp matching tolerance.

    Returns:
        EvaluationResult with all metrics and scores.
    """
    errors = []
    query_results = {}
    results_by_difficulty: Dict[str, List[Dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }

    # Calculate per-query metrics
    all_precisions = []
    all_recalls = []
    all_ttfr = []

    for query_id, results in results_by_query.items():
        gt = ground_truth.get(query_id, [])
        latency = latency_by_query.get(query_id, 0)
        difficulty = query_difficulties.get(query_id, "medium")

        precision = calculate_precision_at_k(results, gt, k, tolerance_ms)
        recall = calculate_recall(results, gt, tolerance_ms)
        ttfr = calculate_time_to_first_result(results, gt, latency, tolerance_ms)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_ttfr.append(ttfr)

        query_result = {
            "precision": precision,
            "recall": recall,
            "time_to_first_result": ttfr,
            "num_results": len(results),
            "latency_ms": latency,
        }
        query_results[query_id] = query_result
        results_by_difficulty[difficulty].append(query_result)

    # Calculate aggregate metrics
    avg_precision = statistics.mean(all_precisions) if all_precisions else 0.0
    avg_recall = statistics.mean(all_recalls) if all_recalls else 0.0
    avg_ttfr = statistics.mean(all_ttfr) if all_ttfr else 0.0

    fps = processing_stats.get("fps", 0)
    memory_mb = processing_stats.get("memory_mb", 0)

    throughput_score = calculate_throughput_score(fps)
    memory_score = calculate_memory_score(memory_mb)
    robustness_score = calculate_robustness_score(results_by_difficulty)

    # Build metrics list with weights
    metrics = [
        MetricResult(
            name="Precision@K",
            score=avg_precision,
            raw_value=avg_precision,
            weight=0.25,
            details={"k": k, "per_query": {q: r["precision"] for q, r in query_results.items()}},
        ),
        MetricResult(
            name="Recall",
            score=avg_recall,
            raw_value=avg_recall,
            weight=0.25,
            details={"per_query": {q: r["recall"] for q, r in query_results.items()}},
        ),
        MetricResult(
            name="Time-to-First-Result",
            score=avg_ttfr,
            raw_value=avg_ttfr,
            weight=0.15,
            details={"per_query": {q: r["time_to_first_result"] for q, r in query_results.items()}},
        ),
        MetricResult(
            name="Throughput",
            score=throughput_score,
            raw_value=fps,
            weight=0.15,
            details={"fps": fps, "threshold_realtime": 30},
        ),
        MetricResult(
            name="Memory Efficiency",
            score=memory_score,
            raw_value=memory_mb,
            weight=0.10,
            details={"memory_mb": memory_mb, "threshold_excellent": 512},
        ),
        MetricResult(
            name="Robustness",
            score=robustness_score,
            raw_value=robustness_score,
            weight=0.10,
            details={"by_difficulty": {d: len(r) for d, r in results_by_difficulty.items()}},
        ),
    ]

    # Calculate weighted total
    total_score = sum(m.score * m.weight for m in metrics)

    return EvaluationResult(
        metrics=metrics,
        total_score=total_score,
        query_results=query_results,
        errors=errors,
    )


def _matches_any_ground_truth(
    timestamp_ms: int,
    ground_truth: List[Dict[str, int]],
    tolerance_ms: int,
) -> bool:
    """Check if a timestamp matches any ground truth interval."""
    for gt in ground_truth:
        start = gt["start_ms"] - tolerance_ms
        end = gt["end_ms"] + tolerance_ms
        if start <= timestamp_ms <= end:
            return True
    return False
