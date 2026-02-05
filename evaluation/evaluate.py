#!/usr/bin/env python3
"""
Main evaluation CLI for video search assignment.

Usage:
    python -m evaluation.evaluate --submission ./submission --video ./data/test_video.mp4 \
        --queries ./data/sample_queries.json --ground-truth ./data/ground_truth/sample_labels.json \
        --output report.json

    python -m evaluation.evaluate --check-interface --submission ./submission

    python -m evaluation.evaluate --submission ./submission --quick
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .interface import VideoSearchInterface, SearchResult, validate_interface
from .metrics import evaluate_results, EvaluationResult
from .report import generate_report, print_summary


def load_submission(submission_path: str) -> Tuple[Optional[VideoSearchInterface], List[str]]:
    """Load the candidate's submission.

    Args:
        submission_path: Path to the submission directory.

    Returns:
        Tuple of (implementation instance, list of errors).
    """
    errors = []
    submission_dir = Path(submission_path)

    # Check directory exists
    if not submission_dir.exists():
        return None, [f"Submission directory not found: {submission_path}"]

    # Check main.py exists
    main_path = submission_dir / "main.py"
    if not main_path.exists():
        return None, [f"main.py not found in {submission_path}"]

    # Load the module
    try:
        spec = importlib.util.spec_from_file_location("submission_main", main_path)
        if spec is None or spec.loader is None:
            return None, [f"Could not load module from {main_path}"]

        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(submission_dir))
        spec.loader.exec_module(module)
    except Exception as e:
        return None, [f"Error loading submission: {e}\n{traceback.format_exc()}"]

    # Find the VideoSearch class
    video_search_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, VideoSearchInterface)
            and obj is not VideoSearchInterface
        ):
            video_search_class = obj
            break

    if video_search_class is None:
        # Also check for a class named VideoSearch that might not inherit properly
        if hasattr(module, "VideoSearch"):
            video_search_class = module.VideoSearch
        else:
            return None, [
                "No VideoSearchInterface implementation found in main.py. "
                "Ensure your class inherits from VideoSearchInterface."
            ]

    # Instantiate the class
    try:
        instance = video_search_class()
    except Exception as e:
        return None, [f"Error instantiating VideoSearch: {e}"]

    # Validate the interface
    validation_errors = validate_interface(instance)
    if validation_errors:
        errors.extend(validation_errors)

    return instance, errors


def load_queries(queries_path: str) -> Dict[str, dict]:
    """Load queries from JSON file.

    Returns:
        Dict mapping query_id to query dict with "text" and "difficulty" keys.
    """
    with open(queries_path) as f:
        data = json.load(f)

    queries = {}
    for q in data.get("queries", []):
        queries[q["id"]] = {
            "text": q["text"],
            "difficulty": q.get("difficulty", "medium"),
        }

    return queries


def load_ground_truth(ground_truth_path: str) -> Dict[str, List[Dict[str, int]]]:
    """Load ground truth from JSON file.

    Returns:
        Dict mapping query text to list of match dicts with "start_ms" and "end_ms".
    """
    with open(ground_truth_path) as f:
        data = json.load(f)

    ground_truth = {}
    for annotation in data.get("annotations", []):
        query = annotation["query"]
        matches = annotation.get("matches", [])
        ground_truth[query] = [
            {"start_ms": m["start_ms"], "end_ms": m["end_ms"]}
            for m in matches
        ]

    return ground_truth


def measure_memory() -> float:
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def run_evaluation(
    submission_path: str,
    video_path: Optional[str] = None,
    queries_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    quick: bool = False,
    k: int = 10,
) -> EvaluationResult:
    """Run the full evaluation pipeline.

    Args:
        submission_path: Path to submission directory.
        video_path: Path to test video.
        queries_path: Path to queries JSON.
        ground_truth_path: Path to ground truth JSON.
        quick: If True, run abbreviated evaluation.
        k: Number of results for Precision@K.

    Returns:
        EvaluationResult with all metrics.
    """
    # Load submission
    implementation, load_errors = load_submission(submission_path)

    if implementation is None:
        return EvaluationResult(
            metrics=[],
            total_score=0.0,
            query_results={},
            errors=load_errors,
        )

    errors = load_errors.copy()

    # If no video/queries provided, just validate interface
    if video_path is None or queries_path is None:
        return EvaluationResult(
            metrics=[],
            total_score=0.0,
            query_results={},
            errors=errors if errors else ["Interface validation passed."],
        )

    # Load queries and ground truth
    try:
        queries = load_queries(queries_path)
    except Exception as e:
        errors.append(f"Error loading queries: {e}")
        queries = {}

    ground_truth_by_text = {}
    if ground_truth_path:
        try:
            ground_truth_by_text = load_ground_truth(ground_truth_path)
        except Exception as e:
            errors.append(f"Error loading ground truth: {e}")

    # Quick mode uses subset of queries
    if quick and len(queries) > 3:
        queries = dict(list(queries.items())[:3])

    # Load video
    start_memory = measure_memory()
    try:
        load_start = time.time()
        implementation.load_video(video_path)
        load_time = time.time() - load_start
    except Exception as e:
        errors.append(f"Error loading video: {e}")
        return EvaluationResult(
            metrics=[],
            total_score=0.0,
            query_results={},
            errors=errors,
        )

    # Run searches
    results_by_query: Dict[str, List[SearchResult]] = {}
    latency_by_query: Dict[str, float] = {}
    query_difficulties: Dict[str, str] = {}

    # Map query text to ground truth
    ground_truth_by_id: Dict[str, List[Dict[str, int]]] = {}

    for query_id, query_data in queries.items():
        query_text = query_data["text"]
        query_difficulties[query_id] = query_data["difficulty"]

        # Get ground truth for this query
        ground_truth_by_id[query_id] = ground_truth_by_text.get(query_text, [])

        try:
            search_start = time.time()
            results = implementation.search(query_text, top_k=k)
            search_time = (time.time() - search_start) * 1000  # Convert to ms

            results_by_query[query_id] = results
            latency_by_query[query_id] = search_time
        except Exception as e:
            errors.append(f"Error searching for '{query_text}': {e}")
            results_by_query[query_id] = []
            latency_by_query[query_id] = 0

    # Get processing stats
    try:
        stats = implementation.get_processing_stats()
    except Exception as e:
        errors.append(f"Error getting stats: {e}")
        stats = {}

    # Enhance stats with measured values
    peak_memory = measure_memory()
    if "memory_mb" not in stats:
        stats["memory_mb"] = peak_memory - start_memory
    if "fps" not in stats:
        stats["fps"] = 0

    # Run evaluation
    return evaluate_results(
        results_by_query=results_by_query,
        ground_truth=ground_truth_by_id,
        query_difficulties=query_difficulties,
        processing_stats=stats,
        latency_by_query=latency_by_query,
        k=k,
    )


def check_interface_only(submission_path: str) -> bool:
    """Check if submission implements the required interface.

    Args:
        submission_path: Path to submission directory.

    Returns:
        True if interface is valid.
    """
    implementation, errors = load_submission(submission_path)

    if errors:
        print("Interface Check FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False

    if implementation is None:
        print("Interface Check FAILED: Could not load implementation")
        return False

    print("Interface Check PASSED")
    print(f"  Implementation class: {type(implementation).__name__}")

    # Check for optional methods
    if hasattr(implementation, "__doc__") and implementation.__doc__:
        print(f"  Documentation: Yes")
    else:
        print(f"  Documentation: No (consider adding a docstring)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video search submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full evaluation:
    python -m evaluation.evaluate --submission ./submission --video ./data/test.mp4 \\
        --queries ./data/sample_queries.json --ground-truth ./data/ground_truth/sample_labels.json

  Interface check only:
    python -m evaluation.evaluate --check-interface --submission ./submission

  Quick evaluation:
    python -m evaluation.evaluate --submission ./submission --video ./data/test.mp4 \\
        --queries ./data/sample_queries.json --quick
        """,
    )

    parser.add_argument(
        "--submission",
        default="./submission",
        help="Path to submission directory (default: ./submission)",
    )
    parser.add_argument(
        "--video",
        help="Path to test video file",
    )
    parser.add_argument(
        "--queries",
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output",
        help="Path to write evaluation report (JSON)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="json",
        help="Report format (default: json)",
    )
    parser.add_argument(
        "--check-interface",
        action="store_true",
        help="Only check if interface is implemented correctly",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run abbreviated evaluation (fewer queries)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of results for Precision@K (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Interface check mode
    if args.check_interface:
        success = check_interface_only(args.submission)
        sys.exit(0 if success else 1)

    # Full evaluation mode
    if args.video and args.queries:
        result = run_evaluation(
            submission_path=args.submission,
            video_path=args.video,
            queries_path=args.queries,
            ground_truth_path=args.ground_truth,
            quick=args.quick,
            k=args.k,
        )
    else:
        # Just validate interface if no video/queries
        result = run_evaluation(
            submission_path=args.submission,
            quick=args.quick,
            k=args.k,
        )

    # Print summary
    if result.metrics:
        print_summary(result)
    elif result.errors:
        print("\nEvaluation could not complete:")
        for error in result.errors:
            print(f"  - {error}")

    # Write report
    if args.output:
        report = generate_report(result, args.output, args.output_format)
        print(f"\nReport written to: {args.output}")

    # Exit with appropriate code
    if result.errors and not result.metrics:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
