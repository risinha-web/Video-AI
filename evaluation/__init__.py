"""Evaluation framework for video search assignment."""

from .interface import VideoSearchInterface, SearchResult
from .metrics import evaluate_results, calculate_precision_at_k, calculate_recall
from .evaluate import run_evaluation
from .report import generate_report

__all__ = [
    "VideoSearchInterface",
    "SearchResult",
    "evaluate_results",
    "calculate_precision_at_k",
    "calculate_recall",
    "run_evaluation",
    "generate_report",
]
