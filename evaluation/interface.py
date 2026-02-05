"""
API Contract for Video Search Implementation.

Candidates MUST implement the VideoSearchInterface class.
The evaluation framework will use this interface to test submissions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    """A single search result representing a matched scene in the video.

    Attributes:
        timestamp_ms: Millisecond offset from the start of the video.
        confidence: Confidence score between 0.0 and 1.0.
        frame_number: The frame index in the video.
        thumbnail_path: Optional path to an extracted frame image.
    """

    timestamp_ms: int
    confidence: float
    frame_number: int
    thumbnail_path: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.timestamp_ms < 0:
            raise ValueError(f"timestamp_ms must be non-negative, got {self.timestamp_ms}")
        if self.frame_number < 0:
            raise ValueError(f"frame_number must be non-negative, got {self.frame_number}")


@dataclass
class ProcessingStats:
    """Statistics about video processing performance.

    Attributes:
        fps: Frames processed per second during indexing.
        memory_mb: Peak memory usage in megabytes.
        model_info: Information about the model(s) used.
        index_time_seconds: Time taken to index the video.
        total_frames: Total number of frames in the video.
    """

    fps: float
    memory_mb: float
    model_info: dict = field(default_factory=dict)
    index_time_seconds: float = 0.0
    total_frames: int = 0


class VideoSearchInterface(ABC):
    """Abstract base class for video search implementations.

    Candidates must create a class that inherits from this interface
    and implements all abstract methods.

    Example:
        ```python
        from evaluation.interface import VideoSearchInterface, SearchResult

        class VideoSearch(VideoSearchInterface):
            def load_video(self, video_path: str) -> None:
                # Your implementation here
                pass

            def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
                # Your implementation here
                return []

            def get_processing_stats(self) -> dict:
                # Your implementation here
                return {"fps": 0, "memory_mb": 0, "model_info": {}}
        ```
    """

    @abstractmethod
    def load_video(self, video_path: str) -> None:
        """Load and optionally preprocess a video for searching.

        This method should:
        - Load the video file from the given path
        - Extract frames and/or features as needed
        - Build any index structures required for efficient search

        Args:
            video_path: Path to the video file (mp4, avi, mov, etc.)

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            ValueError: If the video format is not supported.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for scenes matching the natural language query.

        Args:
            query: Natural language description of the scene to find.
                   Examples: "person wearing a red dress", "car driving fast"
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects, sorted by confidence (descending).
            Results should not overlap significantly in time.

        Raises:
            RuntimeError: If load_video() hasn't been called yet.
        """
        pass

    @abstractmethod
    def get_processing_stats(self) -> dict:
        """Return statistics about the processing performance.

        Returns:
            Dictionary containing at minimum:
            - fps: Frames processed per second during indexing
            - memory_mb: Peak memory usage in megabytes
            - model_info: Dict with model name, version, parameters, etc.

        Example:
            {
                "fps": 24.5,
                "memory_mb": 1024.0,
                "model_info": {
                    "name": "CLIP",
                    "version": "ViT-B/32",
                    "embedding_dim": 512
                },
                "index_time_seconds": 45.2,
                "total_frames": 1200
            }
        """
        pass


def validate_interface(implementation: VideoSearchInterface) -> List[str]:
    """Validate that an implementation conforms to the interface.

    Args:
        implementation: An instance of a VideoSearchInterface implementation.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Check it's actually an instance of the interface
    if not isinstance(implementation, VideoSearchInterface):
        errors.append(
            f"Implementation must inherit from VideoSearchInterface, "
            f"got {type(implementation).__name__}"
        )
        return errors

    # Check required methods exist and are callable
    required_methods = ["load_video", "search", "get_processing_stats"]
    for method_name in required_methods:
        if not hasattr(implementation, method_name):
            errors.append(f"Missing required method: {method_name}")
        elif not callable(getattr(implementation, method_name)):
            errors.append(f"Method {method_name} is not callable")

    return errors
