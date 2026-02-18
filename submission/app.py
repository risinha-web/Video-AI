"""
Video Scene Search Web Interface

This is a basic Gradio template for the web interface. You can:
- Extend this template with additional features
- Replace it entirely with Flask, FastAPI, Streamlit, or custom solution
- Add features like real-time streaming, batch upload, etc.

Run with:
    python -m submission.app
    # or
    cd submission && python app.py
"""

import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import os

# Import your implementation
from main import VideoSearch


# Global searcher instance (reused across requests)
_searcher: Optional[VideoSearch] = None
_current_video: Optional[str] = None


def get_searcher() -> VideoSearch:
    """Get or create the VideoSearch instance."""
    global _searcher
    if _searcher is None:
        _searcher = VideoSearch()
    return _searcher


def load_video(video_file) -> str:
    """Load a video file and prepare it for searching.

    Args:
        video_file: Uploaded video file from Gradio.

    Returns:
        Status message.
    """
    global _current_video

    if video_file is None:
        return "Please upload a video file."

    try:
        searcher = get_searcher()
        video_path = video_file.name if hasattr(video_file, "name") else str(video_file)
        searcher.load_video(video_path)
        _current_video = video_path

        stats = searcher.get_processing_stats()
        return (
            f"Video loaded successfully!\n"
            f"Frames: {stats.get('total_frames', 'N/A')}\n"
            f"Processing speed: {stats.get('fps', 'N/A'):.1f} FPS\n"
            f"Index time: {stats.get('index_time_seconds', 'N/A'):.1f}s"
        )
    except NotImplementedError:
        return "Error: VideoSearch.load_video() not implemented yet."
    except Exception as e:
        return f"Error loading video: {str(e)}"


def search_video(
    query: str,
    top_k: int = 10,
) -> Tuple[List[Tuple[str, str]], dict, str]:
    """Search the loaded video for scenes matching the query.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.

    Returns:
        Tuple of (gallery images, stats dict, message).
    """
    if not query:
        return [], {}, "Please enter a search query."

    if _current_video is None:
        return [], {}, "Please load a video first."

    try:
        searcher = get_searcher()
        results = searcher.search(query, top_k=top_k)
        stats = searcher.get_processing_stats()

        # Format results for gallery
        gallery_items = []
        for i, result in enumerate(results):
            if result.thumbnail_path and Path(result.thumbnail_path).exists():
                # Use actual thumbnail
                caption = f"#{i+1} | {result.timestamp_ms/1000:.1f}s | conf: {result.confidence:.2f}"
                gallery_items.append((result.thumbnail_path, caption))
            else:
                # Create placeholder text (in real implementation, extract frame)
                caption = (
                    f"Result #{i+1}\n"
                    f"Time: {result.timestamp_ms/1000:.1f}s\n"
                    f"Frame: {result.frame_number}\n"
                    f"Confidence: {result.confidence:.2%}"
                )
                gallery_items.append((None, caption))

        message = f"Found {len(results)} results for: '{query}'"
        return gallery_items, stats, message

    except NotImplementedError:
        return [], {}, "Error: VideoSearch.search() not implemented yet."
    except RuntimeError as e:
        return [], {}, str(e)
    except Exception as e:
        return [], {}, f"Error during search: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Video Scene Search",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # Video Scene Search

            Search for specific scenes in videos using natural language queries.

            **How to use:**
            1. Upload a video file
            2. Wait for processing to complete
            3. Enter a search query (e.g., "person wearing a red dress")
            4. View matching scenes with timestamps and confidence scores
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Video input
                video_input = gr.File(
                    label="Upload Video",
                    file_types=["video"],
                    type="filepath",
                )
                load_btn = gr.Button("Load Video", variant="primary")
                load_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                )

            with gr.Column(scale=1):
                # Search controls
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., person wearing a red dress",
                    lines=2,
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Number of Results",
                )
                search_btn = gr.Button("Search", variant="primary")

        # Results section
        with gr.Row():
            with gr.Column(scale=2):
                results_gallery = gr.Gallery(
                    label="Search Results",
                    columns=5,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                )
                search_message = gr.Textbox(
                    label="Search Status",
                    interactive=False,
                )

            with gr.Column(scale=1):
                stats_output = gr.JSON(
                    label="Processing Statistics",
                )

        # Example queries
        gr.Markdown("### Example Queries")
        example_queries = gr.Examples(
            examples=[
                ["person drinking from a cup"],
                ["two people shaking hands"],
                ["someone looking surprised"],
                ["car driving on a road"],
                ["person using a phone"],
            ],
            inputs=query_input,
        )

        # Wire up the callbacks
        load_btn.click(
            fn=load_video,
            inputs=[video_input],
            outputs=[load_status],
        )

        search_btn.click(
            fn=search_video,
            inputs=[query_input, top_k_slider],
            outputs=[results_gallery, stats_output, search_message],
        )

        # Also trigger search on Enter in query box
        query_input.submit(
            fn=search_video,
            inputs=[query_input, top_k_slider],
            outputs=[results_gallery, stats_output, search_message],
        )

    return app


# Create the app
app = create_interface()

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
