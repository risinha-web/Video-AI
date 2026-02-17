# Test Video Data

This directory contains test data for evaluating your video search implementation.

## Obtaining Test Videos

Since we cannot distribute copyrighted content, you'll need to download test videos yourself. Use any video that is **1–10 minutes** long.

### Recommended: Internet Archive

The [Internet Archive](https://archive.org/) hosts thousands of free, public domain videos. Browse and pick any video that interests you:

- [Archive.org Video Collection](https://archive.org/details/movies) — films, clips, animations, documentaries
- [Prelinger Archives](https://archive.org/details/prelinger) — historical/educational films
- [Open Source Movies](https://archive.org/details/opensource_movies) — community-contributed content

To download, find a video you like, click the MP4 download link, and save it to this directory:

```bash
# Example: download a video from archive.org
wget "https://archive.org/download/<video-id>/<filename>.mp4" -O data/test_video.mp4
```

> **Tip:** Choose a video with diverse visual content (people, objects, actions, scenes) — this will give you better signal when testing different query types.

### Alternative Sources

- [Pexels Videos](https://www.pexels.com/videos/) — royalty-free stock footage
- [Pixabay Videos](https://pixabay.com/videos/) — royalty-free clips
- [Coverr](https://coverr.co/) — free stock video
- Your own videos

## Directory Structure

```
data/
├── README.md                 # This file
├── sample_queries.json       # Example test queries
└── ground_truth/
    └── sample_labels.json    # Example ground truth format
```

## Creating Ground Truth

Create ground truth annotations for your chosen video in this format:

```json
{
  "video_id": "your_video",
  "video_path": "data/test_video.mp4",
  "duration_ms": 120000,
  "annotations": [
    {
      "query": "person walking",
      "matches": [
        {
          "start_ms": 5000,
          "end_ms": 8000,
          "description": "Person walks across the frame"
        }
      ]
    }
  ]
}
```

See `ground_truth/sample_labels.json` for a complete example.

## Sample Queries

The `sample_queries.json` file contains example queries at different difficulty levels:
- **Easy**: Simple objects and actions
- **Medium**: Combinations of attributes and actions
- **Hard**: Complex scenes, emotions, or abstract concepts

You should adapt or create queries that are relevant to the video you choose.

## Video Requirements

For the evaluation to work properly, ensure your video:
- Is readable by OpenCV (`cv2.VideoCapture`)
- Has a reasonable frame rate (24-60 fps)
- Is not corrupted or truncated

## Notes

- Ground truth annotations are optional for self-testing
- Without ground truth, only performance metrics will be calculated
- The evaluation framework will warn if ground truth is missing
