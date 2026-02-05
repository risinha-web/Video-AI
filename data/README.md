# Test Video Data

This directory contains test data for evaluating your video search implementation.

## Obtaining Test Videos

Since we cannot distribute copyrighted content, you'll need to download or create test videos. Here are recommended sources:

### Option 1: Big Buck Bunny (Recommended for Testing)

An open-source animated short film perfect for testing:

```bash
# Download 1080p version (~150MB)
wget https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4 -O data/big_buck_bunny.mp4

# Or download higher quality
wget https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov -O data/big_buck_bunny_1080p.mov
```

### Option 2: Free Stock Footage

Download from these royalty-free sources:
- [Pexels Videos](https://www.pexels.com/videos/)
- [Pixabay Videos](https://pixabay.com/videos/)
- [Coverr](https://coverr.co/)

### Option 3: Your Own Videos

You can use any video files you have. Recommended specifications:
- Format: MP4, MOV, AVI
- Resolution: 720p or 1080p
- Duration: 1-10 minutes for testing

## Directory Structure

```
data/
├── README.md                 # This file
├── sample_queries.json       # Test queries for evaluation
└── ground_truth/
    └── sample_labels.json    # Annotations for Big Buck Bunny
```

## Creating Ground Truth

If using your own videos, create ground truth annotations in this format:

```json
{
  "video_id": "your_video",
  "video_path": "data/your_video.mp4",
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

## Sample Queries

The `sample_queries.json` file contains test queries at different difficulty levels:
- **Easy**: Simple objects and actions
- **Medium**: Combinations of attributes and actions
- **Hard**: Complex scenes, emotions, or abstract concepts

## Video Requirements

For the evaluation to work properly, ensure your video:
- Is readable by OpenCV (`cv2.VideoCapture`)
- Has a reasonable frame rate (24-60 fps)
- Is not corrupted or truncated

## Notes

- Ground truth annotations are optional for self-testing
- Without ground truth, only performance metrics will be calculated
- The evaluation framework will warn if ground truth is missing
