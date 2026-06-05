# Traffic Video Risk Prototype

## Installation

```bash
pip install -r requirements.txt
```

## Preprocess a video

```bash
python preprocess_traffic_video.py --video path/to/video.mp4 --output processed/video.json --black-box-output processed/boite_noire.json
```

Example:

```bash
python preprocess_traffic_video.py --video v4.mov --output processed/v4.json --black-box-output processed/boite_noire_v4.json
```

## View processed results

```bash
python view_processed_video.py --video path/to/video.mp4 --data processed/video.json
```

Example:

```bash
python view_processed_video.py --video v4.mov --data processed/v4.json
```

Controls: `Space` pauses or resumes the video, `n` advances one frame while paused, `q` exits.

## Export an annotated video

```bash
python export_processed_video.py --video v4.mov --data processed/v4.json --output processed/v4_tracking.webm
```

Use WebM when the video is meant to be embedded in the HTML dashboard.

## Generate the dashboard

```bash
python generate_dashboard.py --data processed/v4.json --black-box processed/boite_noire_v4.json --video processed/v4_tracking.webm --dashboard-output processed/dashboard_v4.html
```

The dashboard reuses the processed JSON, aggregates risk over short time windows, and builds a rule-based natural-language explanation from the black-box accident record.
