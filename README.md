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
