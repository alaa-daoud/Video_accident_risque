import argparse
import json
from pathlib import Path

import cv2

from view_processed_video import draw_vehicle_boxes, find_vehicle


def parse_args():
    parser = argparse.ArgumentParser(description="Exporte une video annotee a partir du JSON pretraite.")
    parser.add_argument("--video", required=True, help="Video source.")
    parser.add_argument("--data", required=True, help="JSON produit par preprocess_traffic_video.py.")
    parser.add_argument("--output", required=True, help="Video annotee a produire.")
    parser.add_argument("--trail", type=int, default=45, help="Nombre d'images conservees pour les trajectoires.")
    return parser.parse_args()


def codec_for_output(path):
    if Path(path).suffix.lower() == ".webm":
        return "VP80"
    return "mp4v"


def draw_recent_trajectories(frame, frames_data, vehicles, frame_index, trail):
    for vehicle in vehicles:
        vehicle_id = vehicle["id"]
        points = []
        start = max(0, frame_index - trail)
        for previous in frames_data[start : frame_index + 1]:
            tracked = find_vehicle(previous["vehicles"], vehicle_id)
            if tracked is None:
                continue
            center_x, center_y = tracked["center"]
            points.append((int(center_x), int(center_y)))

        if len(points) < 2:
            continue

        color = (255, 220, 0)
        if vehicle.get("status") == "accident":
            color = (0, 0, 255)
        elif vehicle.get("status") == "dangerous":
            color = (0, 165, 255)

        for index in range(1, len(points)):
            cv2.line(frame, points[index - 1], points[index], color, 2)


def main():
    args = parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    frames_data = data["frames"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or data.get("fps", 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = codec_for_output(output_path)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de creer la video annotee: {output_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_index >= len(frames_data):
            break

        vehicles = frames_data[frame_index]["vehicles"]
        draw_recent_trajectories(frame, frames_data, vehicles, frame_index, args.trail)
        draw_vehicle_boxes(frame, vehicles, selected_vehicle_id=None)
        cv2.putText(
            frame,
            f"Frame {frame_index} / t={frames_data[frame_index]['time_s']:.2f}s",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 255),
            2,
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Video annotee exportee: {output_path.resolve()}")


if __name__ == "__main__":
    main()
