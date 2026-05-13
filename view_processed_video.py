import argparse
import json

import cv2
import numpy as np


WINDOW_NAME = "Processed traffic video"
INFO_WINDOW_NAME = "Vehicle data"


# Arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Viewer fluide d'une video deja traitee.")
    parser.add_argument("--video", required=True, help="Chemin vers la video.")
    parser.add_argument("--data", required=True, help="Fichier JSON cree par preprocess_traffic_video.py.")
    parser.add_argument("--display-width", type=int, default=960, help="Largeur de la fenetre d'affichage.")
    return parser.parse_args()


# Display helpers.
def format_value(value, suffix="", digits=1):
    if value is None:
        return "?"
    return f"{value:.{digits}f}{suffix}"


def resize_for_display(frame, target_width):
    height, width = frame.shape[:2]
    if target_width <= 0 or width <= target_width:
        return frame

    ratio = target_width / width
    target_height = int(height * ratio)
    return cv2.resize(frame, (target_width, target_height))


def find_vehicle(vehicles, vehicle_id):
    for vehicle in vehicles:
        if vehicle["id"] == vehicle_id:
            return vehicle
    return None


def get_vehicle_status(vehicle):
    if vehicle.get("status") in {"moving", "stopped", "dangerous", "accident"}:
        return vehicle["status"]
    if vehicle.get("is_stopped"):
        return "stopped"
    return "moving"


# Video drawing.
def draw_vehicle_boxes(frame, vehicles, selected_vehicle_id):
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["box"]
        status = get_vehicle_status(vehicle)
        color = (0, 200, 0)
        label = f'v:{format_value(vehicle["speed_px_s"], "", 0)} {status}'

        if status == "stopped":
            color = (0, 165, 255)
            label = f'v:{format_value(vehicle["speed_px_s"], "", 0)} stopped'
        elif status == "dangerous":
            color = (0, 165, 255)
            label = f'v:{format_value(vehicle["speed_px_s"], "", 0)} danger'
        elif status == "accident":
            color = (0, 0, 255)
            label = f'v:{format_value(vehicle["speed_px_s"], "", 0)} accident'

        if vehicle["id"] == selected_vehicle_id:
            color = (0, 255, 255) if status != "accident" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.putText(
        frame,
        f"Vehicles detected: {len(vehicles)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (40, 40, 255),
        2,
    )


def get_trajectory_until_now(frames_data, vehicle_id, current_frame_index):
    trajectory = []

    if vehicle_id is None:
        return trajectory

    for frame_data in frames_data[: current_frame_index + 1]:
        vehicle = find_vehicle(frame_data["vehicles"], vehicle_id)
        if vehicle is None:
            continue

        center_x, center_y = vehicle["center"]
        trajectory.append((int(center_x), int(center_y)))

    return trajectory


def draw_trajectory(frame, frames_data, vehicle_id, current_frame_index):
    if vehicle_id is None:
        return

    trajectory = get_trajectory_until_now(frames_data, vehicle_id, current_frame_index)
    if len(trajectory) < 2:
        return

    for index in range(1, len(trajectory)):
        previous_point = trajectory[index - 1]
        current_point = trajectory[index]
        cv2.line(frame, previous_point, current_point, (255, 220, 0), 2)


def build_vehicle_data_window(vehicle):
    if vehicle is None:
        return None

    center_x, center_y = vehicle["center"]
    status = get_vehicle_status(vehicle)
    reasons = vehicle.get("risk_reasons", [])

    image = np.zeros((560, 560, 3), dtype=np.uint8)
    image[:] = (24, 24, 24)

    cv2.putText(
        image,
        f'VEHICLE #{vehicle["id"]}',
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.line(image, (25, 65), (535, 65), (90, 90, 90), 1)

    info_lines = [
        f'Type: {vehicle["class_name"]}',
        f'Status: {status}',
        f'Risk score: {vehicle.get("risk_score", 0)}',
        f'Confidence: {vehicle["confidence"]:.2f}',
        "",
        f'Center: ({center_x:.1f}, {center_y:.1f})',
        f'Ground point: ({vehicle.get("ground_point", ["?", "?"])[0]}, {vehicle.get("ground_point", ["?", "?"])[1]})',
        f'Speed: {format_value(vehicle["speed_px_s"], " px/s")}',
        f'Accel: {format_value(vehicle["acceleration_px_s2"], " px/s2")}',
        f'Angle change: {format_value(vehicle["direction_change_deg"], " deg")}',
        "",
        f'Nearest ID: {vehicle["nearest_vehicle_id"] if vehicle["nearest_vehicle_id"] is not None else "?"}',
        f'Nearest dist: {format_value(vehicle["nearest_distance_px"], " px")}',
        f'Rel speed: {format_value(vehicle["relative_speed_px_s"], " px/s")}',
        f'Contact: {format_value(vehicle.get("contact_iou"), "", 2)}',
        f'Stopped time: {format_value(vehicle["stopped_time_s"], " s")}',
    ]

    y = 100
    for line in info_lines:
        if line == "":
            y += 15
            cv2.line(image, (25, y), (535, y), (65, 65, 65), 1)
            y += 25
            continue

        cv2.putText(
            image,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            2,
        )
        y += 30

    if reasons:
        cv2.line(image, (25, y), (535, y), (65, 65, 65), 1)
        y += 30
        cv2.putText(image, "Risk reasons:", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        for reason in reasons[:4]:
            cv2.putText(image, f"- {reason}", (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 25

    return image


def show_vehicle_data_window(vehicle):
    data_image = build_vehicle_data_window(vehicle)
    if data_image is None:
        try:
            cv2.destroyWindow(INFO_WINDOW_NAME)
        except cv2.error:
            pass
        return

    cv2.imshow(INFO_WINDOW_NAME, data_image)


# Mouse interaction.
def select_vehicle_from_click(x, y, vehicles):
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["box"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return vehicle["id"]
    return None


def mouse_callback(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    scale = state["display_scale"]
    original_x = int(x / scale)
    original_y = int(y / scale)
    state["selected_vehicle_id"] = select_vehicle_from_click(original_x, original_y, state["vehicles"])


# Main viewer loop.
def main():
    args = parse_args()
    data = json.loads(open(args.data, encoding="utf-8").read())
    frames_data = data["frames"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or data.get("fps", 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_scale = min(1.0, args.display_width / width)
    delay_ms = max(1, int(1000 / fps))
    state = {"vehicles": [], "selected_vehicle_id": None, "display_scale": display_scale}

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)
    cv2.moveWindow(WINDOW_NAME, 40, 40)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_index >= len(frames_data):
            break

        vehicles = frames_data[frame_index]["vehicles"]
        state["vehicles"] = vehicles
        selected_vehicle = find_vehicle(vehicles, state["selected_vehicle_id"])

        draw_trajectory(frame, frames_data, state["selected_vehicle_id"], frame_index)
        draw_vehicle_boxes(frame, vehicles, state["selected_vehicle_id"])
        show_vehicle_data_window(selected_vehicle)
        if selected_vehicle is not None:
            cv2.moveWindow(INFO_WINDOW_NAME, args.display_width + 70, 40)

        frame = resize_for_display(frame, args.display_width)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
