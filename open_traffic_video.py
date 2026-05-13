import argparse
import math
import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
WINDOW_NAME = "Traffic video + tracking YOLO"
STOP_SPEED_THRESHOLD = 5.0
STOP_TIME_THRESHOLD_S = 1.0


# Cette partie sert juste a lire les options du script.
def parse_args():
    parser = argparse.ArgumentParser(description="Lecture d'une video de trafic avec tracking YOLO.")
    parser.add_argument("--video", required=True, help="Chemin vers la video a analyser.")
    parser.add_argument("--model", default="yolov8n.pt", help="Modele YOLO a utiliser.")
    parser.add_argument("--conf", type=float, default=0.4, help="Seuil de confiance YOLO.")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil anti-doublon YOLO.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille de l'image envoyee a YOLO.")
    parser.add_argument("--display-width", type=int, default=960, help="Largeur de la fenetre d'affichage.")
    return parser.parse_args()


# Fonctions utilitaires pour la video et les calculs simples.
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frame_count


def get_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return center_x, center_y


def compute_speed(previous_center, current_center, fps):
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance_pixels = (dx * dx + dy * dy) ** 0.5
    return distance_pixels * fps


def compute_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def compute_heading(previous_center, current_center):
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def compute_angle_change(previous_heading, current_heading):
    if previous_heading is None or current_heading is None:
        return None

    delta = current_heading - previous_heading
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return abs(delta)


def format_value(value, suffix="", digits=1):
    if value is None:
        return "?"
    return f"{value:.{digits}f}{suffix}"


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def remove_duplicate_vehicles(vehicles, iou_threshold=0.6):
    # Si deux boites se chevauchent beaucoup, on garde la detection la plus confiante.
    vehicles = sorted(vehicles, key=lambda vehicle: vehicle["confidence"], reverse=True)
    filtered_vehicles = []

    for vehicle in vehicles:
        is_duplicate = False
        for kept_vehicle in filtered_vehicles:
            if compute_iou(vehicle["box"], kept_vehicle["box"]) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_vehicles.append(vehicle)

    return filtered_vehicles


def resize_for_display(frame, target_width):
    height, width = frame.shape[:2]
    if target_width <= 0 or width <= target_width:
        return frame

    ratio = target_width / width
    target_height = int(height * ratio)
    return cv2.resize(frame, (target_width, target_height))


# On transforme la sortie YOLO en infos plus simples par vehicule.
def extract_vehicle_data(
    result,
    fps,
    previous_positions,
    previous_speeds,
    previous_headings,
    stopped_frames,
    position_history,
    duplicate_iou,
):
    boxes = result.boxes
    vehicles = []

    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return vehicles

    for box, cls_tensor, conf_tensor, id_tensor in zip(boxes.xyxy, boxes.cls, boxes.conf, boxes.id):
        class_id = int(cls_tensor.item())
        class_name = result.names.get(class_id, str(class_id))

        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        center = get_center(x1, y1, x2, y2)
        vehicle_id = int(id_tensor.item())

        speed_px_s = None
        heading_deg = None
        if vehicle_id in previous_positions:
            speed_px_s = compute_speed(previous_positions[vehicle_id], center, fps)
            heading_deg = compute_heading(previous_positions[vehicle_id], center)

        acceleration_px_s2 = None
        if speed_px_s is not None and vehicle_id in previous_speeds:
            acceleration_px_s2 = (speed_px_s - previous_speeds[vehicle_id]) * fps

        direction_change_deg = compute_angle_change(previous_headings.get(vehicle_id), heading_deg)

        if speed_px_s is not None:
            previous_speeds[vehicle_id] = speed_px_s
        if heading_deg is not None:
            previous_headings[vehicle_id] = heading_deg

        if speed_px_s is not None and speed_px_s < STOP_SPEED_THRESHOLD:
            stopped_frames[vehicle_id] = stopped_frames.get(vehicle_id, 0) + 1
        else:
            stopped_frames[vehicle_id] = 0

        stopped_time_s = stopped_frames[vehicle_id] / fps
        is_stopped = stopped_time_s >= STOP_TIME_THRESHOLD_S

        previous_positions[vehicle_id] = center
        position_history.setdefault(vehicle_id, []).append((int(center[0]), int(center[1])))

        vehicles.append(
            {
                "id": vehicle_id,
                "class_name": class_name,
                "confidence": float(conf_tensor.item()),
                "box": (x1, y1, x2, y2),
                "center": center,
                "speed_px_s": speed_px_s,
                "acceleration_px_s2": acceleration_px_s2,
                "heading_deg": heading_deg,
                "direction_change_deg": direction_change_deg,
                "stopped_time_s": stopped_time_s,
                "is_stopped": is_stopped,
            }
        )

    vehicles = remove_duplicate_vehicles(vehicles, duplicate_iou)
    add_relative_metrics(vehicles)
    return vehicles


def add_relative_metrics(vehicles):
    for vehicle in vehicles:
        nearest_vehicle = None
        nearest_distance = None

        for other_vehicle in vehicles:
            if vehicle["id"] == other_vehicle["id"]:
                continue

            distance = compute_distance(vehicle["center"], other_vehicle["center"])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_vehicle = other_vehicle

        vehicle["nearest_vehicle_id"] = None if nearest_vehicle is None else nearest_vehicle["id"]
        vehicle["nearest_distance_px"] = nearest_distance
        vehicle["relative_speed_px_s"] = None

        if nearest_vehicle is not None:
            if vehicle["speed_px_s"] is not None and nearest_vehicle["speed_px_s"] is not None:
                vehicle["relative_speed_px_s"] = vehicle["speed_px_s"] - nearest_vehicle["speed_px_s"]


# Cette partie gere seulement ce qu'on dessine a l'ecran.
def draw_vehicle_info(frame, vehicles, selected_vehicle_id):
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["box"]
        color = (0, 200, 0)
        label = str(vehicle["id"])

        if vehicle["is_stopped"]:
            color = (0, 0, 255)
            label = f'{vehicle["id"]} STOP'

        if vehicle["id"] == selected_vehicle_id:
            if vehicle["is_stopped"]:
                color = (0, 140, 255)
            else:
                color = (0, 255, 255)

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


def draw_selected_vehicle_trajectory(frame, selected_vehicle_id, position_history):
    if selected_vehicle_id is None:
        return

    points = position_history.get(selected_vehicle_id, [])
    if len(points) < 2:
        return

    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], (255, 220, 0), 2)


def draw_selected_vehicle_panel(frame, vehicles, selected_vehicle_id):
    selected_vehicle = None
    for vehicle in vehicles:
        if vehicle["id"] == selected_vehicle_id:
            selected_vehicle = vehicle
            break

    if selected_vehicle is None:
        return

    center_x, center_y = selected_vehicle["center"]
    info_lines = [
        f'Selected ID: {selected_vehicle["id"]}',
        f'Type: {selected_vehicle["class_name"]}',
        f'Status: {"stopped" if selected_vehicle["is_stopped"] else "moving"}',
        f'Confidence: {selected_vehicle["confidence"]:.2f}',
        f'Center: ({center_x:.1f}, {center_y:.1f})',
        f'Speed: {format_value(selected_vehicle["speed_px_s"], " px/s")}',
        f'Accel: {format_value(selected_vehicle["acceleration_px_s2"], " px/s2")}',
        f'Angle change: {format_value(selected_vehicle["direction_change_deg"], " deg")}',
        f'Nearest ID: {selected_vehicle["nearest_vehicle_id"] if selected_vehicle["nearest_vehicle_id"] is not None else "?"}',
        f'Nearest dist: {format_value(selected_vehicle["nearest_distance_px"], " px")}',
        f'Rel speed: {format_value(selected_vehicle["relative_speed_px_s"], " px/s")}',
        f'Stopped time: {format_value(selected_vehicle["stopped_time_s"], " s")}',
    ]

    frame_height, frame_width = frame.shape[:2]
    panel_x1 = 20
    panel_y1 = frame_height - 290
    panel_x2 = min(470, frame_width - 20)
    panel_y2 = frame_height - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), 2)

    y = panel_y1 + 30
    for line in info_lines:
        cv2.putText(
            frame,
            line,
            (panel_x1 + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += 25


# Un clic sur une boite selectionne la voiture, sinon on efface la selection.
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
    selected_vehicle_id = select_vehicle_from_click(original_x, original_y, state["vehicles"])
    state["selected_vehicle_id"] = selected_vehicle_id

# On fait tourner le tracking YOLO et on affiche le resultat frame par frame.
def main():
    args = parse_args()
    fps, width, height, frame_count = get_video_info(args.video)
    model = YOLO(args.model)
    previous_positions = {}
    previous_speeds = {}
    previous_headings = {}
    stopped_frames = {}
    position_history = {}
    display_scale = min(1.0, args.display_width / width)
    state = {"vehicles": [], "selected_vehicle_id": None, "display_scale": display_scale}

    print("Video ouverte avec succes")
    print(f"FPS: {fps:.2f}")
    print(f"Resolution: {width}x{height}")
    print(f"Nombre de frames: {frame_count}")
    print(f"Modele YOLO: {args.model}")
    print(f"Confiance minimum: {args.conf}")
    print(f"Taille YOLO: {args.imgsz}")
    print("Appuie sur q pour quitter")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    results = model.track(
        source=args.video,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img.copy()
        vehicles = extract_vehicle_data(
            result,
            fps,
            previous_positions,
            previous_speeds,
            previous_headings,
            stopped_frames,
            position_history,
            args.iou,
        )
        state["vehicles"] = vehicles

        draw_selected_vehicle_trajectory(frame, state["selected_vehicle_id"], position_history)
        draw_vehicle_info(frame, vehicles, state["selected_vehicle_id"])
        draw_selected_vehicle_panel(frame, vehicles, state["selected_vehicle_id"])

        frame = resize_for_display(frame, args.display_width)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
