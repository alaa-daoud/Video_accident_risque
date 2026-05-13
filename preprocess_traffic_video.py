import argparse
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


# Detection classes and thresholds.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
STOP_SPEED_THRESHOLD = 5.0
STOP_TIME_THRESHOLD_S = 1.0
PREACCIDENT_CONFIRM_FRAMES = 2
ACCIDENT_CONFIRM_FRAMES = 3
CONTACT_IOU_THRESHOLD = 0.05
CLOSE_DISTANCE_THRESHOLD = 60.0
RELATIVE_SPEED_THRESHOLD = 80.0
STRONG_DECEL_THRESHOLD = -700.0
ANGLE_CHANGE_THRESHOLD = 45.0
PREACCIDENT_DISTANCE_THRESHOLD = 100.0
PREACCIDENT_RELATIVE_SPEED_THRESHOLD = 45.0
PREACCIDENT_DECEL_THRESHOLD = -300.0
PREACCIDENT_ANGLE_THRESHOLD = 20.0
FAST_COLLISION_RELATIVE_SPEED_THRESHOLD = 140.0
FAST_COLLISION_DISTANCE_THRESHOLD = 90.0
VERY_STRONG_DECEL_THRESHOLD = -1200.0
FAST_COLLISION_CONFIRM_FRAMES = 1
FAST_COLLISION_PERSIST_DISTANCE_THRESHOLD = 140.0
QUEUE_RELATIVE_SPEED_THRESHOLD = 45.0
QUEUE_DECEL_THRESHOLD = -500.0
QUEUE_DIRECTION_THRESHOLD = 20.0


# Arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Pretraitement YOLO d'une video de trafic.")
    parser.add_argument("--video", required=True, help="Chemin vers la video a analyser.")
    parser.add_argument("--output", default="processed_video.json", help="Fichier JSON de sortie.")
    parser.add_argument("--black-box-output", default="boite_noire.json", help="Fichier de sortie des accidents detectes.")
    parser.add_argument("--model", default="yolov8n.pt", help="Modele YOLO a utiliser.")
    parser.add_argument("--conf", type=float, default=0.4, help="Seuil de confiance YOLO.")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil anti-doublon YOLO.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille de l'image envoyee a YOLO.")
    return parser.parse_args()


# Geometry helpers.
def get_center(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_ground_point(x1, y1, x2, y2):
    # Le bas de la boite approxime mieux la position au sol que le centre.
    return (x1 + x2) / 2.0, y2


def compute_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def compute_speed(previous_center, current_center, fps):
    return compute_distance(previous_center, current_center) * fps


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


# Video and detection helpers.
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


def remove_duplicate_vehicles(vehicles, iou_threshold):
    # Si deux boites se chevauchent beaucoup, on garde la plus confiante.
    vehicles = sorted(vehicles, key=lambda vehicle: vehicle["confidence"], reverse=True)
    filtered_vehicles = []

    for vehicle in vehicles:
        duplicate = False
        for kept_vehicle in filtered_vehicles:
            if compute_iou(vehicle["box"], kept_vehicle["box"]) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            filtered_vehicles.append(vehicle)

    return filtered_vehicles


# Relative metrics and risk logic.
def add_relative_metrics(vehicles, persistent_accident_ids):
    for vehicle in vehicles:
        nearest_vehicle = None
        nearest_distance = None
        contact_vehicle = None
        contact_iou = 0.0

        for other_vehicle in vehicles:
            if vehicle["id"] == other_vehicle["id"]:
                continue

            if vehicle["id"] not in persistent_accident_ids and other_vehicle["id"] in persistent_accident_ids:
                continue

            distance = compute_distance(vehicle["ground_point"], other_vehicle["ground_point"])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_vehicle = other_vehicle

            iou = compute_iou(vehicle["box"], other_vehicle["box"])
            if iou > contact_iou:
                contact_iou = iou
                contact_vehicle = other_vehicle

        vehicle["nearest_vehicle_id"] = None if nearest_vehicle is None else nearest_vehicle["id"]
        vehicle["nearest_distance_px"] = nearest_distance
        vehicle["relative_speed_px_s"] = None
        vehicle["contact_vehicle_id"] = None if contact_vehicle is None else contact_vehicle["id"]
        vehicle["contact_iou"] = contact_iou

        if nearest_vehicle is not None:
            if vehicle["speed_px_s"] is not None and nearest_vehicle["speed_px_s"] is not None:
                vehicle["relative_speed_px_s"] = vehicle["speed_px_s"] - nearest_vehicle["speed_px_s"]


def is_queueing_perspective_case(vehicle):
    has_low_relative_speed = (
        vehicle["relative_speed_px_s"] is not None
        and abs(vehicle["relative_speed_px_s"]) < QUEUE_RELATIVE_SPEED_THRESHOLD
    )
    has_small_direction_change = (
        vehicle["direction_change_deg"] is None
        or vehicle["direction_change_deg"] < QUEUE_DIRECTION_THRESHOLD
    )
    has_no_hard_brake = (
        vehicle["acceleration_px_s2"] is None
        or vehicle["acceleration_px_s2"] > QUEUE_DECEL_THRESHOLD
    )
    is_close = (
        vehicle["nearest_distance_px"] is not None
        and vehicle["nearest_distance_px"] < CLOSE_DISTANCE_THRESHOLD * 1.5
    )

    return is_close and has_low_relative_speed and has_small_direction_change and has_no_hard_brake


def get_accident_reasons(vehicle):
    # Regle prudente en 2D: le contact visuel seul ne prouve pas un accident.
    reasons = []
    is_queue_case = is_queueing_perspective_case(vehicle)

    if vehicle["contact_iou"] >= CONTACT_IOU_THRESHOLD and not is_queue_case:
        reasons.append("box contact")

    if vehicle["nearest_distance_px"] is not None:
        if vehicle["nearest_distance_px"] < CLOSE_DISTANCE_THRESHOLD and not is_queue_case:
            reasons.append("very close vehicle")

    if vehicle["relative_speed_px_s"] is not None:
        if abs(vehicle["relative_speed_px_s"]) > RELATIVE_SPEED_THRESHOLD:
            reasons.append("high relative speed")

    if vehicle["acceleration_px_s2"] is not None:
        if vehicle["acceleration_px_s2"] < STRONG_DECEL_THRESHOLD:
            reasons.append("strong deceleration")

    if vehicle["acceleration_px_s2"] is not None:
        if vehicle["acceleration_px_s2"] < VERY_STRONG_DECEL_THRESHOLD:
            reasons.append("very strong deceleration")

    if vehicle["direction_change_deg"] is not None:
        if vehicle["direction_change_deg"] > ANGLE_CHANGE_THRESHOLD:
            reasons.append("sudden direction change")

    if vehicle["is_stopped"] and vehicle["nearest_distance_px"] is not None:
        if vehicle["nearest_distance_px"] < CLOSE_DISTANCE_THRESHOLD * 1.5:
            reasons.append("stopped near traffic")

    return reasons


def get_preaccident_reasons(vehicle):
    # Version plus souple pour les comportements dangereux avant collision.
    reasons = []
    is_queue_case = is_queueing_perspective_case(vehicle)

    if vehicle["nearest_distance_px"] is not None:
        if vehicle["nearest_distance_px"] < PREACCIDENT_DISTANCE_THRESHOLD and not is_queue_case:
            reasons.append("close vehicle")

    if vehicle["relative_speed_px_s"] is not None:
        if abs(vehicle["relative_speed_px_s"]) > PREACCIDENT_RELATIVE_SPEED_THRESHOLD:
            reasons.append("elevated relative speed")

    if vehicle["acceleration_px_s2"] is not None:
        if vehicle["acceleration_px_s2"] < PREACCIDENT_DECEL_THRESHOLD:
            reasons.append("moderate deceleration")

    if vehicle["direction_change_deg"] is not None:
        if vehicle["direction_change_deg"] > PREACCIDENT_ANGLE_THRESHOLD:
            reasons.append("trajectory change")

    if vehicle["is_stopped"] and vehicle["nearest_distance_px"] is not None:
        if vehicle["nearest_distance_px"] < PREACCIDENT_DISTANCE_THRESHOLD:
            reasons.append("stopped near traffic")

    if vehicle["contact_iou"] >= CONTACT_IOU_THRESHOLD and not is_queue_case:
        reasons.append("visual contact")

    return reasons


def get_involved_vehicle_ids(vehicle):
    involved_ids = [vehicle["id"]]

    if vehicle["contact_vehicle_id"] is not None:
        involved_ids.append(vehicle["contact_vehicle_id"])
    elif vehicle["nearest_vehicle_id"] is not None:
        if vehicle["nearest_distance_px"] is not None and vehicle["nearest_distance_px"] < CLOSE_DISTANCE_THRESHOLD * 1.5:
            involved_ids.append(vehicle["nearest_vehicle_id"])

    return sorted(set(involved_ids))


def is_fast_collision_candidate(vehicle, reasons):
    has_contact = vehicle["contact_iou"] >= CONTACT_IOU_THRESHOLD
    is_very_close = (
        vehicle["nearest_distance_px"] is not None
        and vehicle["nearest_distance_px"] < FAST_COLLISION_DISTANCE_THRESHOLD
    )
    has_high_relative_speed = (
        vehicle["relative_speed_px_s"] is not None
        and abs(vehicle["relative_speed_px_s"]) > FAST_COLLISION_RELATIVE_SPEED_THRESHOLD
    )
    has_strong_decel = (
        "strong deceleration" in reasons
        or "very strong deceleration" in reasons
    )
    has_direction_change = "sudden direction change" in reasons

    if has_contact and has_high_relative_speed and has_strong_decel:
        return True

    if is_very_close and has_high_relative_speed and has_strong_decel:
        return True

    if has_contact and has_strong_decel and has_direction_change:
        return True

    return False


def should_inherit_accident_status(vehicle, persistent_accident_records):
    for record in persistent_accident_records:
        previous_ground_point = record["ground_point"]
        current_ground_point = vehicle["ground_point"]
        distance = compute_distance(previous_ground_point, current_ground_point)

        if distance > FAST_COLLISION_PERSIST_DISTANCE_THRESHOLD:
            continue

        if vehicle["class_name"] != record["class_name"]:
            continue

        if vehicle["contact_iou"] >= CONTACT_IOU_THRESHOLD or vehicle["nearest_distance_px"] is not None:
            return True

    return False


def refresh_persistent_accident_records(vehicles, persistent_accident_ids, persistent_accident_records):
    refreshed_records = []

    for vehicle in vehicles:
        if vehicle["id"] in persistent_accident_ids:
            refreshed_records.append(
                {
                    "id": vehicle["id"],
                    "class_name": vehicle["class_name"],
                    "ground_point": vehicle["ground_point"],
                }
            )

    if refreshed_records:
        persistent_accident_records.clear()
        persistent_accident_records.extend(refreshed_records)


def add_accident_status(vehicles, accident_frames, preaccident_frames, persistent_accident_ids, persistent_accident_records):
    current_ids = {vehicle["id"] for vehicle in vehicles}
    for vehicle_id in list(accident_frames.keys()):
        if vehicle_id not in current_ids:
            accident_frames[vehicle_id] = 0
    for vehicle_id in list(preaccident_frames.keys()):
        if vehicle_id not in current_ids:
            preaccident_frames[vehicle_id] = 0

    newly_confirmed_ids = []

    for vehicle in vehicles:
        if vehicle["id"] in persistent_accident_ids:
            vehicle["risk_score"] = max(vehicle.get("risk_score", 0), 3)
            vehicle["risk_reasons"] = vehicle.get("risk_reasons", [])
            vehicle["status"] = "accident"
            vehicle["is_accident"] = True
            vehicle["is_dangerous"] = False
            vehicle["is_inherited_accident"] = False
            continue

        if should_inherit_accident_status(vehicle, persistent_accident_records):
            persistent_accident_ids.add(vehicle["id"])
            vehicle["risk_score"] = max(vehicle.get("risk_score", 0), 3)
            vehicle["risk_reasons"] = vehicle.get("risk_reasons", [])
            vehicle["status"] = "accident"
            vehicle["is_accident"] = True
            vehicle["is_dangerous"] = False
            vehicle["is_inherited_accident"] = True
            continue

        accident_reasons = get_accident_reasons(vehicle)
        accident_score = len(accident_reasons)
        has_contact = vehicle["contact_iou"] >= CONTACT_IOU_THRESHOLD
        has_motion_anomaly = (
            "strong deceleration" in accident_reasons
            or "sudden direction change" in accident_reasons
            or "high relative speed" in accident_reasons
        )
        has_spatial_risk = (
            "very close vehicle" in accident_reasons
            or "stopped near traffic" in accident_reasons
            or has_contact
        )

        is_fast_collision = is_fast_collision_candidate(vehicle, accident_reasons)
        is_candidate = (accident_score >= 3 and has_motion_anomaly and has_spatial_risk) or is_fast_collision

        if is_candidate:
            accident_frames[vehicle["id"]] = accident_frames.get(vehicle["id"], 0) + 1
        else:
            accident_frames[vehicle["id"]] = 0

        confirm_frames = FAST_COLLISION_CONFIRM_FRAMES if is_fast_collision else ACCIDENT_CONFIRM_FRAMES
        is_accident = accident_frames[vehicle["id"]] >= confirm_frames
        if is_accident and vehicle["id"] not in persistent_accident_ids:
            newly_confirmed_ids.append(vehicle["id"])

        preaccident_reasons = get_preaccident_reasons(vehicle)
        preaccident_score = len(preaccident_reasons)
        has_pre_motion_anomaly = (
            "moderate deceleration" in preaccident_reasons
            or "trajectory change" in preaccident_reasons
            or "elevated relative speed" in preaccident_reasons
        )
        has_pre_spatial_risk = (
            "close vehicle" in preaccident_reasons
            or "stopped near traffic" in preaccident_reasons
            or "visual contact" in preaccident_reasons
        )
        is_preaccident_candidate = (
            not is_accident
            and preaccident_score >= 2
            and has_pre_motion_anomaly
            and has_pre_spatial_risk
        )

        if is_preaccident_candidate:
            preaccident_frames[vehicle["id"]] = preaccident_frames.get(vehicle["id"], 0) + 1
        else:
            preaccident_frames[vehicle["id"]] = 0

        is_dangerous = preaccident_frames[vehicle["id"]] >= PREACCIDENT_CONFIRM_FRAMES

        vehicle["risk_score"] = accident_score if is_accident else preaccident_score
        vehicle["risk_reasons"] = accident_reasons if is_accident else preaccident_reasons
        vehicle["is_fast_collision"] = is_fast_collision
        if is_accident:
            vehicle["status"] = "accident"
        elif is_dangerous:
            vehicle["status"] = "dangerous"
        elif vehicle["is_stopped"]:
            vehicle["status"] = "stopped"
        else:
            vehicle["status"] = "moving"
        vehicle["is_accident"] = is_accident
        vehicle["is_dangerous"] = is_dangerous
        vehicle["is_inherited_accident"] = False

    for vehicle in vehicles:
        if vehicle["id"] in newly_confirmed_ids:
            involved_ids = get_involved_vehicle_ids(vehicle)
            persistent_accident_ids.update(involved_ids)
            for involved_vehicle in vehicles:
                if involved_vehicle["id"] in involved_ids:
                    persistent_accident_records.append(
                        {
                            "id": involved_vehicle["id"],
                            "class_name": involved_vehicle["class_name"],
                            "ground_point": involved_vehicle["ground_point"],
                        }
                    )

    for vehicle in vehicles:
        if vehicle["id"] in persistent_accident_ids:
            vehicle["status"] = "accident"
            vehicle["is_accident"] = True
            vehicle["is_dangerous"] = False

    refresh_persistent_accident_records(vehicles, persistent_accident_ids, persistent_accident_records)
    return newly_confirmed_ids


def build_black_box_entry(video_path, fps, frame_index, frames, vehicles, trigger_vehicle_id):
    trigger_vehicle = None
    for vehicle in vehicles:
        if vehicle["id"] == trigger_vehicle_id:
            trigger_vehicle = vehicle
            break

    if trigger_vehicle is None:
        return None

    involved_vehicle_ids = get_involved_vehicle_ids(trigger_vehicle)
    lookback_frames = int(5 * fps)
    start_frame = max(0, frame_index - lookback_frames)
    history = []

    for frame_data in frames[start_frame : frame_index + 1]:
        frame_vehicles = []
        for vehicle in frame_data["vehicles"]:
            if vehicle["id"] in involved_vehicle_ids:
                frame_vehicles.append(vehicle)

        if frame_vehicles:
            history.append(
                {
                    "frame_index": frame_data["frame_index"],
                    "time_s": frame_data["time_s"],
                    "vehicles": frame_vehicles,
                }
            )

    return {
        "video": video_path,
        "accident_frame": frame_index,
        "accident_time_s": frame_index / fps,
        "involved_vehicle_ids": involved_vehicle_ids,
        "trigger_vehicle_id": trigger_vehicle_id,
        "risk_reasons": trigger_vehicle.get("risk_reasons", []),
        "is_fast_collision": trigger_vehicle.get("is_fast_collision", False),
        "history_5s_before": history,
    }


# Frame processing.
def extract_frame_data(
    result,
    frame_index,
    fps,
    previous_positions,
    previous_speeds,
    previous_headings,
    stopped_frames,
    accident_frames,
    preaccident_frames,
    persistent_accident_ids,
    persistent_accident_records,
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
        ground_point = get_ground_point(x1, y1, x2, y2)
        vehicle_id = int(id_tensor.item())

        speed_px_s = None
        heading_deg = None
        if vehicle_id in previous_positions:
            speed_px_s = compute_speed(previous_positions[vehicle_id], ground_point, fps)
            heading_deg = compute_heading(previous_positions[vehicle_id], ground_point)

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

        previous_positions[vehicle_id] = ground_point
        position_history.setdefault(vehicle_id, []).append([int(center[0]), int(center[1])])

        vehicles.append(
            {
                "id": vehicle_id,
                "class_name": class_name,
                "confidence": float(conf_tensor.item()),
                "box": [x1, y1, x2, y2],
                "center": [center[0], center[1]],
                "ground_point": [ground_point[0], ground_point[1]],
                "speed_px_s": speed_px_s,
                "acceleration_px_s2": acceleration_px_s2,
                "heading_deg": heading_deg,
                "direction_change_deg": direction_change_deg,
                "stopped_time_s": stopped_time_s,
                "is_stopped": is_stopped,
                "trajectory": position_history[vehicle_id],
                "is_fast_collision": False,
            }
        )

    vehicles = remove_duplicate_vehicles(vehicles, duplicate_iou)
    add_relative_metrics(vehicles, persistent_accident_ids)
    newly_confirmed_ids = add_accident_status(
        vehicles,
        accident_frames,
        preaccident_frames,
        persistent_accident_ids,
        persistent_accident_records,
    )
    return vehicles, newly_confirmed_ids


# Main preprocessing pass.
def main():
    args = parse_args()
    model = YOLO(args.model)
    fps, width, height, frame_count = get_video_info(args.video)

    previous_positions = {}
    previous_speeds = {}
    previous_headings = {}
    stopped_frames = {}
    accident_frames = {}
    preaccident_frames = {}
    persistent_accident_ids = set()
    persistent_accident_records = []
    position_history = {}
    frames = []
    black_box_entries = []
    recorded_accident_ids = set()

    results = model.track(
        source=args.video,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        vehicles, newly_confirmed_ids = extract_frame_data(
            result,
            frame_index,
            fps,
            previous_positions,
            previous_speeds,
            previous_headings,
            stopped_frames,
            accident_frames,
            preaccident_frames,
            persistent_accident_ids,
            persistent_accident_records,
            position_history,
            args.iou,
        )

        frame_data = {
            "frame_index": frame_index,
            "time_s": frame_index / fps,
            "vehicles": vehicles,
        }
        frames.append(frame_data)

        for vehicle_id in newly_confirmed_ids:
            if vehicle_id in recorded_accident_ids:
                continue

            black_box_entry = build_black_box_entry(args.video, fps, frame_index, frames, vehicles, vehicle_id)
            if black_box_entry is not None:
                black_box_entries.append(black_box_entry)
                recorded_accident_ids.update(black_box_entry["involved_vehicle_ids"])

        if frame_index % 50 == 0:
            print(f"Frames traitees: {frame_index}/{frame_count}")

    data = {
        "video": args.video,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frames": frames,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data), encoding="utf-8")
    black_box_path = Path(args.black_box_output)
    black_box_path.parent.mkdir(parents=True, exist_ok=True)
    black_box_path.write_text(json.dumps(black_box_entries), encoding="utf-8")
    print(f"Pretraitement termine: {output_path.resolve()}")
    print(f"Boite noire enregistree: {black_box_path.resolve()}")


if __name__ == "__main__":
    main()
