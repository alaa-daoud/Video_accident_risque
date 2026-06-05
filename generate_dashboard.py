import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REASON_LABELS = {
    "box contact": "des boites de detection qui se touchent dans l'image",
    "visual contact": "un contact visible entre les vehicules",
    "very close vehicle": "une distance tres courte entre vehicules",
    "close vehicle": "une distance courte entre vehicules",
    "high relative speed": "une vitesse relative elevee",
    "elevated relative speed": "une vitesse relative en hausse",
    "strong deceleration": "un freinage fort",
    "very strong deceleration": "un freinage tres fort",
    "moderate deceleration": "un ralentissement notable",
    "sudden direction change": "un changement brusque de direction",
    "trajectory change": "un changement de trajectoire",
    "stopped near traffic": "un vehicule arrete pres du trafic",
}

FAITHFULNESS_NOTE = (
    "Texte genere par regles a partir des donnees observees: positions des vehicules, trajectoires, vitesses, "
    "distance entre vehicules, freinage et score de risque. Aucun element exterieur a la video n'est ajoute."
)


CAUSAL_EDGES = [
    ("Densite de vehicules", "Faible espacement", "Plus il y a de vehicules, plus les distances disponibles diminuent."),
    ("Faible espacement", "Contact visuel", "Quand deux vehicules se rapprochent, ils peuvent finir par se toucher dans l'image."),
    ("Vitesse relative", "Freinage fort", "Un grand ecart de vitesse peut provoquer un freinage soudain."),
    ("Freinage fort", "Score de risque", "Une forte deceleration est un signal direct de risque."),
    ("Changement de trajectoire", "Score de risque", "Une variation brusque de direction indique un mouvement instable."),
    ("Contact visuel", "Score de risque", "Le contact entre les vehicules dans l'image augmente fortement le risque."),
    ("Score de risque", "Alerte accident", "L'alerte est declenchee quand les indices de risque persistent."),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a traffic-risk dashboard from processed JSON files.")
    parser.add_argument("--data", required=True, help="Processed JSON created by preprocess_traffic_video.py.")
    parser.add_argument("--black-box", required=True, help="Black-box accident JSON.")
    parser.add_argument("--video", required=True, help="Video path used by the dashboard player.")
    parser.add_argument("--window-seconds", type=float, default=0.5, help="Temporal aggregation window size.")
    parser.add_argument("--dashboard-output", default="dashboard_v4.html", help="Dashboard HTML output path.")
    parser.add_argument("--window-output", default="processed/window_features_v4.json", help="Window feature JSON output.")
    parser.add_argument("--explanations-output", default="processed/explanations_v4.json", help="Explanation JSON output.")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_values(values):
    return [value for value in values if value is not None and math.isfinite(value)]


def mean(values):
    values = finite_values(values)
    return None if not values else sum(values) / len(values)


def minimum(values):
    values = finite_values(values)
    return None if not values else min(values)


def maximum(values):
    values = finite_values(values)
    return None if not values else max(values)


def round_or_none(value, digits=3):
    if value is None:
        return None
    return round(value, digits)


def risk_probability(max_risk, accident_count, dangerous_count, min_distance, contact_count):
    score = 0.0
    score += min(max_risk, 5) * 0.7
    score += min(accident_count, 3) * 1.4
    score += min(dangerous_count, 3) * 0.65
    score += min(contact_count, 3) * 0.4
    if min_distance is not None:
        if min_distance < 60:
            score += 1.3
        elif min_distance < 120:
            score += 0.7
    return round(1.0 / (1.0 + math.exp(-score + 2.2)), 3)


def aggregate_windows(data, window_seconds):
    fps = float(data.get("fps") or 30.0)
    frames = data["frames"]
    window_frames = max(1, int(round(window_seconds * fps)))
    windows = []

    for start in range(0, len(frames), window_frames):
        chunk = frames[start : start + window_frames]
        vehicles = [vehicle for frame in chunk for vehicle in frame["vehicles"]]
        status_counts = Counter(vehicle.get("status", "moving") for vehicle in vehicles)
        reasons = Counter(reason for vehicle in vehicles for reason in vehicle.get("risk_reasons", []))
        max_risk = max([vehicle.get("risk_score", 0) or 0 for vehicle in vehicles], default=0)
        min_distance = minimum(vehicle.get("nearest_distance_px") for vehicle in vehicles)
        accident_count = status_counts.get("accident", 0)
        dangerous_count = status_counts.get("dangerous", 0)
        contact_count = sum(1 for vehicle in vehicles if (vehicle.get("contact_iou") or 0) > 0)

        windows.append(
            {
                "index": len(windows),
                "start_frame": chunk[0]["frame_index"],
                "end_frame": chunk[-1]["frame_index"],
                "start_time_s": round(chunk[0]["time_s"], 3),
                "end_time_s": round(chunk[-1]["time_s"], 3),
                "frame_count": len(chunk),
                "vehicle_observations": len(vehicles),
                "unique_vehicle_count": len({vehicle["id"] for vehicle in vehicles}),
                "avg_speed_px_s": round_or_none(mean(vehicle.get("speed_px_s") for vehicle in vehicles), 2),
                "max_speed_px_s": round_or_none(maximum(vehicle.get("speed_px_s") for vehicle in vehicles), 2),
                "avg_abs_relative_speed_px_s": round_or_none(
                    mean(abs(vehicle.get("relative_speed_px_s")) for vehicle in vehicles if vehicle.get("relative_speed_px_s") is not None),
                    2,
                ),
                "min_distance_px": round_or_none(min_distance, 2),
                "min_acceleration_px_s2": round_or_none(minimum(vehicle.get("acceleration_px_s2") for vehicle in vehicles), 2),
                "max_direction_change_deg": round_or_none(maximum(vehicle.get("direction_change_deg") for vehicle in vehicles), 2),
                "contact_count": contact_count,
                "dangerous_count": dangerous_count,
                "accident_count": accident_count,
                "max_risk_score": max_risk,
                "risk_probability": risk_probability(max_risk, accident_count, dangerous_count, min_distance, contact_count),
                "top_reasons": [{"reason": reason, "count": count} for reason, count in reasons.most_common(5)],
            }
        )

    return windows


def find_window_for_time(windows, time_s):
    for window in windows:
        if window["start_time_s"] <= time_s <= window["end_time_s"]:
            return window
    return None


def summarize_history(entry):
    vehicles_by_id = defaultdict(list)
    for frame in entry.get("history_5s_before", []):
        for vehicle in frame.get("vehicles", []):
            vehicles_by_id[vehicle["id"]].append(vehicle)

    summaries = {}
    for vehicle_id, vehicles in vehicles_by_id.items():
        first_vehicle = vehicles[0] if vehicles else {}
        last_vehicle = vehicles[-1] if vehicles else {}
        start_point = first_vehicle.get("ground_point")
        end_point = last_vehicle.get("ground_point")
        travel_distance = None
        if start_point is not None and end_point is not None:
            travel_distance = compute_distance(start_point, end_point)

        summaries[vehicle_id] = {
            "max_speed_px_s": round_or_none(maximum(vehicle.get("speed_px_s") for vehicle in vehicles), 2),
            "max_abs_relative_speed_px_s": round_or_none(
                maximum(abs(vehicle.get("relative_speed_px_s")) for vehicle in vehicles if vehicle.get("relative_speed_px_s") is not None),
                2,
            ),
            "min_distance_px": round_or_none(minimum(vehicle.get("nearest_distance_px") for vehicle in vehicles), 2),
            "min_acceleration_px_s2": round_or_none(minimum(vehicle.get("acceleration_px_s2") for vehicle in vehicles), 2),
            "max_contact_iou": round_or_none(maximum(vehicle.get("contact_iou") for vehicle in vehicles), 4),
            "max_direction_change_deg": round_or_none(maximum(vehicle.get("direction_change_deg") for vehicle in vehicles), 2),
            "last_status": vehicles[-1].get("status", "moving") if vehicles else "unknown",
            "start_point": start_point,
            "end_point": end_point,
            "travel_distance_px": round_or_none(travel_distance, 2),
        }
    return summaries


def compute_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def describe_motion(summary):
    start_point = summary.get("start_point")
    end_point = summary.get("end_point")
    if start_point is None or end_point is None:
        return "trajectoire non disponible"

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    horizontal = "vers la droite" if dx > 40 else "vers la gauche" if dx < -40 else "presque sans decalage horizontal"
    vertical = "vers le bas de l'image" if dy > 40 else "vers le haut de l'image" if dy < -40 else "avec peu de variation verticale"
    return f"{horizontal}, {vertical}"


def simplify_reasons(reasons):
    reason_set = set(reasons)
    hidden = set()
    if "box contact" in reason_set:
        hidden.add("visual contact")
    if "very close vehicle" in reason_set:
        hidden.add("close vehicle")
    if "very strong deceleration" in reason_set:
        hidden.update({"strong deceleration", "moderate deceleration"})
    elif "strong deceleration" in reason_set:
        hidden.add("moderate deceleration")

    return [reason for reason in reasons if reason not in hidden]


def explain_reasons(reasons):
    reasons = simplify_reasons(reasons)
    labels = [REASON_LABELS.get(reason, reason) for reason in reasons]
    if not labels:
        return "aucune raison explicite n'a ete enregistree"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " et " + labels[-1]


def format_trigger(trigger_id):
    if trigger_id is None:
        return "Le declenchement vient des trajectoires observees"
    return f"Le declenchement vient surtout du vehicule #{trigger_id}"


def format_vehicle_ids(vehicle_ids):
    if not vehicle_ids:
        return "aucun vehicule identifie"
    return ", ".join(f"#{vehicle_id}" for vehicle_id in vehicle_ids)


def build_evidence(entry):
    history_summary = summarize_history(entry)
    evidence = []
    for vehicle_id in entry.get("involved_vehicle_ids", []):
        summary = history_summary.get(vehicle_id, {})
        evidence.append(
            {
                "vehicle_id": vehicle_id,
                "max_speed_px_s": summary.get("max_speed_px_s"),
                "max_abs_relative_speed_px_s": summary.get("max_abs_relative_speed_px_s"),
                "min_distance_px": summary.get("min_distance_px"),
                "min_acceleration_px_s2": summary.get("min_acceleration_px_s2"),
                "max_contact_iou": summary.get("max_contact_iou"),
                "max_direction_change_deg": summary.get("max_direction_change_deg"),
                "last_status": summary.get("last_status"),
                "motion": describe_motion(summary),
                "travel_distance_px": summary.get("travel_distance_px"),
            }
        )
    return evidence


def compose_explanation(entry, evidence, local_window):
    time_s = entry["accident_time_s"]
    involved_ids = entry.get("involved_vehicle_ids", [])
    trigger_id = entry.get("trigger_vehicle_id")
    reasons = entry.get("risk_reasons", [])

    motion_text = "; ".join(
        f"le vehicule #{item['vehicle_id']} se deplace {item['motion']}"
        for item in evidence
    )
    if not motion_text:
        motion_text = "les trajectoires disponibles sont insuffisantes"

    trigger_text = format_trigger(trigger_id)
    text = (
        f"A {time_s:.2f}s, une alerte accident est declenchee pour les vehicules "
        f"{format_vehicle_ids(involved_ids)}. Juste avant l'alerte, {motion_text}. "
        f"{trigger_text}: les boites de detection, les trajectoires et les variations de vitesse font ressortir "
        f"{explain_reasons(reasons)}."
    )

    if local_window is not None:
        window_duration = local_window["end_time_s"] - local_window["start_time_s"]
        text += (
            f" Sur la fenetre temporelle correspondante ({window_duration:.2f}s), "
            f"le risque affiche atteint {local_window['risk_probability']:.0%}."
        )

    return text


def build_explanations(black_box_entries, windows):
    explanations = []
    for index, entry in enumerate(black_box_entries, start=1):
        time_s = entry["accident_time_s"]
        involved_ids = entry.get("involved_vehicle_ids", [])
        trigger_id = entry.get("trigger_vehicle_id")
        reasons = entry.get("risk_reasons", [])
        local_window = find_window_for_time(windows, time_s)
        evidence = build_evidence(entry)
        summary_text = compose_explanation(entry, evidence, local_window)

        explanations.append(
            {
                "id": index,
                "time_s": round(time_s, 3),
                "accident_frame": entry["accident_frame"],
                "involved_vehicle_ids": involved_ids,
                "trigger_vehicle_id": trigger_id,
                "risk_reasons": reasons,
                "display_risk_reasons": simplify_reasons(reasons),
                "summary": summary_text,
                "evidence": evidence,
                "faithfulness_note": FAITHFULNESS_NOTE,
            }
        )
    return explanations


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0 or denom_y <= 0:
        return 0.0
    return numerator / math.sqrt(denom_x * denom_y)


def build_causal_graph(windows):
    risk = [window["risk_probability"] for window in windows]
    variables = {
        "Densite de vehicules": [window["unique_vehicle_count"] for window in windows],
        "Faible espacement": [None if window["min_distance_px"] is None else -window["min_distance_px"] for window in windows],
        "Vitesse relative": [window["avg_abs_relative_speed_px_s"] for window in windows],
        "Freinage fort": [None if window["min_acceleration_px_s2"] is None else -window["min_acceleration_px_s2"] for window in windows],
        "Changement de trajectoire": [window["max_direction_change_deg"] for window in windows],
        "Contact visuel": [window["contact_count"] for window in windows],
    }
    node_weights = {
        name: round(abs(pearson(values, risk)), 3)
        for name, values in variables.items()
    }
    node_weights["Score de risque"] = 1.0
    node_weights["Alerte accident"] = 1.0 if any(window["accident_count"] > 0 for window in windows) else 0.0
    return {
        "nodes": [{"id": name, "risk_correlation": node_weights.get(name, 0)} for name in node_weights],
        "edges": [{"source": source, "target": target, "rationale": rationale} for source, target, rationale in CAUSAL_EDGES],
        "note": "Lecture du graphe: on part des observations a gauche, elles alimentent les indices de danger au centre, puis le score declenche l'alerte a droite.",
    }


def json_for_html(value):
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def build_html(data, windows, explanations, causal_graph, video_path):
    video_uri = Path(video_path).resolve().as_uri()
    dashboard_data = {
        "video": data.get("video"),
        "video_uri": video_uri,
        "fps": data.get("fps"),
        "width": data.get("width"),
        "height": data.get("height"),
        "frame_count": data.get("frame_count"),
        "duration_s": round((data.get("frame_count") or len(data.get("frames", []))) / (data.get("fps") or 30.0), 3),
        "windows": windows,
        "explanations": explanations,
        "causal_graph": causal_graph,
        "reason_labels": REASON_LABELS,
    }

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard explicable du risque routier</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #667085;
      --line: #d9dee7;
      --red: #c92a2a;
      --amber: #b7791f;
      --green: #2f7d51;
      --cyan: #0f7285;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 8px; font-size: clamp(26px, 3vw, 40px); letter-spacing: 0; }}
    h2 {{ margin-bottom: 14px; font-size: 20px; }}
    h3 {{ margin-bottom: 8px; font-size: 16px; }}
    .subtitle {{ max-width: 960px; margin-bottom: 0; color: var(--muted); }}
    main {{ padding: 22px clamp(18px, 4vw, 48px) 42px; }}
    .grid {{ display: grid; gap: 16px; }}
    .kpis {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 16px; }}
    .two-col {{ grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr); align-items: start; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .kpi {{ min-height: 104px; }}
    .kpi .label {{ color: var(--muted); font-size: 13px; }}
    .kpi .value {{ margin: 8px 0 4px; font-size: 28px; font-weight: 750; }}
    .kpi .detail {{ color: var(--muted); font-size: 13px; }}
    video {{ width: 100%; max-height: 520px; display: block; background: #101418; border-radius: 6px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 650;
    }}
    button.primary {{ border-color: var(--red); color: var(--red); }}
    canvas {{ width: 100%; height: 260px; display: block; }}
    .event-list {{ display: grid; gap: 12px; }}
    .event {{
      border-left: 4px solid var(--red);
      padding: 12px 12px 12px 14px;
      background: #fff8f8;
      border-radius: 6px;
    }}
    .event p {{ margin-bottom: 10px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      background: #edf2f7;
      color: #344054;
      font-size: 12px;
      font-weight: 650;
    }}
    .pill.red {{ background: #ffe3e3; color: var(--red); }}
    .pill.amber {{ background: #fff3cd; color: var(--amber); }}
    .pill.green {{ background: #ddf4e7; color: var(--green); }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fff; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 2px; font-size: 15px; }}
    .graph {{ width: 100%; min-height: 420px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #fafbfc; }}
    .risk-high {{ color: var(--red); font-weight: 750; }}
    .risk-mid {{ color: var(--amber); font-weight: 750; }}
    .note {{ color: var(--muted); font-size: 13px; margin-bottom: 0; }}
    .section {{ margin-top: 16px; }}
    @media (max-width: 920px) {{
      .kpis, .two-col, .metrics {{ grid-template-columns: 1fr; }}
      canvas {{ height: 220px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dashboard explicable du risque routier</h1>
    <p class="subtitle">Couche prototype construite a partir du JSON traite: aggregation temporelle du risque, explication de l'accident et graphe causal guide par le domaine.</p>
  </header>
  <main>
    <section class="grid kpis" id="kpis"></section>

    <section class="grid two-col">
      <div class="panel">
        <h2>Video source</h2>
        <video id="video" src="{video_uri}" controls preload="metadata"></video>
        <div class="controls" id="eventButtons"></div>
      </div>

      <div class="panel">
        <h2>Explication en langage naturel</h2>
        <div class="event-list" id="explanations"></div>
      </div>
    </section>

    <section class="panel section">
      <h2>Evolution du risque</h2>
      <canvas id="riskChart" width="1200" height="420"></canvas>
      <p class="note">Chaque point represente une fenetre de 0,5 seconde. L'axe horizontal est le temps de la video; l'axe vertical est le niveau de risque estime.</p>
    </section>

    <section class="grid two-col section">
      <div class="panel">
        <h2>Structure causale utilisee</h2>
        <svg class="graph" id="causalGraph" viewBox="0 0 920 420" role="img" aria-label="Graphe causal"></svg>
        <p class="note" id="graphNote"></p>
      </div>

      <div class="panel">
        <h2>Fenetres les plus risquees</h2>
        <table>
          <thead>
            <tr>
              <th>Temps</th>
              <th>Risque</th>
              <th>Vehicules</th>
              <th>Raison principale</th>
            </tr>
          </thead>
          <tbody id="windowTable"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">{json_for_html(dashboard_data)}</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const fmt = (value, digits = 1) => value === null || value === undefined ? "?" : Number(value).toFixed(digits);
    const pct = value => `${{Math.round(value * 100)}}%`;
    const reasonLabel = reason => data.reason_labels[reason] || reason;

    function topWindows() {{
      return [...data.windows].sort((a, b) => b.risk_probability - a.risk_probability).slice(0, 8);
    }}

    function renderKpis() {{
      const maxRiskWindow = data.windows.reduce((best, item) => item.risk_probability > best.risk_probability ? item : best, data.windows[0]);
      const accident = data.explanations[0];
      const uniqueVehicles = new Set();
      data.windows.forEach(window => uniqueVehicles.add(window.unique_vehicle_count));
      const kpis = [
        ["Duree video", `${{fmt(data.duration_s, 2)}}s`, `${{data.frame_count}} frames a ${{fmt(data.fps, 1)}} FPS`],
        ["Pic de risque", pct(maxRiskWindow.risk_probability), `${{fmt(maxRiskWindow.start_time_s, 2)}}s a ${{fmt(maxRiskWindow.end_time_s, 2)}}s`],
        ["Evenements detectes", data.explanations.length, accident ? `Premiere alerte a ${{fmt(accident.time_s, 2)}}s` : "Aucune alerte accident"],
        ["Variables causales", data.causal_graph.nodes.length, "Espacement, vitesse, freinage, contact, trajectoire"],
      ];
      document.getElementById("kpis").innerHTML = kpis.map(([label, value, detail]) => `
        <article class="panel kpi">
          <div class="label">${{label}}</div>
          <div class="value">${{value}}</div>
          <div class="detail">${{detail}}</div>
        </article>
      `).join("");
    }}

    function renderEvents() {{
      const video = document.getElementById("video");
      document.getElementById("eventButtons").innerHTML = data.explanations.map(event => `
        <button class="primary" data-seek="${{event.time_s}}">Aller a l'accident ${{fmt(event.time_s, 2)}}s</button>
      `).join("");
      document.querySelectorAll("[data-seek]").forEach(button => {{
        button.addEventListener("click", () => {{
          video.currentTime = Number(button.dataset.seek);
          video.play();
        }});
      }});

      document.getElementById("explanations").innerHTML = data.explanations.map(event => `
        <article class="event">
          <h3>Evenement #${{event.id}} a ${{fmt(event.time_s, 2)}}s</h3>
          <p>${{event.summary}}</p>
          <div class="pill-row">
            ${{(event.display_risk_reasons || event.risk_reasons).map(reason => `<span class="pill red">${{reasonLabel(reason)}}</span>`).join("")}}
            <span class="pill amber">${{event.trigger_vehicle_id === null || event.trigger_vehicle_id === undefined ? "declencheur non isole" : "declencheur #" + event.trigger_vehicle_id}}</span>
            <span class="pill green">vehicules ${{event.involved_vehicle_ids.map(id => "#" + id).join(", ")}}</span>
          </div>
          <div class="metrics">
            ${{event.evidence.map(item => `
              <div class="metric">
                <span>Vehicule #${{item.vehicle_id}}</span>
                <strong>accel. min: ${{fmt(item.min_acceleration_px_s2, 0)}} px/s2</strong>
                <span>vitesse relative max: ${{fmt(item.max_abs_relative_speed_px_s, 0)}} px/s</span>
                <span>distance min: ${{fmt(item.min_distance_px, 0)}} px</span>
                <span>contact max: ${{fmt(item.max_contact_iou, 3)}}</span>
              </div>
            `).join("")}}
          </div>
          <p class="note">${{event.faithfulness_note}}</p>
        </article>
      `).join("");
    }}

    function renderRiskChart() {{
      const canvas = document.getElementById("riskChart");
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const pad = {{ left: 92, right: 40, top: 34, bottom: 74 }};
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.font = "14px system-ui";
      ctx.strokeStyle = "#e4e7ec";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const y = pad.top + i * (h - pad.top - pad.bottom) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
      }}
      const maxTime = Math.max(...data.windows.map(item => item.end_time_s));
      const x = time => pad.left + (time / maxTime) * (w - pad.left - pad.right);
      const y = risk => h - pad.bottom - risk * (h - pad.top - pad.bottom);

      ctx.strokeStyle = "#344054";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = "#475467";
      ctx.textAlign = "right";
      [0, .25, .5, .75, 1].forEach(value => {{
        ctx.fillText(pct(value), pad.left - 12, y(value) + 5);
      }});

      ctx.textAlign = "center";
      for (let t = 0; t <= Math.ceil(maxTime); t += 1) {{
        const px = x(t);
        ctx.strokeStyle = "#98a2b3";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, h - pad.bottom);
        ctx.lineTo(px, h - pad.bottom + 7);
        ctx.stroke();
        ctx.fillText(`${{t}}s`, px, h - pad.bottom + 27);
      }}

      ctx.beginPath();
      data.windows.forEach((item, index) => {{
        const px = x((item.start_time_s + item.end_time_s) / 2);
        const py = y(item.risk_probability);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }});
      ctx.strokeStyle = "#c92a2a";
      ctx.lineWidth = 3;
      ctx.stroke();

      data.explanations.forEach(event => {{
        const px = x(event.time_s);
        ctx.fillStyle = "rgba(201,42,42,.10)";
        ctx.fillRect(x(Math.max(0, event.time_s - 0.25)), pad.top, x(event.time_s + 0.25) - x(Math.max(0, event.time_s - 0.25)), h - pad.top - pad.bottom);
        ctx.strokeStyle = "#18202a";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, pad.top);
        ctx.lineTo(px, h - pad.bottom);
        ctx.stroke();
        ctx.fillStyle = "#18202a";
        ctx.textAlign = "left";
        ctx.fillText(`accident detecte a ${{fmt(event.time_s, 2)}}s`, px + 8, pad.top + 18);
      }});

      ctx.fillStyle = "#667085";
      ctx.textAlign = "center";
      ctx.font = "700 15px system-ui";
      ctx.fillText("Temps de la video", (pad.left + w - pad.right) / 2, h - 18);
      ctx.save();
      ctx.translate(24, (pad.top + h - pad.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Niveau de risque", 0, 0);
      ctx.restore();
    }}

    function renderGraph() {{
      const svg = document.getElementById("causalGraph");
      const positions = {{
        "Densite de vehicules": [120, 82],
        "Vitesse relative": [120, 210],
        "Faible espacement": [335, 82],
        "Freinage fort": [335, 210],
        "Contact visuel": [550, 82],
        "Changement de trajectoire": [550, 300],
        "Score de risque": [725, 190],
        "Alerte accident": [840, 190],
      }};
      const labels = {{
        "Densite de vehicules": ["Densite", "vehicules"],
        "Vitesse relative": ["Vitesse", "relative"],
        "Faible espacement": ["Faible", "espacement"],
        "Freinage fort": ["Freinage", "fort"],
        "Contact visuel": ["Contact", "visuel"],
        "Changement de trajectoire": ["Changement", "trajectoire"],
        "Score de risque": ["Score", "de risque"],
        "Alerte accident": ["Alerte", "accident"],
      }};
      const edgeLines = data.causal_graph.edges.map(edge => {{
        const [x1, y1] = positions[edge.source];
        const [x2, y2] = positions[edge.target];
        const startX = x1 + 78;
        const endX = x2 - 78;
        const midX = (startX + endX) / 2;
        return `<path d="M ${{startX}} ${{y1}} C ${{midX}} ${{y1}}, ${{midX}} ${{y2}}, ${{endX}} ${{y2}}" fill="none" stroke="#98a2b3" stroke-width="2" marker-end="url(#arrow)" />`;
      }}).join("");
      const nodes = Object.entries(positions).map(([name, [x, y]]) => {{
        const found = data.causal_graph.nodes.find(node => node.id === name);
        const corr = found ? found.risk_correlation : 0;
        const color = name === "Alerte accident" ? "#c92a2a" : name === "Score de risque" ? "#0f7285" : "#ffffff";
        const textColor = color === "#ffffff" ? "#18202a" : "#ffffff";
        const label = (labels[name] || [name]).map((line, index) => (
          `<tspan x="${{x}}" dy="${{index === 0 ? -7 : 15}}">${{line}}</tspan>`
        )).join("");
        return `
          <g>
            <rect x="${{x - 76}}" y="${{y - 34}}" width="152" height="68" rx="8" fill="${{color}}" stroke="#667085" />
            <text x="${{x}}" y="${{y - 2}}" text-anchor="middle" font-size="12.5" font-weight="700" fill="${{textColor}}">${{label}}</text>
            <text x="${{x}}" y="${{y + 25}}" text-anchor="middle" font-size="10.5" fill="${{textColor}}">lien risque ${{fmt(corr, 2)}}</text>
          </g>
        `;
      }}).join("");
      svg.innerHTML = `
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#98a2b3"></path>
          </marker>
        </defs>
        <rect x="38" y="28" width="620" height="330" rx="10" fill="#f8fafc" stroke="#e4e7ec" />
        <rect x="680" y="92" width="214" height="196" rx="10" fill="#fff8f8" stroke="#ffd0d0" />
        <text x="120" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#667085">Observations</text>
        <text x="440" y="32" text-anchor="middle" font-size="13" font-weight="700" fill="#667085">Indices de danger</text>
        <text x="790" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#667085">Decision</text>
        ${{edgeLines}}
        ${{nodes}}
      `;
      document.getElementById("graphNote").textContent = data.causal_graph.note;
    }}

    function renderWindowTable() {{
      document.getElementById("windowTable").innerHTML = topWindows().map(window => {{
        const reason = window.top_reasons[0]?.reason || "aucune";
        const riskClass = window.risk_probability >= .75 ? "risk-high" : window.risk_probability >= .45 ? "risk-mid" : "";
        return `
          <tr>
            <td>${{fmt(window.start_time_s, 2)}}-${{fmt(window.end_time_s, 2)}}s</td>
            <td class="${{riskClass}}">${{pct(window.risk_probability)}}</td>
            <td>${{window.unique_vehicle_count}}</td>
            <td>${{reasonLabel(reason)}}</td>
          </tr>
        `;
      }}).join("");
    }}

    renderKpis();
    renderEvents();
    renderRiskChart();
    renderGraph();
    renderWindowTable();
  </script>
</body>
</html>
"""


def write_json(path, value):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    data = load_json(args.data)
    black_box_entries = load_json(args.black_box)
    windows = aggregate_windows(data, args.window_seconds)
    explanations = build_explanations(black_box_entries, windows)
    causal_graph = build_causal_graph(windows)

    write_json(args.window_output, windows)
    write_json(args.explanations_output, explanations)

    dashboard_path = Path(args.dashboard_output)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(build_html(data, windows, explanations, causal_graph, args.video), encoding="utf-8")

    print(f"Window features: {Path(args.window_output).resolve()}")
    print(f"Explanations: {Path(args.explanations_output).resolve()}")
    print(f"Dashboard: {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
