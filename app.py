from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, jsonify, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

from vision import analyze_image


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "static" / "captures"
OVERLAY_DIR = BASE_DIR / "static" / "overlays"
TIMELAPSE_DIR = BASE_DIR / "static" / "timelapses"
DB_PATH = BASE_DIR / "growth_logs.db"

db = SQLAlchemy()


class GrowthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    sprout_count = db.Column(db.Integer, nullable=False)
    total_white_area = db.Column(db.Float, nullable=False)
    expected_seed_count = db.Column(db.Integer, nullable=False, default=25)
    germination_percentage = db.Column(db.Float, nullable=False, default=0.0)
    treated_sprout_count = db.Column(db.Integer, nullable=False, default=0)
    treated_white_area = db.Column(db.Float, nullable=False, default=0.0)
    treated_germination_percentage = db.Column(db.Float, nullable=False, default=0.0)
    untreated_sprout_count = db.Column(db.Integer, nullable=False, default=0)
    untreated_white_area = db.Column(db.Float, nullable=False, default=0.0)
    untreated_germination_percentage = db.Column(db.Float, nullable=False, default=0.0)
    raw_image_path = db.Column(db.String(255), nullable=False)
    overlay_image_path = db.Column(db.String(255), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sprout_count": self.sprout_count,
            "total_white_area": self.total_white_area,
            "expected_seed_count": self.expected_seed_count,
            "germination_percentage": self.germination_percentage,
            "treated_sprout_count": self.treated_sprout_count,
            "treated_white_area": self.treated_white_area,
            "treated_germination_percentage": self.treated_germination_percentage,
            "untreated_sprout_count": self.untreated_sprout_count,
            "untreated_white_area": self.untreated_white_area,
            "untreated_germination_percentage": self.untreated_germination_percentage,
            "germination_difference": round(
                self.treated_germination_percentage - self.untreated_germination_percentage,
                2,
            ),
            "white_area_difference": round(self.treated_white_area - self.untreated_white_area, 2),
            "treated_growth_index": calculate_growth_index(self.treated_white_area, self.expected_seed_count),
            "untreated_growth_index": calculate_growth_index(self.untreated_white_area, self.expected_seed_count),
            "growth_percentage_difference": calculate_percentage_difference(
                self.treated_white_area,
                self.untreated_white_area,
            ),
            "raw_image_path": self.raw_image_path,
            "overlay_image_path": self.overlay_image_path,
        }


class EnvironmentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    temperature_c = db.Column(db.Float, nullable=False)
    humidity_percent = db.Column(db.Float, nullable=False)
    temperature_score = db.Column(db.Float, nullable=False)
    humidity_score = db.Column(db.Float, nullable=False)
    germination_suitability = db.Column(db.Float, nullable=False)
    moisture_stress = db.Column(db.Float, nullable=False)
    vapor_pressure_deficit = db.Column(db.Float, nullable=False)
    risk_label = db.Column(db.String(80), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "temperature_score": self.temperature_score,
            "humidity_score": self.humidity_score,
            "germination_suitability": self.germination_suitability,
            "moisture_stress": self.moisture_stress,
            "vapor_pressure_deficit": self.vapor_pressure_deficit,
            "risk_label": self.risk_label,
        }


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    ensure_project_folders()

    with app.app_context():
        db.create_all()
        ensure_growth_log_columns()

    @app.route("/")
    def index():
        logs = GrowthLog.query.order_by(GrowthLog.timestamp.desc()).limit(10).all()
        latest = logs[0] if logs else None
        comparison = build_comparison_summary(days=3)
        environment = get_or_create_environment_snapshot()
        return render_template(
            "index.html",
            logs=logs,
            latest=latest,
            comparison=comparison,
            environment=environment,
        )

    @app.route("/upload_image", methods=["POST"])
    def upload_image():
        image_bytes = read_incoming_image()
        if not image_bytes:
            return jsonify({"error": "No JPEG image data received."}), 400

        timestamp = datetime.now(timezone.utc)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.jpg"
        raw_path = CAPTURE_DIR / filename
        overlay_path = OVERLAY_DIR / filename

        raw_path.write_bytes(image_bytes)

        try:
            result = analyze_image(raw_path, overlay_path)
        except ValueError as exc:
            raw_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 422

        expected_seed_count = read_expected_seed_count()
        treated_germination = calculate_germination_percentage(result.treated_sprout_count, expected_seed_count)
        untreated_germination = calculate_germination_percentage(result.untreated_sprout_count, expected_seed_count)
        germination_percentage = calculate_germination_percentage(result.sprout_count, expected_seed_count * 2)

        log = GrowthLog(
            timestamp=timestamp,
            sprout_count=result.sprout_count,
            total_white_area=result.total_white_area,
            expected_seed_count=expected_seed_count,
            germination_percentage=germination_percentage,
            treated_sprout_count=result.treated_sprout_count,
            treated_white_area=result.treated_white_area,
            treated_germination_percentage=treated_germination,
            untreated_sprout_count=result.untreated_sprout_count,
            untreated_white_area=result.untreated_white_area,
            untreated_germination_percentage=untreated_germination,
            raw_image_path=static_path(raw_path),
            overlay_image_path=static_path(overlay_path),
        )
        db.session.add(log)
        db.session.commit()

        payload = log.to_dict()
        payload["overlay_url"] = url_for("static", filename=f"overlays/{overlay_path.name}")
        payload["raw_url"] = url_for("static", filename=f"captures/{raw_path.name}")
        payload["comparison"] = build_comparison_summary(days=3)
        return jsonify(payload), 201

    @app.route("/api/logs")
    def api_logs():
        logs = GrowthLog.query.order_by(GrowthLog.timestamp.desc()).limit(50).all()
        return jsonify([log.to_dict() for log in logs])

    @app.route("/api/comparison")
    def api_comparison():
        days = request.args.get("days", 3, type=int)
        return jsonify(build_comparison_summary(days=max(days, 1)))

    @app.route("/api/latest")
    def api_latest():
        log = GrowthLog.query.order_by(GrowthLog.timestamp.desc()).first()
        if not log:
            return jsonify(None)
        payload = log.to_dict()
        payload["overlay_url"] = url_for("static", filename=log.overlay_image_path.replace("static/", ""))
        payload["raw_url"] = url_for("static", filename=log.raw_image_path.replace("static/", ""))
        return jsonify(payload)

    @app.route("/api/environment")
    def api_environment():
        return jsonify(create_environment_log().to_dict())

    @app.route("/api/environment/simulate", methods=["POST"])
    def api_environment_simulate():
        temperature = request.args.get("temperature", type=float)
        humidity = request.args.get("humidity", type=float)
        return jsonify(create_environment_log(temperature, humidity).to_dict()), 201

    @app.route("/create_timelapse", methods=["POST"])
    def create_timelapse():
        days = request.args.get("days", 3, type=int)
        fps = request.args.get("fps", 6, type=int)
        result = build_timelapse(days=max(days, 1), fps=max(fps, 1))
        if "error" in result:
            return jsonify(result), 400
        result["timelapse_url"] = url_for("static", filename=f"timelapses/{result['filename']}")
        return jsonify(result), 201

    @app.route("/clear_photos", methods=["POST"])
    def clear_photos():
        clear_image_history()
        return jsonify({"status": "cleared"}), 200

    return app


def ensure_project_folders():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    TIMELAPSE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "templates").mkdir(exist_ok=True)


def ensure_growth_log_columns():
    existing_columns = {
        row[1]
        for row in db.session.execute(text("PRAGMA table_info(growth_log)")).fetchall()
    }
    column_definitions = {
        "expected_seed_count": "INTEGER NOT NULL DEFAULT 25",
        "germination_percentage": "FLOAT NOT NULL DEFAULT 0.0",
        "treated_sprout_count": "INTEGER NOT NULL DEFAULT 0",
        "treated_white_area": "FLOAT NOT NULL DEFAULT 0.0",
        "treated_germination_percentage": "FLOAT NOT NULL DEFAULT 0.0",
        "untreated_sprout_count": "INTEGER NOT NULL DEFAULT 0",
        "untreated_white_area": "FLOAT NOT NULL DEFAULT 0.0",
        "untreated_germination_percentage": "FLOAT NOT NULL DEFAULT 0.0",
    }

    for column_name, definition in column_definitions.items():
        if column_name not in existing_columns:
            db.session.execute(text(f"ALTER TABLE growth_log ADD COLUMN {column_name} {definition}"))

    db.session.commit()


def read_incoming_image():
    if request.files:
        image_file = request.files.get("image") or next(iter(request.files.values()))
        return image_file.read()

    return request.get_data(cache=False)


def read_expected_seed_count():
    candidates = [
        request.args.get("seed_count"),
        request.form.get("seed_count") if request.form else None,
        request.headers.get("X-Seed-Count"),
    ]

    for value in candidates:
        if value is None:
            continue
        try:
            seed_count = int(value)
        except ValueError:
            continue
        if seed_count > 0:
            return seed_count

    return 25


def calculate_germination_percentage(sprout_count, expected_seed_count):
    percentage = (sprout_count / expected_seed_count) * 100
    return round(min(percentage, 100.0), 2)


def calculate_growth_index(white_area, expected_seed_count):
    if expected_seed_count <= 0:
        return 0.0
    return round(white_area / expected_seed_count, 2)


def calculate_percentage_difference(treated_value, control_value):
    if control_value <= 0:
        return 100.0 if treated_value > 0 else 0.0
    return round(((treated_value - control_value) / control_value) * 100, 2)


def get_or_create_environment_snapshot():
    latest = EnvironmentLog.query.order_by(EnvironmentLog.timestamp.desc()).first()
    if latest and datetime.now(timezone.utc) - as_utc(latest.timestamp) < timedelta(minutes=5):
        return latest
    return create_environment_log()


def create_environment_log(temperature=None, humidity=None):
    if temperature is None or humidity is None:
        simulated = simulate_chickpea_environment()
        temperature = simulated["temperature_c"]
        humidity = simulated["humidity_percent"]

    metrics = calculate_chickpea_environment_metrics(temperature, humidity)
    log = EnvironmentLog(
        temperature_c=metrics["temperature_c"],
        humidity_percent=metrics["humidity_percent"],
        temperature_score=metrics["temperature_score"],
        humidity_score=metrics["humidity_score"],
        germination_suitability=metrics["germination_suitability"],
        moisture_stress=metrics["moisture_stress"],
        vapor_pressure_deficit=metrics["vapor_pressure_deficit"],
        risk_label=metrics["risk_label"],
    )
    db.session.add(log)
    db.session.commit()
    return log


def simulate_chickpea_environment():
    now = datetime.now(timezone.utc)
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    day_angle = (seconds / 86400) * 2 * math.pi
    short_wave = (seconds / 90) * 2 * math.pi
    temperature = 24.5 + 3.2 * math.sin(day_angle - math.pi / 3) + 0.45 * math.sin(short_wave)
    humidity = 68 - 8.5 * math.sin(day_angle - math.pi / 4) + 1.8 * math.cos(short_wave * 0.8)
    return {
        "temperature_c": round(temperature, 1),
        "humidity_percent": round(min(max(humidity, 45), 88), 1),
    }


def calculate_chickpea_environment_metrics(temperature, humidity):
    temperature = round(float(temperature), 1)
    humidity = round(float(humidity), 1)
    temperature_score = triangular_score(temperature, low=15, optimum_low=22, optimum_high=28, high=35)
    humidity_score = triangular_score(humidity, low=45, optimum_low=60, optimum_high=75, high=90)
    suitability = round((temperature_score * 0.58) + (humidity_score * 0.42), 1)
    moisture_stress = round(max(0, 100 - humidity_score), 1)
    vpd = calculate_vapor_pressure_deficit(temperature, humidity)

    if temperature < 15:
        risk_label = "Cold delay risk"
    elif temperature > 35:
        risk_label = "Heat stress risk"
    elif humidity > 88:
        risk_label = "Excess moisture/fungal risk"
    elif humidity < 45:
        risk_label = "Drying stress risk"
    elif suitability >= 80:
        risk_label = "Favorable for chickpea germination"
    elif suitability >= 55:
        risk_label = "Moderate germination conditions"
    else:
        risk_label = "Suboptimal germination conditions"

    return {
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "temperature_score": temperature_score,
        "humidity_score": humidity_score,
        "germination_suitability": suitability,
        "moisture_stress": moisture_stress,
        "vapor_pressure_deficit": vpd,
        "risk_label": risk_label,
    }


def triangular_score(value, low, optimum_low, optimum_high, high):
    if value <= low or value >= high:
        return 0.0
    if optimum_low <= value <= optimum_high:
        return 100.0
    if value < optimum_low:
        return round(((value - low) / (optimum_low - low)) * 100, 1)
    return round(((high - value) / (high - optimum_high)) * 100, 1)


def calculate_vapor_pressure_deficit(temperature, humidity):
    saturation_vapor_pressure = 0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3))
    actual_vapor_pressure = saturation_vapor_pressure * (humidity / 100)
    return round(max(saturation_vapor_pressure - actual_vapor_pressure, 0), 2)


def build_comparison_summary(days=3):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = (
        GrowthLog.query.filter(GrowthLog.timestamp >= since)
        .order_by(GrowthLog.timestamp.asc())
        .all()
    )

    if len(logs) < 2:
        return empty_comparison(days, len(logs))

    first = logs[0]
    latest = logs[-1]
    elapsed_days = max((as_utc(latest.timestamp) - as_utc(first.timestamp)).total_seconds() / 86400, 1 / 86400)
    latest_germination_difference = round(
        latest.treated_germination_percentage - latest.untreated_germination_percentage,
        2,
    )
    latest_growth_difference = round(latest.treated_white_area - latest.untreated_white_area, 2)

    if elapsed_days < (1 / 24):
        summary = empty_comparison(days, len(logs))
        summary["elapsed_days"] = round(elapsed_days, 2)
        summary["latest_germination_difference"] = latest_germination_difference
        summary["latest_growth_difference"] = latest_growth_difference
        summary["leader"] = "Current difference is visible; speed needs at least 1 hour between frames."
        return summary

    treated_germination_speed = rate_per_day(
        latest.treated_germination_percentage,
        first.treated_germination_percentage,
        elapsed_days,
    )
    untreated_germination_speed = rate_per_day(
        latest.untreated_germination_percentage,
        first.untreated_germination_percentage,
        elapsed_days,
    )
    treated_growth_speed = rate_per_day(latest.treated_white_area, first.treated_white_area, elapsed_days)
    untreated_growth_speed = rate_per_day(latest.untreated_white_area, first.untreated_white_area, elapsed_days)

    return {
        "days": days,
        "sample_count": len(logs),
        "elapsed_days": round(elapsed_days, 2),
        "treated_germination_speed": treated_germination_speed,
        "untreated_germination_speed": untreated_germination_speed,
        "germination_speed_difference": round(treated_germination_speed - untreated_germination_speed, 2),
        "treated_growth_speed": treated_growth_speed,
        "untreated_growth_speed": untreated_growth_speed,
        "growth_speed_difference": round(treated_growth_speed - untreated_growth_speed, 2),
        "latest_germination_difference": latest_germination_difference,
        "latest_growth_difference": latest_growth_difference,
        "latest_growth_percentage_difference": calculate_percentage_difference(
            latest.treated_white_area,
            latest.untreated_white_area,
        ),
        "leader": choose_leader(latest),
    }


def empty_comparison(days, sample_count):
    return {
        "days": days,
        "sample_count": sample_count,
        "elapsed_days": 0,
        "treated_germination_speed": 0,
        "untreated_germination_speed": 0,
        "germination_speed_difference": 0,
        "treated_growth_speed": 0,
        "untreated_growth_speed": 0,
        "growth_speed_difference": 0,
        "latest_germination_difference": 0,
        "latest_growth_difference": 0,
        "latest_growth_percentage_difference": 0,
        "leader": "Collect at least two chickpea image frames to compare speed.",
    }


def rate_per_day(latest_value, first_value, elapsed_days):
    return round((latest_value - first_value) / elapsed_days, 2)


def as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def choose_leader(log):
    germination_gap = log.treated_germination_percentage - log.untreated_germination_percentage
    growth_gap = log.treated_white_area - log.untreated_white_area
    if abs(germination_gap) < 1 and abs(growth_gap) < 50:
        return "No clear difference yet."
    if germination_gap > 0 or growth_gap > 0:
        return "Treated chickpea batch is ahead."
    return "Control chickpea batch is ahead."


def build_timelapse(days=3, fps=6):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = (
        GrowthLog.query.filter(GrowthLog.timestamp >= since)
        .order_by(GrowthLog.timestamp.asc())
        .all()
    )

    if len(logs) < 2:
        return {"error": "At least two analyzed frames are needed to create a timelapse."}

    first_frame = None
    frame_size = None
    frames = []

    for log in logs:
        image_path = BASE_DIR / log.overlay_image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        if first_frame is None:
            first_frame = frame
            frame_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size)

        add_timelapse_caption(frame, log)
        frames.append(frame)

    if len(frames) < 2:
        return {"error": "Could not read enough overlay frames to create a timelapse."}

    filename = f"chickpea_timelapse_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = TIMELAPSE_DIR / filename
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

    for frame in frames:
        writer.write(frame)

    writer.release()

    return {
        "filename": filename,
        "frame_count": len(frames),
        "fps": fps,
        "days": days,
    }


def clear_image_history():
    for log in GrowthLog.query.all():
        for image_path in [log.raw_image_path, log.overlay_image_path]:
            path = BASE_DIR / image_path
            path.unlink(missing_ok=True)
        db.session.delete(log)

    for timelapse_path in TIMELAPSE_DIR.glob("*.mp4"):
        timelapse_path.unlink(missing_ok=True)

    db.session.commit()


def add_timelapse_caption(frame, log):
    text_lines = [
        log.timestamp.strftime("%Y-%m-%d %H:%M"),
        f"Treated chickpea: {log.treated_germination_percentage:.1f}% | {log.treated_white_area:.0f}px",
        f"Control chickpea: {log.untreated_germination_percentage:.1f}% | {log.untreated_white_area:.0f}px",
    ]
    padding = 12
    line_height = 28
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for line in text_lines)
    box_height = padding * 2 + line_height * len(text_lines)
    cv2.rectangle(frame, (10, frame.shape[0] - box_height - 10), (width + 34, frame.shape[0] - 10), (255, 255, 255), cv2.FILLED)

    y = frame.shape[0] - box_height + padding + 8
    for line in text_lines:
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (45, 106, 79), 2, cv2.LINE_AA)
        y += line_height


def static_path(path):
    return str(path.relative_to(BASE_DIR)).replace("\\", "/")


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
