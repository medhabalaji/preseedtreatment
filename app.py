from datetime import datetime, timedelta, timezone
from io import BytesIO
import math
import os
from pathlib import Path
import re
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash

from vision import analyze_image


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "static" / "captures"
OVERLAY_DIR = BASE_DIR / "static" / "overlays"
TIMELAPSE_DIR = BASE_DIR / "static" / "timelapses"
SPROUT_UPLOAD_DIR = BASE_DIR / "static" / "sprout_uploads"
DB_PATH = BASE_DIR / "growth_logs.db"

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


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


class SproutAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    image_path = db.Column(db.String(255), nullable=False)
    sprout_key = db.Column(db.String(80), nullable=False)
    sprout_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    visual_summary = db.Column(db.String(255), nullable=False)
    care_plan = db.Column(db.Text, nullable=False)
    harvest_plan = db.Column(db.Text, nullable=False)
    indian_food_ideas = db.Column(db.Text, nullable=False)
    nutrition_benefits = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "image_url": url_for("static", filename=self.image_path.replace("static/", "")) if self.image_path else None,
            "sprout_key": self.sprout_key,
            "sprout_name": self.sprout_name,
            "confidence": self.confidence,
            "visual_summary": self.visual_summary,
            "care_plan": split_lines(self.care_plan),
            "harvest_plan": split_lines(self.harvest_plan),
            "indian_food_ideas": split_lines(self.indian_food_ideas),
            "nutrition_benefits": split_lines(self.nutrition_benefits),
        }


SPROUT_LIBRARY = {
    "chickpea": {
        "name": "Chickpea / Chana sprout",
        "aliases": ["chickpea", "chana", "channa", "chole", "gram", "kabuli", "bengal"],
        "visual": "round beige seeds with thick pale shoots",
        "care": [
            "Soak 8-12 hours, rinse well, then drain completely.",
            "Keep in a breathable jar or cloth in indirect light at room temperature.",
            "Rinse and drain every 8-12 hours so the sprouts stay moist but not waterlogged.",
        ],
        "harvest": [
            "Harvest when shoots are about 0.5-1.5 cm for a sweet, nutty crunch.",
            "Refrigerate after a final rinse and use within 2-3 days.",
            "Light steaming is a good idea for sensitive stomachs.",
        ],
        "food": [
            "Kala chana sprout chaat with onion, tomato, coriander, lemon, and chaat masala.",
            "Sprouted chana usal with mustard seeds, curry leaves, coconut, and mild masala.",
            "Add to poha, upma, millet khichdi, or roti rolls for extra bite.",
            "Toss with cucumber, carrot, curd, and roasted cumin for a quick raita-style bowl.",
        ],
        "nutrition": [
            "Good plant protein and fibre for satiety.",
            "Provides folate, iron, magnesium, and slow-release carbohydrates.",
            "Sprouting improves digestibility and can improve mineral availability.",
        ],
    },
    "mung": {
        "name": "Green gram / Moong sprout",
        "aliases": ["mung", "moong", "green gram", "green"],
        "visual": "small green seed coats with slender white tails",
        "care": [
            "Soak 6-8 hours, drain, and keep the beans in a damp cloth or sprouting jar.",
            "Rinse twice daily; moong sprouts quickly and likes airflow.",
            "Keep away from harsh sunlight to avoid drying and bitterness.",
        ],
        "harvest": [
            "Harvest in 24-48 hours when tails are 1-3 cm.",
            "For a milder taste, stop early; for more crunch, let them lengthen another half day.",
            "Store chilled in a dry container after draining thoroughly.",
        ],
        "food": [
            "Classic moong sprout salad with kachumber, lemon, coriander, and kala namak.",
            "Matki-style misal base, but with moong for a lighter sprouted curry.",
            "Add to dosa batter, pesarattu, cheela, or adai for protein-rich breakfasts.",
            "Stir into bhel, sev puri topping, or curd rice just before serving.",
        ],
        "nutrition": [
            "Rich in fibre, vitamin C after sprouting, folate, and potassium.",
            "Light, quick-cooking protein source that works well in Indian breakfasts.",
            "Lower calorie density with good hydration and crunch.",
        ],
    },
    "fenugreek": {
        "name": "Fenugreek / Methi sprout",
        "aliases": ["fenugreek", "methi"],
        "visual": "small amber seeds with fine shoots and a bitter aroma",
        "care": [
            "Soak only 4-6 hours; methi can become slimy if kept too wet.",
            "Rinse gently twice daily and drain very well.",
            "Use a thin layer in the sprouter so air reaches all seeds.",
        ],
        "harvest": [
            "Harvest at 1-2 cm shoots, usually in 2-3 days.",
            "Taste before using; older methi sprouts become more bitter.",
            "Use fresh or lightly saute to soften the bitterness.",
        ],
        "food": [
            "Mix into methi sprout thepla dough with curd and ajwain.",
            "Use in koshimbir with grated carrot, coconut, lemon, and peanuts.",
            "Add a small handful to dal, sambar, or sprouts sabzi for a pleasantly bitter note.",
            "Blend into green chutney with coriander, mint, lemon, and green chilli.",
        ],
        "nutrition": [
            "Contains fibre and traditional bitter phytonutrients associated with glucose-friendly meals.",
            "Adds iron, magnesium, and distinctive digestive bitters.",
            "Best used in smaller portions because the flavour is strong.",
        ],
    },
    "lentil": {
        "name": "Lentil / Masoor sprout",
        "aliases": ["lentil", "masoor"],
        "visual": "flat lens-shaped seeds with short white shoots",
        "care": [
            "Soak 6-8 hours, rinse, drain, and spread loosely.",
            "Rinse twice daily; avoid crowding because lentils heat up when densely packed.",
            "Keep in shade with steady airflow.",
        ],
        "harvest": [
            "Harvest in 1-2 days for tender, slightly peppery sprouts.",
            "Cook briefly if adding to curries or feeding children.",
            "Refrigerate after draining and use within 2 days.",
        ],
        "food": [
            "Masoor sprout dal with tomato, garlic, cumin, and a short simmer.",
            "Use in sprout pulao with peas, carrots, turmeric, and garam masala.",
            "Fold into paratha stuffing with potato, coriander, and amchur.",
            "Make a warm sundal-style snack with mustard seeds, curry leaves, and coconut.",
        ],
        "nutrition": [
            "Good protein, folate, iron, and soluble fibre.",
            "Sprouting shortens cooking time and improves texture.",
            "Works well for balanced meals with rice, roti, or millets.",
        ],
    },
    "mustard": {
        "name": "Mustard / Rai sprout",
        "aliases": ["mustard", "rai", "sarson"],
        "visual": "tiny dark seeds with sharp peppery shoots",
        "care": [
            "Use a tray or mesh rather than a deep jar because mustard forms gel when wet.",
            "Mist lightly and keep the layer thin.",
            "Give indirect light after germination for greener micro-sprouts.",
        ],
        "harvest": [
            "Harvest young at 3-5 days when shoots are tender and spicy.",
            "Cut with clean scissors and rinse lightly.",
            "Use fresh; mustard sprouts lose punch quickly in the fridge.",
        ],
        "food": [
            "Top dahi puri, sev puri, or bhel with a small pinch for heat.",
            "Add to curd dips, raita, or cucumber salad.",
            "Use as a garnish on khichdi, dal, or sarson-flavoured saag bowls.",
            "Mix into chutney with coriander and lemon for a sharper finish.",
        ],
        "nutrition": [
            "Peppery cruciferous sprout with vitamin K, vitamin C, and glucosinolate compounds.",
            "Strong flavour means small amounts go a long way.",
            "Adds freshness without much oil, salt, or sugar.",
        ],
    },
}


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "sonicseed-local-dev-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    ensure_project_folders()

    with app.app_context():
        db.create_all()
        ensure_growth_log_columns()
        recalibrate_saved_sprout_analyses()

    @app.route("/")
    def index():
        if "user_id" not in session:
            return redirect(url_for("login"))
        logs = GrowthLog.query.order_by(GrowthLog.timestamp.desc()).limit(10).all()
        latest = logs[0] if logs else None
        comparison = build_comparison_summary(days=3)
        environment = get_or_create_environment_snapshot()
        sprout_analyses = SproutAnalysis.query.filter_by(user_id=session.get("user_id")).order_by(
            SproutAnalysis.timestamp.desc()
        ).limit(6).all()
        return render_template(
            "index.html",
            logs=logs,
            latest=latest,
            comparison=comparison,
            environment=environment,
            sprout_analyses=sprout_analyses,
            current_user=get_current_user(),
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = normalize_email(request.form.get("email", ""))
            password = request.form.get("password", "")
            user = User.query.filter_by(email=email).first()
            if user and user.verify_password(password):
                session["user_id"] = user.id
                session["user_name"] = user.name
                return redirect(url_for("index"))
            flash("Email or password is incorrect.", "danger")

        return render_template("login.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = normalize_email(request.form.get("email", ""))
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not name or not email or not password:
                flash("Name, email, and password are required.", "danger")
            elif not is_valid_email(email):
                flash("Enter a valid email address.", "danger")
            elif len(password) < 6:
                flash("Use at least 6 characters for the password.", "danger")
            elif password != confirm_password:
                flash("Passwords do not match.", "danger")
            elif User.query.filter_by(email=email).first():
                flash("That email is already registered. Please log in.", "warning")
                return redirect(url_for("login"))
            else:
                user = User(name=name, email=email, password_hash=generate_password_hash(password))
                db.session.add(user)
                db.session.commit()
                session["user_id"] = user.id
                session["user_name"] = user.name
                return redirect(url_for("index"))

        return render_template("register.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

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

    @app.route("/analyze_sprout", methods=["POST"])
    def analyze_sprout_upload():
        identification_mode = request.form.get("identification_mode", "image")
        seed_name = request.form.get("seed_name", "").strip()
        image_file = request.files.get("sprout_image")
        timestamp = datetime.now(timezone.utc)

        if identification_mode == "name":
            if not seed_name:
                return jsonify({"error": "Enter a seed or sprout name first."}), 400
            try:
                analysis_data = identify_sprout_by_name(seed_name)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 422
            image_path = ""
        else:
            if not image_file or not image_file.filename:
                return jsonify({"error": "Upload a sprout photo first."}), 400

            extension = Path(image_file.filename).suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                extension = ".jpg"

            filename = f"sprout_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}{extension}"
            upload_path = SPROUT_UPLOAD_DIR / filename
            image_file.save(upload_path)

            try:
                analysis_data = identify_sprout(upload_path, image_file.filename)
            except ValueError as exc:
                upload_path.unlink(missing_ok=True)
                return jsonify({"error": str(exc)}), 422
            image_path = static_path(upload_path)

        analysis = SproutAnalysis(
            user_id=session.get("user_id"),
            timestamp=timestamp,
            image_path=image_path,
            **analysis_data,
        )
        db.session.add(analysis)
        db.session.commit()
        return jsonify(analysis.to_dict()), 201

    @app.route("/download_sprout_report/<int:analysis_id>")
    def download_sprout_report(analysis_id):
        analysis = SproutAnalysis.query.get_or_404(analysis_id)
        current_user_id = session.get("user_id")
        if analysis.user_id and current_user_id and analysis.user_id != current_user_id:
            return jsonify({"error": "That report belongs to another account."}), 403

        pdf = build_sprout_report_pdf(analysis)
        filename = f"{slugify_filename(analysis.sprout_name)}-sprout-report.pdf"
        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )

    return app


def ensure_project_folders():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    TIMELAPSE_DIR.mkdir(parents=True, exist_ok=True)
    SPROUT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def normalize_email(email):
    return email.strip().lower()


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def split_lines(value):
    if not value:
        return []
    return [line for line in value.split("\n") if line]


def recalibrate_saved_sprout_analyses():
    changed = False
    for analysis in SproutAnalysis.query.all():
        if not analysis.image_path:
            continue
        image_path = BASE_DIR / analysis.image_path
        if not image_path.exists():
            db.session.delete(analysis)
            changed = True
            continue
        try:
            updated = identify_sprout(image_path)
        except ValueError:
            continue

        if analysis.sprout_key == updated["sprout_key"]:
            continue

        for key, value in updated.items():
            setattr(analysis, key, value)
        changed = True

    if changed:
        db.session.commit()


def identify_sprout(image_path, original_filename=""):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read that image. Try a clearer JPG or PNG.")

    metrics = extract_sprout_image_metrics(image)
    scores = score_sprout_candidates(metrics, original_filename)
    sprout_key, score = max(scores.items(), key=lambda item: item[1])
    profile = SPROUT_LIBRARY[sprout_key]
    confidence = min(96, max(58, round(score)))

    if confidence < 64:
        visual_summary = (
            "The photo has mixed visual signals, so this is a best-fit identification. "
            f"It most closely matches {profile['visual']}."
        )
    else:
        visual_summary = f"The image most closely matches {profile['visual']}."

    return build_sprout_profile_response(sprout_key, confidence, visual_summary)


def identify_sprout_by_name(seed_name):
    normalized_name = seed_name.strip().lower()
    if not normalized_name:
        raise ValueError("Enter a seed or sprout name first.")

    for sprout_key, profile in SPROUT_LIBRARY.items():
        searchable_names = [profile["name"].lower(), *profile["aliases"]]
        if any(alias == normalized_name or alias in normalized_name for alias in searchable_names):
            return build_sprout_profile_response(
                sprout_key,
                confidence=100,
                visual_summary="Guidance is based on the seed or sprout name you entered.",
            )

    supported_names = ", ".join(profile["name"] for profile in SPROUT_LIBRARY.values())
    raise ValueError(f"That seed is not in the guide yet. Try one of: {supported_names}.")


def build_sprout_profile_response(sprout_key, confidence, visual_summary):
    profile = SPROUT_LIBRARY[sprout_key]
    return {
        "sprout_key": sprout_key,
        "sprout_name": profile["name"],
        "confidence": confidence,
        "visual_summary": visual_summary,
        "care_plan": "\n".join(profile["care"]),
        "harvest_plan": "\n".join(profile["harvest"]),
        "indian_food_ideas": "\n".join(profile["food"]),
        "nutrition_benefits": "\n".join(profile["nutrition"]),
    }


def slugify_filename(value):
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "sonicseed"


def build_sprout_report_pdf(analysis):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
        title=f"{analysis.sprout_name} report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SonicTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1f5d43"),
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "SonicSubtitle",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#5f7068"),
        fontSize=10.5,
        leading=15,
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "SonicSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#2d6a4f"),
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=4,
        spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "SonicBody",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#2f463c"),
        fontSize=10,
        leading=14,
    )
    chip_style = ParagraphStyle(
        "SonicChip",
        parent=styles["BodyText"],
        textColor=colors.HexColor("#1f5d43"),
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
    )

    story = [
        Paragraph("SonicSeed Sprout Intelligence Report", title_style),
        Paragraph(
            f"{analysis.sprout_name} &nbsp; | &nbsp; {analysis.confidence:.0f}% confidence &nbsp; | &nbsp; "
            f"{analysis.timestamp.strftime('%Y-%m-%d %H:%M')}",
            subtitle_style,
        ),
        Table(
            [
                [
                    Paragraph("Likely sprout", chip_style),
                    Paragraph(analysis.sprout_name, body_style),
                    Paragraph("Signal", chip_style),
                    Paragraph(analysis.visual_summary, body_style),
                ]
            ],
            colWidths=[1.1 * inch, 1.6 * inch, 0.75 * inch, 3.05 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf8f2")),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#b7e4c7")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8eadf")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            ),
        ),
        Spacer(1, 0.18 * inch),
    ]

    for title, lines in [
        ("Care & Growing", split_lines(analysis.care_plan)),
        ("Harvest Timing", split_lines(analysis.harvest_plan)),
        ("Indian Food Ideas", split_lines(analysis.indian_food_ideas)),
        ("Nutrition Notes", split_lines(analysis.nutrition_benefits)),
    ]:
        story.extend(build_pdf_section(title, lines, section_style, body_style))

    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Keep sprouts clean, rinse thoroughly, and lightly cook when serving children, elders, pregnant people, or anyone with a sensitive stomach.",
            subtitle_style,
        )
    )

    document.build(story, onFirstPage=draw_pdf_frame, onLaterPages=draw_pdf_frame)
    buffer.seek(0)
    return buffer


def build_pdf_section(title, lines, section_style, body_style):
    rows = [[Paragraph(title, section_style)]]
    for line in lines:
        rows.append([Paragraph(f"&#8226; {line}", body_style)])

    return [
        Table(
            rows,
            colWidths=[6.55 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#d8eadf")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#e8ecef")),
                ]
            ),
        ),
        Spacer(1, 0.12 * inch),
    ]


def draw_pdf_frame(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#f4fbf7"))
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#d8eadf"))
    canvas.setLineWidth(1)
    canvas.roundRect(0.35 * inch, 0.32 * inch, width - 0.7 * inch, height - 0.64 * inch, 18, stroke=1, fill=0)
    canvas.setFillColor(colors.HexColor("#d9f0e4"))
    canvas.circle(width - 0.75 * inch, height - 0.55 * inch, 0.18 * inch, fill=1, stroke=0)
    canvas.circle(0.72 * inch, 0.56 * inch, 0.12 * inch, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#74c69d"))
    canvas.setLineWidth(2)
    canvas.bezier(width - 1.35 * inch, height - 0.78 * inch, width - 1.05 * inch, height - 1.12 * inch, width - 0.7 * inch, height - 1.05 * inch, width - 0.48 * inch, height - 1.28 * inch)
    canvas.restoreState()


def extract_sprout_image_metrics(image):
    resized = resize_for_analysis(image)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    total_pixels = resized.shape[0] * resized.shape[1]

    green_mask = ((hue >= 35) & (hue <= 90) & (saturation > 45) & (value > 45))
    olive_green_mask = ((hue >= 18) & (hue <= 38) & (saturation > 70) & (value > 50))
    yellow_mask = ((hue >= 14) & (hue <= 35) & (saturation > 35) & (value > 50))
    beige_mask = ((hue >= 10) & (hue <= 32) & (saturation > 18) & (saturation < 95) & (value > 90))
    dark_mask = value < 75
    white_mask = (saturation < 45) & (value > 150)

    edges = cv2.Canny(gray, 70, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roundish_count = 0
    elongated_count = 0
    tiny_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = max(width, height) / max(min(width, height), 1)
        if area < 120:
            tiny_count += 1
        if aspect < 1.55:
            roundish_count += 1
        elif aspect > 2.2:
            elongated_count += 1

    return {
        "green_ratio": float(np.count_nonzero(green_mask) / total_pixels),
        "olive_green_ratio": float(np.count_nonzero(olive_green_mask) / total_pixels),
        "yellow_ratio": float(np.count_nonzero(yellow_mask) / total_pixels),
        "beige_ratio": float(np.count_nonzero(beige_mask) / total_pixels),
        "dark_ratio": float(np.count_nonzero(dark_mask) / total_pixels),
        "white_ratio": float(np.count_nonzero(white_mask) / total_pixels),
        "roundish_count": roundish_count,
        "elongated_count": elongated_count,
        "tiny_count": tiny_count,
    }


def resize_for_analysis(image):
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= 900:
        return image
    scale = 900 / max_side
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def score_sprout_candidates(metrics, original_filename):
    filename = original_filename.lower()
    scores = {key: 45.0 for key in SPROUT_LIBRARY}

    for key, profile in SPROUT_LIBRARY.items():
        if any(alias in filename for alias in profile["aliases"]):
            scores[key] += 34

    green = metrics["green_ratio"]
    olive_green = metrics["olive_green_ratio"]
    yellow = metrics["yellow_ratio"]
    beige = metrics["beige_ratio"]
    dark = metrics["dark_ratio"]
    white = metrics["white_ratio"]
    roundish = metrics["roundish_count"]
    elongated = metrics["elongated_count"]
    tiny = metrics["tiny_count"]

    scores["mung"] += olive_green * 105 + green * 45 + white * 14 + min(elongated, 22) * 1.1
    scores["chickpea"] += beige * 54 + min(roundish, 16) * 0.9 + white * 7
    scores["fenugreek"] += yellow * 42 + min(tiny, 24) * 0.9 + dark * 10
    scores["lentil"] += min(roundish, 20) * 0.7 + white * 16 + yellow * 16
    scores["mustard"] += dark * 34 + min(tiny, 30) * 1.2 + green * 20

    if (green + olive_green) > 0.18 and white > 0.08:
        scores["mung"] += 14
    if olive_green > 0.22 and tiny > 35:
        scores["mung"] += 34
        scores["chickpea"] -= 28
    if beige > 0.15 and white > 0.35 and dark < 0.08 and roundish >= 12:
        scores["chickpea"] += 38
        scores["fenugreek"] -= 32
    if yellow > 0.25 and white > 0.35 and dark < 0.06:
        scores["chickpea"] += 18
        scores["fenugreek"] -= 18
    if beige > 0.12 and roundish > elongated and olive_green < 0.12:
        scores["chickpea"] += 8
    if dark > 0.22 and tiny > 8:
        scores["mustard"] += 10

    return scores


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
