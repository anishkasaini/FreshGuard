
from pathlib import Path
import sqlite3
import uuid
import os

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "freshguard-local-development-key"
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

DB = ROOT / "instance" / "freshguard.db"
MODEL_PATH = ROOT / "models" / "freshguard_model.joblib"
UPLOAD_DIR = ROOT / "static" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CONFIDENCE_THRESHOLD = 60.0

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

PAGES = [
    "dashboard",
    "scanner",
    "inventory",
    "storage",
    "analytics",
    "history"
]


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                label TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL,
                quantity TEXT,
                storage TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def extract_features(path):
    image = Image.open(path).convert("RGB").resize((96, 96))
    array = np.asarray(image)

    features = hog(
        array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )

    return features.reshape(1, -1)


def get_stats():
    with get_db() as conn:
        total_scans = conn.execute(
            "SELECT COUNT(*) FROM scans"
        ).fetchone()[0]

        fresh = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE label LIKE 'fresh_%'"
        ).fetchone()[0]

        rotten = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE label LIKE 'rotten_%'"
        ).fetchone()[0]

        inventory_count = conn.execute(
            "SELECT COUNT(*) FROM inventory"
        ).fetchone()[0]

        recent = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 5"
        ).fetchall()

    return {
        "total_scans": total_scans,
        "fresh": fresh,
        "rotten": rotten,
        "inventory_count": inventory_count,
        "recent": recent
    }


@app.route("/", defaults={"page": "dashboard"}, methods=["GET", "POST"])
@app.route("/<page>", methods=["GET", "POST"])
def pages(page):
    if page not in PAGES:
        return "Page not found", 404

    result = None

    if request.method == "POST":

        if page == "scanner":
            file = request.files.get("food_image")

            if not file or not file.filename:
                flash("Please select an image first.")
                return redirect(url_for("pages", page=page))

            extension = Path(file.filename).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                flash("Please upload JPG, PNG or WEBP.")
                return redirect(url_for("pages", page=page))

            safe_name = secure_filename(file.filename)

            if not safe_name:
                flash("Invalid image filename.")
                return redirect(url_for("pages", page=page))

            filename = f"{uuid.uuid4().hex}_{safe_name}"
            image_path = UPLOAD_DIR / filename

            try:
                file.save(image_path)

                with Image.open(image_path) as uploaded_image:
                    uploaded_image.verify()

                features = extract_features(image_path)
                prediction = str(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities)) * 100

                uncertain = confidence < CONFIDENCE_THRESHOLD

                with get_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO scans (filename, label, confidence)
                        VALUES (?, ?, ?)
                        """,
                        (filename, prediction, confidence)
                    )

                result = {
                    "label": (
                        "Uncertain - Retest Recommended"
                        if uncertain
                        else prediction.replace("_", " ").title()
                    ),
                    "confidence": round(confidence, 2),
                    "image": filename,
                    "fresh": prediction.startswith("fresh") and not uncertain,
                    "uncertain": uncertain
                }

            except (UnidentifiedImageError, OSError):
                if image_path.exists():
                    image_path.unlink()
                flash("Invalid or damaged image. Please try another image.")

            except Exception:
                if image_path.exists():
                    image_path.unlink()
                app.logger.exception("Scanner error")
                flash("Image processing failed. Please try another image.")

        elif page == "inventory":
            food_name = request.form.get("food_name", "").strip()
            quantity = request.form.get("quantity", "").strip()
            storage = request.form.get("storage", "").strip()

            if not food_name:
                flash("Please enter a food name.")

            elif len(food_name) > 100 or len(quantity) > 100:
                flash("Please enter shorter food details.")

            else:
                with get_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO inventory (food_name, quantity, storage)
                        VALUES (?, ?, ?)
                        """,
                        (food_name, quantity, storage)
                    )

                flash("Food item added successfully!")

            return redirect(url_for("pages", page=page))

        elif page == "storage":
            try:
                temperature = float(request.form.get("temperature", ""))
                humidity = float(request.form.get("humidity", ""))
                duration = float(request.form.get("duration", ""))

                if not all(np.isfinite(v) for v in (
                    temperature, humidity, duration
                )):
                    raise ValueError("Enter finite numeric values.")

                if not 0 <= humidity <= 100:
                    raise ValueError("Humidity must be between 0 and 100.")

                if not 0 <= duration <= 365:
                    raise ValueError("Duration must be between 0 and 365.")

                if not -30 <= temperature <= 60:
                    raise ValueError("Enter a realistic temperature.")

                if temperature > 25 or humidity > 75 or duration > 5:
                    status = "Higher storage risk"
                    advice = "Review storage conditions and inspect the food."

                elif temperature > 10 or humidity > 60:
                    status = "Moderate storage risk"
                    advice = "Monitor the food and follow suitable storage guidance."

                else:
                    status = "Lower storage risk"
                    advice = "Conditions appear more suitable based on these inputs."

                result = {
                    "status": status,
                    "advice": advice,
                    "temperature": temperature,
                    "humidity": humidity,
                    "duration": duration
                }

            except ValueError as error:
                flash(f"Please enter valid values: {error}")

    with get_db() as conn:
        inventory = conn.execute(
            "SELECT * FROM inventory ORDER BY id DESC"
        ).fetchall()

        history = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC"
        ).fetchall()

        class_counts = conn.execute("""
            SELECT label, COUNT(*) AS total
            FROM scans
            GROUP BY label
            ORDER BY total DESC
        """).fetchall()

    stats = get_stats()

    return render_template(
        "app.html",
        page=page,
        stats=stats,
        result=result,
        inventory=inventory,
        history=history,
        class_counts=class_counts
    )


@app.route("/delete-item/<int:item_id>", methods=["POST"])
def delete_item(item_id):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM inventory WHERE id = ?",
            (item_id,)
        )

    flash("Inventory item deleted.")
    return redirect(url_for("pages", page="inventory"))


@app.route("/delete-scan/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id):
    with get_db() as conn:
        scan = conn.execute(
            "SELECT filename FROM scans WHERE id = ?",
            (scan_id,)
        ).fetchone()

        conn.execute(
            "DELETE FROM scans WHERE id = ?",
            (scan_id,)
        )

    if scan and scan["filename"]:
        image_path = UPLOAD_DIR / scan["filename"]

        if image_path.exists():
            image_path.unlink()

    flash("Scan record deleted.")
    return redirect(url_for("pages", page="history"))


@app.errorhandler(413)
def file_too_large(error):
    flash("Image is too large. Maximum allowed size is 8 MB.")
    return redirect(url_for("pages", page="scanner"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)
