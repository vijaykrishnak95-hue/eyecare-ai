import os

from flask import (
    Flask, render_template, request, send_from_directory,
    redirect, url_for, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from database import db, PredictionHistory, User, LoginHistory
from datetime import datetime
from flask import send_file
from openpyxl import Workbook

from src.prediction.predict import predict_image
from src.xai.gradcam import generate_gradcam


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-only-change-before-deployment"
)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + str(BASE_DIR / "eye_disease.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = BASE_DIR / "results"

@app.route("/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route("/uploads/<path:filename>")
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template(
        "index.html",
        username=session.get("username")
    )

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must contain at least 8 characters."
            )

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            return render_template(
                "register.html",
                error="Username or email is already registered."
            )

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        return (
            "Registration successful! "
            '<a href="/login">Go to login</a>'
        )

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_record = LoginHistory(
                user_id=user.id,
                email=user.email,
                username=user.username,
                success=True
            )

            db.session.add(login_record)
            db.session.commit()

            session["user_id"] = user.id
            session["username"] = user.username

            return redirect(url_for("home"))

        login_record = LoginHistory(
            user_id=user.id if user else None,
            email=email,
            username=user.username if user else None,
            success=False,
            failure_reason=(
                "Incorrect password" if user else "Unknown email"
            )
        )

        db.session.add(login_record)
        db.session.commit()

        return render_template(
            "login.html",
            error="Invalid email or password."
        )

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
def prediction_history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    records = PredictionHistory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        PredictionHistory.id.desc()
    ).all()

    return render_template("history.html", records=records)



@app.route("/upload", methods=["POST"])
def upload_image():
    if "eye_image" not in request.files:
        return "No image was selected. Please go back and choose an image.", 400

    file = request.files["eye_image"]

    if file.filename == "":
        return "No image was selected. Please go back and choose an image.", 400

    if not allowed_file(file.filename):
        return "Please upload a PNG, JPG, or JPEG image.", 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        result = predict_image(save_path)
        gradcam_result = generate_gradcam(save_path)
        history = PredictionHistory(
            user_id=session.get("user_id"),
            filename=filename,
            disease=result["disease"],
            confidence=result["confidence"]
        )

        db.session.add(history)
        db.session.commit()

        return render_template(
            "result.html",
            filename=filename,
            disease=result["disease"].replace("_", " ").title(),
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            gradcam_heatmap=gradcam_result["heatmap_path"],
            gradcam_overlay=gradcam_result["overlay_path"]
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return (
            "The image was uploaded, but prediction failed. "
            "Please check the model checkpoint and terminal error.",
            500
        )


@app.errorhandler(413)
def file_too_large(error):
    return "The image is too large. Maximum upload size is 10 MB.", 413


@app.route("/admin")
def admin_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    current_user = db.session.get(User, session["user_id"])

    if current_user is None or not current_user.is_admin:
        return "Access denied: administrators only.", 403

    total_users = User.query.count()
    total_predictions = PredictionHistory.query.count()

    disease_counts = db.session.query(
        PredictionHistory.disease,
        db.func.count(PredictionHistory.id)
    ).group_by(PredictionHistory.disease).all()

    recent_predictions = PredictionHistory.query.order_by(
        PredictionHistory.id.desc()
    ).limit(10).all()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        disease_counts=disease_counts,
        recent_predictions=recent_predictions
    )

@app.route("/disease-info")
def disease_info():
    return render_template("disease_info.html")


@app.route("/admin/export-excel")
def export_admin_excel():
    if "user_id" not in session:
        return redirect(url_for("login"))

    current_user = db.session.get(User, session["user_id"])

    if current_user is None or not current_user.is_admin:
        return "Access denied: administrators only.", 403

    workbook = Workbook()

    # Sheet 1: Registered users
    users_sheet = workbook.active
    users_sheet.title = "Registered Users"
    users_sheet.append([
        "User ID", "Username", "Email", "Is Admin", "Created At"
    ])

    for user in User.query.order_by(User.id).all():
        users_sheet.append([
            user.id,
            user.username,
            user.email,
            user.is_admin,
            user.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if user.created_at else ""
        ])

    # Sheet 2: Login attempts
    login_sheet = workbook.create_sheet("Login History")
    login_sheet.append([
        "Attempt ID", "Email", "Successful", "Attempted At",
        "Failure Reason"
    ])

    for item in LoginHistory.query.order_by(LoginHistory.id).all():
        login_sheet.append([
            item.id,
            item.email,
            item.success,
            item.attempted_at.strftime("%Y-%m-%d %H:%M:%S")
            if item.attempted_at else "",
            item.failure_reason or ""
        ])

    # Sheet 3: Prediction history
    predictions_sheet = workbook.create_sheet("Predictions")
    predictions_sheet.append([
        "Prediction ID", "User ID", "Filename", "Disease",
        "Confidence (%)", "Uploaded At"
    ])

    for item in PredictionHistory.query.order_by(PredictionHistory.id).all():
        predictions_sheet.append([
            item.id,
            item.user_id,
            item.filename,
            item.disease,
            item.confidence,
            item.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")
            if item.uploaded_at else ""
        ])

    # Make column headings readable
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            max_length = max(
                len(str(cell.value or "")) for cell in column_cells
            )
            sheet.column_dimensions[
                column_cells[0].column_letter
            ].width = min(max_length + 2, 40)

    output = BASE_DIR / "admin_export.xlsx"
    workbook.save(output)

    return send_file(
        output,
        as_attachment=True,
        download_name="EyeCareAI_Database_Export.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


if __name__ == "__main__":
    app.run(debug=True)