from datetime import timedelta
import os
import mimetypes
from module3_data_collection import collect_data
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_sqlalchemy import SQLAlchemy
import traceback
from sentence_transformers import SentenceTransformer, util



app = Flask(__name__)
CORS(app)

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "medgraphx.db")
DATASET_PATH = os.path.join(BASE_DIR, "medgraphx_300_profiles_dataset.xlsx")
PROFILE_PICS_DIR = os.path.join(BASE_DIR, "profile_pics")

os.makedirs(PROFILE_PICS_DIR, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "medgraphx_secret_key_123_very_secure_key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=2)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)


# ----------------------------
# DATABASE MODELS
# ----------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")

    profile = db.relationship(
        "PatientProfile",
        backref="user",
        uselist=False,
        cascade="all, delete-orphan"
    )


class PatientProfile(db.Model):
    __tablename__ = "patient_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)

    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    diseases = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    lifestyle = db.Column(db.Text, nullable=True)
    height_cm = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)
    diet_preference = db.Column(db.String(100), nullable=True)

class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    risk_analysis = db.Column(db.Integer)
    nutrient_analysis = db.Column(db.Integer)
    knowledge_mapping = db.Column(db.Integer)
    data_extraction = db.Column(db.Integer)
    meal_planning = db.Column(db.Integer)

    comments = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ----------------------------
# HELPERS
# ----------------------------
def json_error(message, code=400):
    return jsonify({"error": message}), code


def serialize_profile(profile):
    return {
        "age": profile.age if profile and profile.age is not None else 0,
        "gender": profile.gender if profile and profile.gender else "Male",
        "diseases": profile.diseases if profile and profile.diseases else "",
        "medications": profile.medications if profile and profile.medications else "",
        "allergies": profile.allergies if profile and profile.allergies else "",
        "lifestyle": profile.lifestyle if profile and profile.lifestyle else "",
        "height_cm": profile.height_cm if profile and profile.height_cm is not None else 0.0,
        "weight_kg": profile.weight_kg if profile and profile.weight_kg is not None else 0.0,
        "diet_preference": profile.diet_preference if profile and profile.diet_preference else "Vegetarian"
    }


def get_or_create_profile(user_id: int):
    profile = PatientProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = PatientProfile(user_id=user_id)
        db.session.add(profile)
        db.session.commit()
    return profile


def _profile_pic_path(user_id: int, filename: str | None = None):
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg"}:
            ext = ".png"
        return os.path.join(PROFILE_PICS_DIR, f"user_{user_id}{ext}")

    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(PROFILE_PICS_DIR, f"user_{user_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(PROFILE_PICS_DIR, f"user_{user_id}.png")


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MedGraphX backend is running",
        "routes": [
            "/register",
            "/login",
            "/profile",
            "/admin/users",
            "/import_profiles"
        ]
    }), 200


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        if not data:
            return json_error("No input data received")

        full_name = str(data.get("full_name", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()
        role = str(data.get("role", "user")).strip().lower()

        if not full_name or not email or not password:
            return json_error("Full name, email, and password are required")

        if role not in ["user", "admin"]:
            role = "user"

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return json_error("Email already registered", 409)

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hashed_password,
            role=role
        )
        db.session.add(new_user)
        db.session.commit()

        empty_profile = PatientProfile(user_id=new_user.id)
        db.session.add(empty_profile)
        db.session.commit()

        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        db.session.rollback()
        return json_error(f"Registration error: {str(e)}", 500)


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()

        if not data:
            return json_error("No input data received")

        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()

        if not email or not password:
            return json_error("Email and password are required")

        user = User.query.filter_by(email=email).first()

        if not user:
            return json_error("Invalid email or password", 401)

        try:
            password_ok = bcrypt.check_password_hash(user.password_hash, password)
        except Exception:
            return json_error("Stored password hash is invalid. Please run the password fix script.", 500)

        if not password_ok:
            return json_error("Invalid email or password", 401)

        token = create_access_token(identity=str(user.id))
        profile = get_or_create_profile(user.id)

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role
            },
            "profile": serialize_profile(profile)
        }), 200

    except Exception as e:
        return json_error(f"Login error: {str(e)}", 500)


@app.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    try:
        user_id = int(get_jwt_identity())
        profile = get_or_create_profile(user_id)
        return jsonify(serialize_profile(profile)), 200

    except Exception as e:
        return json_error(f"Profile fetch error: {str(e)}", 500)


@app.route("/profile", methods=["POST"])
@jwt_required()
def save_profile():
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if not data:
            return json_error("No input data received")

        profile = get_or_create_profile(user_id)

        profile.age = int(data.get("age", 0)) if data.get("age") not in [None, ""] else 0
        profile.gender = str(data.get("gender", "Male")).strip()
        profile.diseases = str(data.get("diseases", "")).strip()
        profile.medications = str(data.get("medications", "")).strip()
        profile.allergies = str(data.get("allergies", "")).strip()
        profile.lifestyle = str(data.get("lifestyle", "")).strip()
        profile.height_cm = float(data.get("height_cm", 0.0)) if data.get("height_cm") not in [None, ""] else 0.0
        profile.weight_kg = float(data.get("weight_kg", 0.0)) if data.get("weight_kg") not in [None, ""] else 0.0
        profile.diet_preference = str(data.get("diet_preference", "Vegetarian")).strip()

        db.session.commit()

        return jsonify({
            "message": "Profile saved successfully",
            "profile": serialize_profile(profile)
        }), 200

    except Exception as e:
        db.session.rollback()
        return json_error(f"Profile save error: {str(e)}", 500)


@app.route("/admin/users", methods=["GET"])
@jwt_required()
def admin_users():
    try:
        user_id = int(get_jwt_identity())
        current_user = User.query.get(user_id)

        if not current_user or current_user.role != "admin":
            return json_error("Admin access required", 403)

        users = User.query.all()
        data = []

        for user in users:
            profile = PatientProfile.query.filter_by(user_id=user.id).first()
            data.append({
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
                "age": profile.age if profile and profile.age is not None else 0,
                "gender": profile.gender if profile and profile.gender else "",
                "diet_preference": profile.diet_preference if profile and profile.diet_preference else ""
            })

        return jsonify(data), 200

    except Exception as e:
        return json_error(f"Admin fetch error: {str(e)}", 500)


@app.route("/admin/stats", methods=["GET"])
@jwt_required()
def admin_stats():
    try:
        user_id = int(get_jwt_identity())
        current_user = User.query.get(user_id)

        if not current_user or current_user.role != "admin":
            return json_error("Admin access required", 403)

        users = User.query.all()
        profiles = PatientProfile.query.all()

        return jsonify({
            "total_users": len(users),
            "user_count": len([u for u in users if u.role == "user"]),
            "admin_count": len([u for u in users if u.role == "admin"]),
            "profiles_with_diseases": len([p for p in profiles if p.diseases]),
            "profiles_with_medications": len([p for p in profiles if p.medications])
        }), 200
    except Exception as e:
        return json_error(str(e), 500)

@app.route("/admin/system_metrics", methods=["POST"])
@jwt_required()
def system_metrics():
    try:
        user_id = int(get_jwt_identity())
        current_user = User.query.get(user_id)

        if current_user.role != "admin":
            return jsonify({"error": "Admin access required"}), 403

        data = request.get_json()

        if not data:
            return jsonify({"error": "No dataset provided"}), 400

        # ----------------------------
        # IMPORT YOUR MODULES
        # ----------------------------
        from module4_nlp_preprocessing import (
            run_nlp_pipeline,
            extract_entities_spacy,
            build_graph_dynamic
        )

        import numpy as np

        # ----------------------------
        # NLP PROCESSING
        # ----------------------------
        processed = run_nlp_pipeline(data)

        total_tokens = sum(len(p) for p in processed)
        avg_tokens = total_tokens / (len(processed) + 1)

        # NLP Accuracy heuristic
        nlp_accuracy = min(100, (avg_tokens * 5))

        # ----------------------------
        # ENTITY EXTRACTION
        # ----------------------------
        entities = extract_entities_spacy(data)

        unique_entities = list(set(entities))
        entity_count = len(unique_entities)

        entity_accuracy = min(100, entity_count * 2)

        # ----------------------------
        # SEMANTIC ANALYSIS
        # ----------------------------
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = [e[0] for e in unique_entities[:10]]

            if len(texts) > 1:
                embeddings = model.encode(texts)

                sim_scores = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        score = util.cos_sim(embeddings[i], embeddings[j]).item()
                        sim_scores.append(score)

                semantic_accuracy = float(np.mean(sim_scores)) * 100
            else:
                semantic_accuracy = 50.0

        except Exception:
            semantic_accuracy = 50.0

        # ----------------------------
        # KNOWLEDGE GRAPH QUALITY
        # ----------------------------
        G = build_graph_dynamic(unique_entities)

        nodes = len(G.nodes)
        edges = len(G.edges)

        if nodes > 0:
            density = edges / (nodes + 1)
            kg_accuracy = min(100, density * 50)
        else:
            kg_accuracy = 0

        return jsonify({
            "nlp_accuracy": round(nlp_accuracy, 2),
            "semantic_accuracy": round(semantic_accuracy, 2),
            "entity_extraction_accuracy": round(entity_accuracy, 2),
            "knowledge_graph_accuracy": round(kg_accuracy, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/user_details", methods=["GET"])
@jwt_required()
def admin_user_details():
    try:
        user_id = int(get_jwt_identity())
        current_user = User.query.get(user_id)

        if not current_user or current_user.role != "admin":
            return json_error("Admin access required", 403)

        users = User.query.all()
        data = []

        for user in users:
            profile = PatientProfile.query.filter_by(user_id=user.id).first()

            data.append({
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
                "age": profile.age if profile else 0,
                "gender": profile.gender if profile else "",
                "diseases": profile.diseases if profile else "",
                "medications": profile.medications if profile else "",
                "allergies": profile.allergies if profile else "",
                "height_cm": profile.height_cm if profile else 0,
                "weight_kg": profile.weight_kg if profile else 0,
                "diet_preference": profile.diet_preference if profile else ""
            })

        return jsonify(data), 200

    except Exception as e:
        return json_error(str(e), 500)


@app.route("/profile/pic", methods=["POST"])
@jwt_required()
def upload_profile_pic():
    try:
        user_id = int(get_jwt_identity())

        if "file" not in request.files:
            return json_error("No file uploaded")

        file = request.files["file"]
        if not file or not file.filename:
            return json_error("Invalid file")

        old_path = _profile_pic_path(user_id)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

        save_path = _profile_pic_path(user_id, file.filename)
        file.save(save_path)

        return jsonify({"message": "Profile picture uploaded successfully"}), 200
    except Exception as e:
        return json_error(f"Profile picture upload error: {str(e)}", 500)


@app.route("/profile/pic", methods=["GET"])
@jwt_required()
def get_profile_pic():
    try:
        user_id = int(get_jwt_identity())
        pic_path = _profile_pic_path(user_id)

        if not os.path.exists(pic_path):
            return "", 204

        mimetype, _ = mimetypes.guess_type(pic_path)
        return send_file(pic_path, mimetype=mimetype or "application/octet-stream")
    except Exception as e:
        return json_error(f"Profile picture fetch error: {str(e)}", 500)


@app.route("/import_profiles", methods=["POST"])
def import_profiles():
    try:
        if not os.path.exists(DATASET_PATH):
            return json_error(f"File not found: {DATASET_PATH}", 404)

        if DATASET_PATH.lower().endswith(".csv"):
            df = pd.read_csv(DATASET_PATH)
        else:
            df = pd.read_excel(DATASET_PATH, engine="openpyxl")

        required_columns = [
            "full_name", "email", "age", "gender", "diseases",
            "medications", "allergies", "lifestyle",
            "height_cm", "weight_kg", "diet_preference"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return json_error(f"Missing columns in dataset: {', '.join(missing_columns)}", 400)

        imported = 0
        skipped = 0

        for _, row in df.iterrows():
            email = str(row["email"]).strip().lower()

            if not email:
                skipped += 1
                continue

            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                skipped += 1
                continue

            password_hash = bcrypt.generate_password_hash("1234").decode("utf-8")

            user = User(
                full_name=str(row["full_name"]).strip(),
                email=email,
                password_hash=password_hash,
                role="user"
            )
            db.session.add(user)
            db.session.commit()

            profile = PatientProfile(
                user_id=user.id,
                age=int(row["age"]) if pd.notna(row["age"]) else 0,
                gender=str(row["gender"]).strip() if pd.notna(row["gender"]) else "Male",
                diseases=str(row["diseases"]).strip() if pd.notna(row["diseases"]) else "",
                medications=str(row["medications"]).strip() if pd.notna(row["medications"]) else "",
                allergies=str(row["allergies"]).strip() if pd.notna(row["allergies"]) else "",
                lifestyle=str(row["lifestyle"]).strip() if pd.notna(row["lifestyle"]) else "",
                height_cm=float(row["height_cm"]) if pd.notna(row["height_cm"]) else 0.0,
                weight_kg=float(row["weight_kg"]) if pd.notna(row["weight_kg"]) else 0.0,
                diet_preference=str(row["diet_preference"]).strip() if pd.notna(row["diet_preference"]) else "Vegetarian"
            )

            db.session.add(profile)
            db.session.commit()
            imported += 1

        return jsonify({
            "message": "Dataset imported successfully",
            "imported": imported,
            "skipped": skipped
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return json_error(f"Import error: {str(e)}", 500)

@app.route("/admin/fix_passwords", methods=["POST"])
def admin_fix_passwords():
    try:
        users = User.query.all()
        updated = 0

        for user in users:
            user.password_hash = bcrypt.generate_password_hash("1234").decode("utf-8")
            updated += 1

        db.session.commit()

        return jsonify({
            "message": "Passwords fixed successfully",
            "updated_users": updated
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collect_data", methods=["POST"])
def collect_data_api():
    try:
        data = request.get_json()

        source = data.get("source")
        query = data.get("query")

        if not source or not query:
            return jsonify({"error": "Source and query required"}), 400

        result = collect_data(source, query)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/collect_data", methods=["POST"])
@app.route("/feedback", methods=["POST"])
@jwt_required()
def submit_feedback():
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        feedback = Feedback(
            user_id=user_id,
            risk_analysis=int(data.get("risk_analysis", 0)),
            nutrient_analysis=int(data.get("nutrient_analysis", 0)),
            knowledge_mapping=int(data.get("knowledge_mapping", 0)),
            data_extraction=int(data.get("data_extraction", 0)),
            meal_planning=int(data.get("meal_planning", 0)),
            comments=data.get("comments", "")
        )

        db.session.add(feedback)
        db.session.commit()

        return jsonify({"message": "Feedback submitted"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/feedback", methods=["GET"])
@jwt_required()
def get_feedback():
    try:
        user_id = int(get_jwt_identity())
        current_user = User.query.get(user_id)

        if current_user.role != "admin":
            return jsonify({"error": "Admin access required"}), 403

        feedbacks = Feedback.query.all()

        data = []
        for f in feedbacks:
            user = User.query.get(f.user_id)

            data.append({
                "user": user.full_name if user else "Unknown",
                "risk_analysis": f.risk_analysis,
                "nutrient_analysis": f.nutrient_analysis,
                "knowledge_mapping": f.knowledge_mapping,
                "data_extraction": f.data_extraction,
                "meal_planning": f.meal_planning,
                "comments": f.comments,
                "created_at": str(f.created_at)
            })

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=False, port=5000)
