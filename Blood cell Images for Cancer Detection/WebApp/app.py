"""
CellVision - Blood Cell Classification Web Application

Model:
    CNN feature extractor + Random Forest classifier

Classes:
    basophil, erythroblast, monocyte, myeloblast, seg_neutrophil
"""

import logging
import os
import uuid
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil",
]

IMAGE_SIZE = 224
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CellVision")

feature_model = None
classifier = None
model_load_error = None


def load_artifacts():
    """Load the CNN feature extractor and trained Random Forest."""
    global feature_model, classifier, model_load_error

    cnn_path = MODEL_DIR / "cnn_ml_best.keras"
    rf_path = MODEL_DIR / "cnn_ml_random_forest.joblib"

    missing = [str(p) for p in (cnn_path, rf_path) if not p.is_file()]

    if missing:
        model_load_error = "Missing model file(s): " + ", ".join(missing)
        logger.error(model_load_error)
        return

    try:
        cnn = tf.keras.models.load_model(
            str(cnn_path),
            compile=False,
        )

        try:
            feature_layer = cnn.get_layer("cnn_features")
        except ValueError as exc:
            raise ValueError(
                "The CNN must contain a layer named 'cnn_features'."
            ) from exc

        new_feature_model = tf.keras.Model(
            inputs=cnn.inputs,
            outputs=feature_layer.output,
            name="CNN_Feature_Extractor",
        )

        new_classifier = joblib.load(rf_path)

        expected = getattr(new_classifier, "n_features_in_", None)
        actual_shape = new_feature_model.output_shape
        actual = actual_shape[-1] if actual_shape else None

        if (
            expected is not None
            and actual is not None
            and int(expected) != int(actual)
        ):
            raise ValueError(
                f"Feature mismatch: CNN outputs {actual} features, "
                f"but Random Forest expects {expected}."
            )

        if not hasattr(new_classifier, "predict_proba"):
            raise ValueError(
                "Random Forest must support predict_proba()."
            )

        feature_model = new_feature_model
        classifier = new_classifier
        model_load_error = None

        logger.info("CNN feature extractor loaded: %s", cnn_path)
        logger.info("Random Forest loaded: %s", rf_path)

    except Exception as exc:
        feature_model = None
        classifier = None
        model_load_error = f"Model loading failed: {exc}"
        logger.exception(model_load_error)


load_artifacts()


def allowed_file(filename):
    """Check whether the filename has a supported image extension."""
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def wants_json_response():
    """Identify JavaScript and API requests."""
    return (
        "application/json" in request.headers.get("Accept", "").lower()
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.is_json
    )


def render_dashboard(error=None, status_code=200):
    """Render the main dashboard with safe initial values."""
    return render_template(
        "index.html",
        model_ready=(feature_model is not None and classifier is not None),
        model_error=model_load_error,
        error=error,
    ), status_code


def error_response(message, status_code=400):
    """Return an error in JSON or HTML according to the request."""
    if wants_json_response():
        return jsonify({
            "status": "error",
            "error": message,
        }), status_code

    return render_dashboard(message, status_code)


def preprocess_image(image_path):
    """
    RGB, resize to 224 x 224, convert to float32, divide by 255.
    """
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)

    array /= 255.0
    return np.expand_dims(array, axis=0)


def normalize_label(label):
    """Standardize string labels for comparison."""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def get_class_probabilities(features):
    """Return probabilities aligned with CLASS_NAMES."""
    raw = np.asarray(classifier.predict_proba(features), dtype=np.float64)

    if raw.ndim != 2 or raw.shape[0] != 1:
        raise ValueError(f"Unexpected probability shape: {raw.shape}")

    raw = raw[0]
    model_classes = getattr(classifier, "classes_", None)

    if model_classes is None or len(model_classes) != len(raw):
        raise ValueError("Cannot verify classifier classes and probabilities.")

    probabilities = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    seen = set()

    for model_label, probability in zip(model_classes, raw):
        if isinstance(model_label, (int, np.integer)):
            index = int(model_label)

            if not 0 <= index < len(CLASS_NAMES):
                raise ValueError(f"Unexpected encoded label: {model_label}")
        else:
            normalized = normalize_label(model_label)
            matches = [
                i for i, name in enumerate(CLASS_NAMES)
                if normalize_label(name) == normalized
            ]

            if len(matches) != 1:
                raise ValueError(f"Unknown classifier label: {model_label}")

            index = matches[0]

        if index in seen:
            raise ValueError("Duplicate class labels returned by the model.")

        seen.add(index)
        probabilities[index] = float(probability)

    if len(seen) != len(CLASS_NAMES):
        raise ValueError(
            "The classifier must contain all five configured classes. "
            "Check the label encoding used during training."
        )

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("The classifier returned invalid probabilities.")

    return probabilities


@app.route("/", methods=["GET"])
def index():
    """Display the upload dashboard."""
    return render_dashboard()


@app.route("/predict", methods=["POST"])
def predict():
    """Analyze an uploaded image and return its results."""
    try:
        image_file = request.files.get("file")

        if image_file is None:
            image_file = request.files.get("image")

        if image_file is None or not image_file.filename:
            return error_response(
                "Please select a blood cell image first.",
                400,
            )

        original_filename = secure_filename(image_file.filename)

        if not allowed_file(original_filename):
            return error_response(
                "Unsupported image type. Use JPG, JPEG, PNG, BMP, TIFF, or WEBP.",
                400,
            )

        if feature_model is None or classifier is None:
            return error_response(
                model_load_error or "The trained model is unavailable.",
                503,
            )

        try:
            image_file.stream.seek(0)
            with Image.open(image_file.stream) as check_image:
                check_image.verify()
            image_file.stream.seek(0)
        except (UnidentifiedImageError, OSError, ValueError):
            return error_response(
                "The selected file is not a valid image.",
                400,
            )

        extension = Path(original_filename).suffix.lower() or ".jpg"
        saved_name = f"cell_{uuid.uuid4().hex}{extension}"
        saved_path = UPLOAD_DIR / saved_name
        image_file.save(str(saved_path))

        image_array = preprocess_image(saved_path)
        features = np.asarray(
            feature_model.predict(image_array, verbose=0)
        )

        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        probabilities = get_class_probabilities(features)

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        prediction_display = predicted_class.replace("_", " ").title()
        confidence = round(float(probabilities[predicted_index] * 100), 2)

        order = np.argsort(probabilities)[::-1]
        top_predictions = [
            {
                "label": CLASS_NAMES[int(i)].replace("_", " ").title(),
                "probability": round(float(probabilities[i] * 100), 2),
            }
            for i in order
        ]

        image_url = f"/static/uploads/{saved_name}"

        logger.info(
            "Prediction completed: %s (%.2f%%)",
            predicted_class,
            confidence,
        )

        return jsonify({
            "status": "success",
            "prediction": prediction_display,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "image_url": image_url,
            "filename": original_filename,
            "top_predictions": top_predictions,
            "model": "CNN Feature Extractor + Random Forest",
        })

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Image processing failed: %s", exc)
        return error_response(
            f"The image could not be processed: {exc}",
            400,
        )

    except Exception:
        logger.exception("Prediction failed")
        return error_response(
            "Prediction failed. Check the application terminal for details.",
            500,
        )


@app.route("/api/status", methods=["GET"])
@app.route("/health", methods=["GET"])
def api_status():
    """Report whether the models are ready."""
    ready = feature_model is not None and classifier is not None

    return jsonify({
        "status": "ok" if ready else "model_unavailable",
        "model_ready": ready,
        "model_error": model_load_error,
        "model": "CNN Feature Extractor + Random Forest",
        "classes": CLASS_NAMES,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": "RGB, resize to 224x224, divide by 255",
    }), (200 if ready else 503)


@app.errorhandler(413)
def file_too_large(error):
    """Handle images exceeding the upload size limit."""
    return error_response(
        "The image exceeds the 10 MB upload limit.",
        413,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )