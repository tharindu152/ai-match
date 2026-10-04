"""HTTP API for lecturer matching."""

import logging
import math
import os
import threading
from functools import wraps
from pathlib import Path

import joblib
import jwt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from werkzeug.exceptions import HTTPException


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR)).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = BASE_DIR / MODEL_DIR
MODEL_DIR = MODEL_DIR.resolve()
MODEL_PATH = MODEL_DIR / "lecturer_matcher_rf.joblib"
ENCODER_DIR = MODEL_DIR / "encoders"
CATEGORICAL_COLUMNS = [
    "program",
    "level",
    "time_pref",
    "subject",
    "division",
    "status",
    "language",
]
NUMERICAL_COLUMNS = [
    "hourly_pay",
    "student_count",
    "credits",
    "institute_rating",
    "duration",
]
FEATURE_COLUMNS = [
    "program",
    "hourly_pay",
    "level",
    "time_pref",
    "student_count",
    "subject",
    "credits",
    "institute_rating",
    "duration",
    "division",
    "status",
    "language",
]
PREDICTION_FIELDS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
TRAINING_FIELDS = PREDICTION_FIELDS + ["lecturer_id"]
KEYCLOAK_ISSUER_URI = os.environ.get(
    "KEYCLOAK_ISSUER_URI", "http://localhost:7080/realms/lecturelink"
)
KEYCLOAK_JWK_SET_URI = os.environ.get(
    "KEYCLOAK_JWK_SET_URI",
    "http://localhost:7080/realms/lecturelink/protocol/openid-connect/certs",
)

app = Flask(__name__)
logger = logging.getLogger(__name__)
state_lock = threading.RLock()
jwks_client = jwt.PyJWKClient(KEYCLOAK_JWK_SET_URI, cache_keys=True)


class RequestValidationError(ValueError):
    """An invalid request that should be returned to the caller as a 400."""


def _load_artifacts():
    required_files = [MODEL_PATH, ENCODER_DIR / "numerical_scaler.joblib"]
    required_files.extend(
        ENCODER_DIR / f"{column}_encoder.joblib"
        for column in CATEGORICAL_COLUMNS
    )
    required_files.append(ENCODER_DIR / "lecturer_id_encoder.joblib")
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Required model artifacts are missing: " + ", ".join(missing_files)
        )

    model = joblib.load(MODEL_PATH)
    encoders = {
        column: joblib.load(ENCODER_DIR / f"{column}_encoder.joblib")
        for column in CATEGORICAL_COLUMNS
    }
    lecturer_id_encoder = joblib.load(ENCODER_DIR / "lecturer_id_encoder.joblib")
    scaler = joblib.load(ENCODER_DIR / "numerical_scaler.joblib")

    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        raise TypeError("The model must support predict and predict_proba")
    model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    if set(model_features) != set(FEATURE_COLUMNS):
        raise ValueError(
            "The model feature columns do not match the service request schema"
        )
    if getattr(scaler, "n_features_in_", None) != len(NUMERICAL_COLUMNS):
        raise ValueError("The numerical scaler does not match the service schema")
    if not hasattr(lecturer_id_encoder, "classes_"):
        raise TypeError("The lecturer ID encoder is invalid")

    classes = np.asarray(getattr(model, "classes_", []))
    uses_encoded_ids = (
        len(classes) > 0
        and all(
            isinstance(value, (int, np.integer))
            and 0 <= int(value) < len(lecturer_id_encoder.classes_)
            for value in classes
        )
        and any(int(value) == 0 for value in classes)
    )
    return {
        "model": model,
        "encoders": encoders,
        "lecturer_id_encoder": lecturer_id_encoder,
        "scaler": scaler,
        "encoded_lecturer_ids": uses_encoded_ids,
    }


state = _load_artifacts()


def _error(message, status):
    return jsonify({"error": message}), status


def require_roles(*expected_roles):
    def decorate(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            authorization = request.headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                return _error("A bearer token is required", 401)

            token = authorization[len("Bearer "):].strip()
            if not token:
                return _error("A bearer token is required", 401)

            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token).key
                claims = jwt.decode(
                    token,
                    signing_key,
                    algorithms=["RS256"],
                    issuer=KEYCLOAK_ISSUER_URI,
                    options={"verify_aud": False},
                )
            except jwt.PyJWKClientConnectionError:
                logger.exception("Unable to reach the Keycloak JWKS endpoint")
                return _error("Identity provider is unavailable", 503)
            except jwt.PyJWTError:
                return _error("Invalid or expired bearer token", 401)

            realm_access = claims.get("realm_access")
            roles = realm_access.get("roles") if isinstance(realm_access, dict) else None
            if not isinstance(roles, list) or not any(
                role in expected_roles for role in roles if isinstance(role, str)
            ):
                return _error("The token does not have the required role", 403)

            return view(*args, **kwargs)

        return wrapped

    return decorate


def _read_json():
    if not request.is_json:
        raise RequestValidationError("Request must be JSON")
    payload = request.get_json(silent=True)
    if payload is None:
        raise RequestValidationError("Request body must contain valid JSON")
    return payload


def _validate_fields(data, expected_fields):
    if not isinstance(data, dict):
        raise RequestValidationError("Request body must be a JSON object")
    missing = [field for field in expected_fields if field not in data]
    if missing:
        raise RequestValidationError(f"Missing required fields: {missing}")
    unexpected = sorted(set(data) - set(expected_fields))
    if unexpected:
        raise RequestValidationError(f"Unexpected fields: {unexpected}")


def _normalise_category(value, column, encoder):
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{column} must be a non-empty string")
    normalised = value.strip().casefold()
    known_values = [str(category) for category in encoder.classes_]
    lookup = {category.casefold(): category for category in known_values}
    if normalised not in lookup:
        raise RequestValidationError(
            f"Invalid value for {column}. Valid values are: {known_values}"
        )
    return lookup[normalised]


def _validate_number(value, column):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestValidationError(f"{column} must be a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise RequestValidationError(f"{column} must be finite") from error
    if not math.isfinite(number):
        raise RequestValidationError(f"{column} must be finite")
    return number


def _prepare_prediction(data, current_state):
    _validate_fields(data, PREDICTION_FIELDS)
    row = {}
    for column in CATEGORICAL_COLUMNS:
        category = _normalise_category(
            data[column], column, current_state["encoders"][column]
        )
        row[column] = int(
            current_state["encoders"][column].transform([category])[0]
        )
    for column in NUMERICAL_COLUMNS:
        row[column] = _validate_number(data[column], column)

    frame = pd.DataFrame([row])
    frame.loc[:, NUMERICAL_COLUMNS] = current_state["scaler"].transform(
        frame[NUMERICAL_COLUMNS]
    )
    return frame[FEATURE_COLUMNS]


def _json_lecturer_id(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value)
    if text.isdecimal():
        return int(text)
    return text


def _decode_lecturer_id(value, current_state):
    if current_state["encoded_lecturer_ids"]:
        return _json_lecturer_id(
            current_state["lecturer_id_encoder"].inverse_transform(
                [int(value)]
            )[0]
        )
    return _json_lecturer_id(value)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
@require_roles("LECTURER")
def predict():
    data = _read_json()
    with state_lock:
        current_state = state
        features = _prepare_prediction(data, current_state)
        prediction = current_state["model"].predict(features)[0]
        probabilities = current_state["model"].predict_proba(features)[0]
        classes = current_state["model"].classes_

        top_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            {
                "lecturer_id": _decode_lecturer_id(classes[index], current_state),
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ]
        predicted_id = _decode_lecturer_id(prediction, current_state)

    return jsonify(
        {
            "predicted_lecturer_id": predicted_id,
            "top_3_recommendations": recommendations,
        }
    )


def _prepare_training_data(records):
    if not isinstance(records, list) or not records:
        raise RequestValidationError("Retraining data must be a non-empty JSON array")
    for index, record in enumerate(records):
        _validate_fields(record, TRAINING_FIELDS)
        for column in CATEGORICAL_COLUMNS:
            value = record[column]
            if not isinstance(value, str) or not value.strip():
                raise RequestValidationError(
                    f"{column} in row {index} must be a non-empty string"
                )
    frame = pd.DataFrame(records, columns=TRAINING_FIELDS)
    if len(frame) < 2:
        raise RequestValidationError("At least two training rows are required")

    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].map(
            lambda value: value.strip().casefold() if isinstance(value, str) else value
        )
        if frame[column].isna().any():
            raise RequestValidationError(f"{column} cannot contain null values")
    for column in NUMERICAL_COLUMNS:
        frame[column] = [
            _validate_number(value, column) for value in frame[column]
        ]
    for value in frame["lecturer_id"]:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise RequestValidationError("lecturer_id must be a string or number")
        if isinstance(value, float):
            number = _validate_number(value, "lecturer_id")
            if not number.is_integer():
                raise RequestValidationError("Numeric lecturer_id values must be integers")
        if isinstance(value, str) and not value.strip():
            raise RequestValidationError("lecturer_id must not be empty")
    frame["lecturer_id"] = frame["lecturer_id"].map(
        lambda value: (
            value.strip()
            if isinstance(value, str)
            else str(int(value))
            if isinstance(value, float)
            else str(value)
        )
    )
    if frame["lecturer_id"].nunique() < 2:
        raise RequestValidationError("At least two distinct lecturer IDs are required")
    return frame


def _train_model(frame):
    encoders = {
        column: LabelEncoder().fit(frame[column])
        for column in CATEGORICAL_COLUMNS
    }
    features = frame[FEATURE_COLUMNS].copy()
    for column, encoder in encoders.items():
        features[column] = encoder.transform(features[column])

    scaler = StandardScaler()
    features.loc[:, NUMERICAL_COLUMNS] = scaler.fit_transform(
        features[NUMERICAL_COLUMNS]
    )
    lecturer_id_encoder = LabelEncoder().fit(frame["lecturer_id"])
    targets = lecturer_id_encoder.transform(frame["lecturer_id"])
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(features[FEATURE_COLUMNS], targets)
    return {
        "model": model,
        "encoders": encoders,
        "lecturer_id_encoder": lecturer_id_encoder,
        "scaler": scaler,
        "encoded_lecturer_ids": True,
    }, features[FEATURE_COLUMNS], targets


def _save_artifacts(new_state):
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_state["model"], MODEL_PATH)
    for column, encoder in new_state["encoders"].items():
        joblib.dump(encoder, ENCODER_DIR / f"{column}_encoder.joblib")
    joblib.dump(
        new_state["lecturer_id_encoder"],
        ENCODER_DIR / "lecturer_id_encoder.joblib",
    )
    joblib.dump(new_state["scaler"], ENCODER_DIR / "numerical_scaler.joblib")


@app.route("/retrain", methods=["POST"])
@require_roles("INSTITUTE")
def retrain():
    records = _read_json()
    frame = _prepare_training_data(records)
    with state_lock:
        new_state, features, targets = _train_model(frame)
        accuracy = float(new_state["model"].score(features, targets))
        _save_artifacts(new_state)
        global state
        state = new_state

    return jsonify(
        {
            "message": "Model trained from scratch successfully",
            "training_accuracy": accuracy,
            "n_samples": len(frame),
            "n_features": features.shape[1],
        }
    )


@app.errorhandler(RequestValidationError)
def handle_validation_error(error):
    return _error(str(error), 400)


@app.errorhandler(HTTPException)
def handle_http_error(error):
    return _error(error.description, error.code or 500)


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unhandled request error", exc_info=error)
    return _error("An internal server error occurred", 500)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
