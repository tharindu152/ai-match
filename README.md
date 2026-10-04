# AI Match Service

This Python service recommends lecturers for a course using the supplied
Random Forest model. Artifacts are loaded from `MODEL_DIR`, which defaults to
the directory containing `app.py`; startup does not depend on the current
working directory. Keep `lecturer_matcher_rf.joblib` and the `encoders/`
directory together under the configured model directory.

## Run locally

From this directory, install the pinned model/runtime dependencies and start
the production WSGI server:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The service listens on `http://localhost:5000`; it does not use Flask's debug
server. Use `python3 -m venv .venv`, `source .venv/bin/activate` on Linux/macOS.
Scikit-learn is pinned to the version used to serialize the supplied model.
To store artifacts elsewhere, set `MODEL_DIR` before starting the service. For
example, on Linux/macOS use `MODEL_DIR=/path/to/model python app.py`; in
PowerShell use `$env:MODEL_DIR = 'C:\path\to\model'` before `python app.py`.
Retraining writes the replacement model and all encoders to that directory.

## Run with Docker

Build and start the service from this directory:

```sh
docker build -t ai-match-service .
docker run --rm -p 5000:5000 ai-match-service
```

The image contains only the API, dependencies, trained model, and encoders;
training notebooks and source data are not included. It sets `MODEL_DIR=/model`
and stores the supplied artifacts at `/model`, separate from the application
code in `/app`. Mount a volume there to retain trained artifacts without
hiding application files:

```sh
docker run --rm -p 5000:5000 -v ai-match-model:/model ai-match-service
```

## API

All error responses use `{"error": "..."}`. Invalid or incomplete input returns
HTTP 400; unexpected server errors return HTTP 500. Categorical values are
case-insensitive and surrounding whitespace is ignored. They must otherwise
match the fitted encoders' known values. Every feature below is required.
Except for `GET /health`, requests require a Keycloak bearer token. Set
`KEYCLOAK_ISSUER_URI` to the exact issuer in the token and
`KEYCLOAK_JWK_SET_URI` to the Keycloak realm's signing-key endpoint. Prediction
requires the `LECTURER` realm role; retraining requires `INSTITUTE`.

### `GET /health`

Returns `{"status": "healthy"}` when the model artifacts have loaded and the
service is ready.

### `POST /predict`

Requires `Authorization: Bearer <LECTURER access token>`.

Send a JSON object with these fields:

| Field | Type | Example |
| --- | --- | --- |
| `program` | string | `"Bachelor of Commerce"` |
| `level` | string | `"Bachelors"` |
| `time_pref` | string | `"Weekend"` |
| `subject` | string | `"Strategic Management"` |
| `division` | string | `"Kotte"` |
| `status` | string | `"ACTIVE"` |
| `language` | string | `"English"` |
| `hourly_pay` | number | `3300` |
| `student_count` | number | `50` |
| `credits` | number | `3` |
| `institute_rating` | number | `4.8` |
| `duration` | number | `1095` |

Example:

```sh
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Bachelor of Commerce",
    "level": "Bachelors",
    "time_pref": "Weekend",
    "subject": "Strategic Management",
    "division": "Kotte",
    "status": "ACTIVE",
    "language": "English",
    "hourly_pay": 3300,
    "student_count": 50,
    "credits": 3,
    "institute_rating": 4.8,
    "duration": 1095
  }'
```

Successful predictions return the predicted ID and up to three recommendations:

```json
{
  "predicted_lecturer_id": 5,
  "top_3_recommendations": [
    {"lecturer_id": 5, "probability": 0.85},
    {"lecturer_id": 2, "probability": 0.1},
    {"lecturer_id": 7, "probability": 0.05}
  ]
}
```

### `POST /retrain`

Requires `Authorization: Bearer <INSTITUTE access token>`.

Accepts a non-empty JSON array of training records. Each record must contain
all prediction fields plus a `lecturer_id`. Categorical values may include new
categories; retraining fits fresh encoders and scaler. At least two records and
two distinct lecturer IDs are required. A successful request replaces the
in-memory model and writes the updated model and encoders under `MODEL_DIR`.
Do not send untrusted datasets to this endpoint.

```sh
curl -X POST http://localhost:5000/retrain \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[{"program":"Bachelor of Commerce","level":"Bachelors","time_pref":"Weekend","subject":"Strategic Management","division":"Kotte","status":"ACTIVE","language":"English","hourly_pay":3300,"student_count":50,"credits":3,"institute_rating":4.8,"duration":1095,"lecturer_id":5},{"program":"Bachelor of Commerce","level":"Bachelors","time_pref":"Weekday","subject":"Business Statistics","division":"Kotte","status":"ACTIVE","language":"English","hourly_pay":3300,"student_count":35,"credits":3,"institute_rating":4.8,"duration":1095,"lecturer_id":2}]'
```
