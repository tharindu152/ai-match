# Lecturer Matcher API

This API provides lecturer recommendations for courses based on a trained Random Forest model.

## Setup

1. Install required packages:
```bash
pip install flask pandas numpy scikit-learn joblib
```

2. Run the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check

Check if the API is running.

```bash
GET /health
```

Sample request:
```bash
curl http://localhost:5000/health
```

Sample response:
```json
{
    "status": "healthy"
}
```

### Predict Lecturer

Get lecturer recommendations for a course.

```bash
POST /predict
```

#### Request Format

Content-Type: `application/json`

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| program | string | Name of the program | "Bachelor of Commerce" |
| hourly_pay | number | Hourly payment for lecturer | 3000 |
| level | string | Academic level | "Bachelors" |
| time_pref | string | Time preference | "Weekend" |
| student_count | number | Number of students | 50 |
| subject | string | Subject name | "Business Analytics" |
| credits | number | Number of credits | 3 |
| institute_rating | number | Institute rating | 4.8 |

#### Sample Request

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "program": "Bachelor of Commerce",
    "hourly_pay": 3000,
    "level": "Bachelors",
    "time_pref": "Weekend",
    "student_count": 50,
    "subject": "Business Analytics",
    "credits": 3,
    "institute_rating": 4.8
}'
```

#### Sample Response

```json
{
    "predicted_lecturer_id": 5,
    "top_3_recommendations": [
        {
            "lecturer_id": 5,
            "probability": 0.85
        },
        {
            "lecturer_id": 2,
            "probability": 0.10
        },
        {
            "lecturer_id": 7,
            "probability": 0.05
        }
    ]
}
```

#### Error Response

If there's an error, the API will return a 400 or 500 status code with an error message:

```json
{
    "error": "Invalid value for level. Valid values are: ['Bachelors', 'Masters', 'Doctorate', 'PostGraduate', 'HND', 'HNC']"
}
```

## Notes

1. All fields in the request are required
2. Values for categorical fields (program, level, time_pref, subject) must match the trained model's categories
3. Numerical values should be within reasonable ranges
4. The API returns the top 3 recommended lecturers with their probability scores

## Error Handling

The API handles several types of errors:
- Missing required fields
- Invalid values for categorical fields
- Server-side processing errors

Each error response includes a descriptive message to help identify the issue.