from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('lecturer_matcher_rf.joblib')

# Load the label encoders and scaler from data cleaning
# Note: You'll need to save these during training
categorical_columns = ['program', 'level', 'time_pref', 'subject']
numerical_columns = ['hourly_pay', 'student_count', 'credits', 'institute_rating']

# Initialize dictionaries to store encoders
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = joblib.load(f'encoders/{column}_encoder.joblib')

# Load the scaler
scaler = joblib.load('encoders/numerical_scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Create DataFrame from input data
        input_data = pd.DataFrame([data])

        # Encode categorical variables
        for column in categorical_columns:
            if column in input_data.columns:
                try:
                    input_data[column] = label_encoders[column].transform(input_data[column])
                except ValueError:
                    return jsonify({
                        'error': f'Invalid value for {column}. Valid values are: {list(label_encoders[column].classes_)}'
                    }), 400

        # Scale numerical variables
        if all(col in input_data.columns for col in numerical_columns):
            input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
        else:
            missing_cols = [col for col in numerical_columns if col not in input_data.columns]
            return jsonify({
                'error': f'Missing numerical columns: {missing_cols}'
            }), 400

        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Get top 3 recommendations with probabilities
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            {
                'lecturer_id': int(idx),
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]

        return jsonify({
            'predicted_lecturer_id': int(prediction),
            'top_3_recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True) 