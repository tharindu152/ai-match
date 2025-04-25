from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('lecturer_matcher_rf.joblib')

# Load the label encoders and scaler from data cleaning
# Note: You'll need to save these during training
categorical_columns = ['program', 'level', 'time_pref', 'subject', 'division', 'status', 'language']
numerical_columns = ['hourly_pay', 'student_count', 'credits', 'institute_rating', 'duration']

# Initialize dictionaries to store encoders
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = joblib.load(f'encoders/{column}_encoder.joblib')

lecturer_id_encoder = joblib.load(f'encoders/lecturer_id_encoder.joblib')

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
        # Decode the predicted lecturer id
        predicted_lecturer_id = lecturer_id_encoder.inverse_transform([prediction])[0]

        # Decode the top 3 recommendations
        recommendations = [
            {
                'lecturer_id': int(lecturer_id_encoder.inverse_transform([idx])[0]),
                'probability': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]


        return jsonify({
            'predicted_lecturer_id': predicted_lecturer_id,
            'top_3_recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Check if JSON data is provided
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        # Parse JSON data directly
        data = request.get_json()
        new_data = pd.DataFrame(data)

        # Validate required columns
        required_columns = categorical_columns + numerical_columns + ['lecturer_id']
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400

        # Train model from scratch with new data
        new_model, X, y = train_model(new_data)

        # Calculate training metrics
        train_score = new_model.score(X, y)

        # Update global model
        global model
        model = new_model

        return jsonify({
            'message': 'Model trained from scratch successfully',
            'training_accuracy': float(train_score),
            'n_samples': len(new_data),
            'n_features': X.shape[1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def train_model(df):
    """
    Train the model with new data
    """
    # Prepare features and target
    X = df.drop('lecturer_id', axis=1)
    y = df['lecturer_id']

    # Encode categorical variables
    for column in categorical_columns:
        if column in X.columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
            joblib.dump(label_encoders[column], f'encoders/{column}_encoder.joblib')

    # Scale numerical variables
    scaler_new = StandardScaler()
    X[numerical_columns] = scaler_new.fit_transform(X[numerical_columns])
    joblib.dump(scaler_new, 'encoders/numerical_scaler.joblib')

    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X, y)

    # Save model
    joblib.dump(rf_model, 'lecturer_matcher_rf.joblib')

    return rf_model, X, y

if __name__ == '__main__':
    app.run(debug=True) 