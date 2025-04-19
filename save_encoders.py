import joblib

# After training your model in training.ipynb, add these lines:
def save_encoders(label_encoders, scaler):
    # Save label encoders
    for column, encoder in label_encoders.items():
        joblib.dump(encoder, f'encoders/{column}_encoder.joblib')
    
    # Save scaler
    joblib.dump(scaler, 'encoders/numerical_scaler.joblib') 