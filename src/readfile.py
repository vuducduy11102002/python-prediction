import joblib

# Load the scaler
scaler = joblib.load("models/diabetes_model.pkl")

# Check the type
print(type(scaler))
