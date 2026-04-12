import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoders at startup
print("🔄 Chargement du modèle...")
model = joblib.load("models/xgboost_churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")
print("✅ Modèle chargé avec succès!")

# List of categorical columns expected by the model
categorical_cols = ['SpendingCategory', 'PreferredTimeOfDay', 'WeekendPreference', 
                    'BasketSizeCategory', 'ProductDiversity', 'Country']

@app.route('/')
def home():
    """Home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction from form data"""
    
    # Get form data
    input_data = {}
    for col in categorical_cols:
        input_data[col] = [request.form.get(col)]
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Encode categorical values using saved encoders
    for col in categorical_cols:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Interpret result
    if prediction == 1:
        result = f"⚠️ Client à RISQUE de churn (probabilité: {probability:.1%})"
    else:
        result = f"✅ Client FIDÈLE (probabilité de churn: {probability:.1%})"
    
    return render_template('index.html', prediction=result, probability=f"{probability:.1%}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    data = request.get_json()
    
    input_df = pd.DataFrame([data])
    
    for col in categorical_cols:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return jsonify({
        'churn_prediction': int(prediction),
        'churn_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)