import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load pickle model relative to the current file location
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "logistic.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Level Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #f4f7f6; color: #333; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
        .container { background: #ffffff; padding: 30px 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); max-width: 650px; width: 100%; }
        h2 { margin-bottom: 20px; color: #2c3e50; text-align: center; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 0.85rem; font-weight: 600; margin-bottom: 5px; color: #555; }
        input, select { padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; outline: none; transition: border 0.3s; }
        input:focus, select:focus { border-color: #3498db; }
        .full-width { grid-column: span 2; }
        button { background-color: #3498db; color: white; border: none; padding: 12px; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; margin-top: 15px; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        .result { margin-top: 20px; padding: 15px; border-radius: 6px; text-align: center; font-weight: bold; font-size: 1.1rem; }
        .success { background-color: #e8f8f5; color: #27ae60; border: 1px solid #27ae60; }
        .error { background-color: #fde8e8; color: #e74c3c; border: 1px solid #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Level Prediction</h2>
        <form method="POST" action="/predict">
            <div class="form-grid">
                <!-- Numerical Inputs -->
                <div class="form-group">
                    <label>Attendance (%)</label>
                    <input type="number" step="any" name="attendance" required>
                </div>
                <div class="form-group">
                    <label>Study Hours / Week</label>
                    <input type="number" step="any" name="study_hours" required>
                </div>
                <div class="form-group">
                    <label>Past Failures</label>
                    <input type="number" name="past_failures" required>
                </div>
                <div class="form-group">
                    <label>Assignments Completed (%)</label>
                    <input type="number" step="any" name="assignments_completed_pct" required>
                </div>
                <div class="form-group">
                    <label>Previous Grade</label>
                    <input type="number" step="any" name="previous_grade" required>
                </div>
                <div class="form-group">
                    <label>Final Score</label>
                    <input type="number" step="any" name="final_score" required>
                </div>

                <!-- Categorical Inputs -->
                <div class="form-group">
                    <label>Parental Education</label>
                    <select name="parental_education" required>
                        <option value="0">High School</option>
                        <option value="1">Associate Degree</option>
                        <option value="2">Bachelor's Degree</option>
                        <option value="3">Master's / Higher</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Family Income Level</label>
                    <select name="family_income" required>
                        <option value="0">Low</option>
                        <option value="1">Medium</option>
                        <option value="2">High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Extracurricular Activities</label>
                    <select name="extracurricular" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Internet Access</label>
                    <select name="internet_access" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
            </div>
            <button type="submit" class="full-width">Predict Risk Level</button>
        </form>

        {% if prediction %}
            <div class="result success">Prediction: {{ prediction }}</div>
        {% endif %}
        {% if error %}
            <div class="result error">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model file not loaded successfully.")

    try:
        # Collect and order features exactly as trained
        feature_order = [
            'attendance', 'study_hours', 'past_failures', 'assignments_completed_pct',
            'parental_education', 'family_income', 'extracurricular', 'internet_access',
            'previous_grade', 'final_score'
        ]
        
        input_data = [float(request.form[col]) for col in feature_order]
        prediction_result = model.predict([input_data])[0]

        return render_template_string(HTML_TEMPLATE, prediction=prediction_result)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

# Entry point for Vercel WSGI
app = app
