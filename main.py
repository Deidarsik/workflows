from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values
        diabetes = int(request.form['diabetes'])
        ctnrise = int(request.form['ctnrise'])

        # Check for double comorbidity condition
        if diabetes == 1 and ctnrise == 1:
            result = ("The model does not include incidence of double comorbidity of Rise in Troponin and "
                      "Diabetes Mellitus 2. Thus, it cannot predict likelihood of developing myocarditis in such cases.")
        else:
            # Prepare the input data array
            data = np.array([[diabetes, ctnrise]])
            data_scaled = scaler.transform(data)

            # Get model's prediction
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[0][1]

            # Determine the risk based on the input values
            high_risk = diabetes == 1 or ctnrise == 1

            # Format the result based on risk assessment
            if high_risk:
                result = f"High Probability of Myocarditis: {probability:.2f}%"
            else:
                result = "Low Probability of Myocarditis"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
