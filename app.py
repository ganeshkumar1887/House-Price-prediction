from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        medinc = float(request.form['MedInc'])
        houseage = float(request.form['HouseAge'])
        latitude = float(request.form['Latitude'])

        # Arrange input in correct order
        input_data = [[medinc, houseage, latitude]]

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        output = round(prediction[0], 3)

        return render_template('index.html', prediction_text=f'Predicted House Price: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run app
if __name__ == "__main__":
    app.run(debug=True)