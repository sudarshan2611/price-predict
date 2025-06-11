# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model/house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        rooms = float(request.form['rooms'])
        occupancy = float(request.form['occupancy'])

        features = np.array([[income, rooms, occupancy]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=f"Estimated House Price: ${prediction * 100000:.2f}")
    except Exception as e:
        return render_template('index.html', prediction="Error in input or prediction")

if __name__ == '__main__':
    app.run(debug=True)
