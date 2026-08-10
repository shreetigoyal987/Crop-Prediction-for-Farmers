from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')  # 'templates' folder must contain index.html

# Load your trained Random Forest model
model = pickle.load(open('model.pkl', 'rb'))  # model.pkl must be in the same folder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values from index.html
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Convert input to NumPy array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict crop
        prediction = model.predict(features)
        output = prediction[0]

        return render_template('index.html', prediction_text=f"🌾 Recommended Crop: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
