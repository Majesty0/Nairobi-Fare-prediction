from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load pre-trained model, encoder, and imputer
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

app = Flask(__name__)

# Prediction function
def predict_price(cab_type, destination, wind, rain, distance, humidity, temp, clouds):
    try:
        # Create a dataframe with the input values
        input_df = pd.DataFrame({
            'cab_type': [cab_type],
            'destination': [destination],
            'wind': [wind],
            'rain': [rain],
            'distance': [distance],
            'humidity': [humidity],
            'temp': [temp],
            'clouds': [clouds]
        })

        # Encode categorical features
        input_cat = encoder.transform(input_df[['cab_type', 'destination']])
        input_num = input_df[['wind', 'rain', 'distance', 'humidity', 'temp', 'clouds']]

        # Handle missing values for numerical features
        input_num_imputed = imputer.transform(input_num)

        # Combine encoded and numerical features
        input_final = np.hstack((input_num_imputed, input_cat))

        # Predict the price
        predicted_price = model.predict(input_final)

        return f"ksh {predicted_price[0]:.2f}"
    
    except Exception as e:
        return f"Error: {e}"

# Home route to display the form
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        cab_type = request.form['cab_type']
        destination = request.form['destination']
        wind = float(request.form['wind'])
        rain = float(request.form['rain'])
        distance = float(request.form['distance'])
        humidity = float(request.form['humidity'])
        temp = float(request.form['temp'])
        clouds = float(request.form['clouds'])

        # Call the prediction function
        predicted_price = predict_price(cab_type, destination, wind, rain, distance, humidity, temp, clouds)
        
        return render_template('index.html', predicted_price=predicted_price)
    
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
