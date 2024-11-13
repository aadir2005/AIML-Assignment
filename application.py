import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialize the Flask application
application = Flask(__name__)
app = application

# Load the Ridge regressor model and StandardScaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # Retrieve form data
        try:
            temperature = float(request.form['Temperature'])
            rh = float(request.form['RH'])
            ws = float(request.form['Ws'])
            rain = float(request.form['Rain'])
            ffmc = float(request.form['FFMC'])
            dmc = float(request.form['DMC'])
            isi = float(request.form['ISI'])
            classes = float(request.form['Classes'])
            region = float(request.form['Region'])

            # Create an array for the input features
            features = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

            # Scale the features
            scaled_features = standard_scaler.transform(features)

            # Predict using the Ridge model
            prediction = ridge_model.predict(scaled_features)

            # Return the result to the HTML template
            return render_template('home.html', result=prediction[0])

        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return render_template('home.html', result="")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
