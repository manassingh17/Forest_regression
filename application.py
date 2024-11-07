from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load('models/gradient_boosting_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        "X": int(request.form['X']),
        "Y": int(request.form['Y']),
        "month": int(request.form['month']), 
        "day": int(request.form['day']),     
        "FFMC": float(request.form['FFMC']),
        "DMC": float(request.form['DMC']),
        "DC": float(request.form['DC']),
        "ISI": float(request.form['ISI']),
        "temp": float(request.form['temp']),
        "RH": int(request.form['RH']),
        "wind": float(request.form['wind']),
        "rain": float(request.form['rain'])
    }

   
    input_data = pd.DataFrame([data])
    
   
    prediction = model.predict(input_data)[0]
    
   
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
