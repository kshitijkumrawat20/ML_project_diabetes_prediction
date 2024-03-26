from flask import Flask, jsonify, request,app, render_template, Response
import pickle 
import numpy as np 
import pandas as pd

application = Flask(__name__)

app = application

scaler = pickle.load(open("model/standardScaler.pkl","rb"))
model = pickle.load(open("model/naiveModel.pkl","rb"))

## route for home page 
@app.route('/')
def index():
    return render_template('index.html')

# ROUTE for single data point prediction 
@app.route('/prediction', methods=['GET','POST'])
def predict_datapoints():
    result = ""
    if request.method =='POST':

        Pregnencies = int(request.form.get("Pregnancies", 0)) 
        Glucose = float(request.form.get('Glucose', 0))
        BloodPressure = float(request.form.get('BloodPressure', 0))
        SkinThickness = float(request.form.get('SkinThickness', 0))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))

        new_data = scaler.transform([[Pregnencies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model.predict(new_data)

        if prediction[0]==1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'
            
        return render_template('result.html', result= result)
    else:
        return render_template('index.html')        


if __name__ == "__main__":
    app.run(host="0.0.0.0")