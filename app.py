from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings

app = Flask(__name__)

loaded_model = pickle.load(open("recommendations.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_model1', methods=['POST'])


def predict_model1():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # if prediction[0] in crop_dict:
    #     crop = crop_dict[prediction[0]]
    #     result = "{} is suitable to cultivate".format(crop)
    # else:
    #     result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    if N <= 0 or P <= 0 or K <= 0 or temp <= -20 or humidity <= 0 or ph <= 0 or rainfall <= 0:
        result = "Sorry! Not able to recommend any crops for this environment"
        # print("Sorry! Not able to recommend any crops for this environment")
    elif N >= 400 or P >= 400 or K >= 400 or temp >= 80 or humidity >= 300 or ph >= 10 or rainfall >= 400:
        result = "Sorry! Not able to recommend any crops for this environment"
        # print("Sorry! Not able to recommend any crops for this environment")
    elif prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is suitable to cultivate".format(crop)
    else:
        result = "Sorry! Not able to recommend any crops for this environment"

    return render_template('index.html', result=result)


#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

@app.route('/predict_model2', methods=['POST'])

def predict_model2():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        predection = dtr.predict(transformed_features).reshape(1, -1)

        # if/else logic
    # if Year <= 0 or average_rain_fall_mm_per_year <= 0 or pesticides_tonnes <= 0 or avg_temp <= 0:
    #     predection = "Sorry! not able to predict for the given environment"
    # elif Year >= 4000 or average_rain_fall_mm_per_year >= 3500 or pesticides_tonnes >= 370000 or avg_temp >= 50:
    #     predection = "Sorry! not able to predict for the given environment"
    #
    # else:
    #     predection = "Sorry! not able to predict for the given environment"
        return render_template('index.html', predection=predection)

if __name__=="__main__":
    app.run(debug=True)