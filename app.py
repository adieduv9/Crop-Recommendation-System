from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from datetime import datetime
from weather import search_city, get_weather_by_coords

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def get_season(month):
    if month in [6,7,8,9]:
        return "Kharif"
    elif month in [10,11,12,1]:
        return "Rabi"
    else:
        return "Summer"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search-city")
def city_search():
    query = request.args.get("q")
    if not query:
        return jsonify([])
    cities = search_city(query)
    return jsonify(cities)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
        duration = int(request.form["duration"])

        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        weather = get_weather_by_coords(lat, lon)
        if weather is None:
            return render_template("index.html", error="Weather API error")

        temperature, humidity = weather

        # Project future temperature
        projected_temp = temperature + (0.4 * duration)

        current_month = datetime.now().month
        future_month = (current_month + duration - 1) % 12 + 1
        season = get_season(future_month)

        features = np.array([[nitrogen, phosphorus, potassium,
                              projected_temp, humidity, ph, rainfall]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return render_template("index.html",
                               prediction=prediction,
                               season=season,
                               temperature=round(projected_temp, 2),
                               humidity=humidity)

    except Exception:
        return render_template("index.html", error="Something went wrong.")

if __name__ == "__main__":
    app.run(debug=True)