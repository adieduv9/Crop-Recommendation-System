from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from datetime import datetime
from weather import search_city, get_weather_by_coords

app = Flask(__name__)

# Load model and scaler safely
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print("Model loading error:", e)
    model = None

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("Scaler loading error:", e)
    scaler = None

# Map numeric labels to crop names
crop_dict = {
    0: "Wheat",
    1: "Rice",
    2: "Maize",
    3: "Barley",
    4: "Soybean",
    5: "Cotton",
    6: "Sugarcane",
    7: "Groundnut",
    8: "Millets",
    9: "Tea"
}

# Seasonal factor for temperature projection
seasonal_temp_factors = {
    "Kharif": 1.0,
    "Rabi": 0.8,
    "Summer": 1.2
}

def get_season(month):
    if month in [6, 7, 8, 9]:
        return "Kharif"
    elif month in [10, 11, 12, 1]:
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
    if model is None:
        return render_template("index.html", error="Model not loaded.")

    # Validate city selection
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    if not lat or not lon:
        return render_template("index.html", error="Please select a city from suggestions.")
    try:
        lat = float(lat)
        lon = float(lon)
    except:
        return render_template("index.html", error="Invalid city selection.")

    # Validate input values
    try:
        duration = int(request.form["duration"])
        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
    except:
        return render_template("index.html", error="Invalid input values.")

    # Get weather
    weather = get_weather_by_coords(lat, lon)
    if weather is None:
        return render_template("index.html", error="Weather API error.")
    temperature, humidity = weather

    # Determine season
    current_month = datetime.now().month
    future_month = (current_month + duration - 1) % 12 + 1
    season = get_season(future_month)

    # Project temperature dynamically
    factor = seasonal_temp_factors.get(season, 1.0)
    projected_temp = temperature * factor

    # Prepare features for model
    features = np.array([[nitrogen, phosphorus, potassium,
                          projected_temp, humidity, ph, rainfall]])

    # Apply scaler if available
    try:
        if scaler is not None:
            features = scaler.transform(features)
    except Exception as e:
        return render_template("index.html", error=f"Scaler error: {e}")

    # Predict top crops
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            top_indices = np.argsort(probs)[::-1][:3]  # top 3
            top_crops = [(crop_dict.get(i, "Unknown"), round(probs[i]*100, 2)) for i in top_indices]
            main_prediction = top_crops[0][0]
        else:
            main_prediction = crop_dict.get(model.predict(features)[0], "Unknown")
            top_crops = [(main_prediction, 100.0)]
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {e}")

    return render_template("index.html",
                           prediction=main_prediction,
                           top_crops=top_crops,
                           season=season,
                           temperature=round(projected_temp, 2),
                           humidity=humidity)

if __name__ == "__main__":
    app.run(debug=True)