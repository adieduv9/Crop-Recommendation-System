from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from datetime import datetime
from weather import search_city, get_weather_by_coords

app = Flask(__name__)

# Load model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print("Model loading error:", e)
    model = None

# Load scaler safely
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("Scaler loading error:", e)
    scaler = None


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

    # Check model availability
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

    # Future projection
    projected_temp = temperature + (0.4 * duration)

    current_month = datetime.now().month
    future_month = (current_month + duration - 1) % 12 + 1
    season = get_season(future_month)

    features = np.array([[nitrogen, phosphorus, potassium,
                          projected_temp, humidity, ph, rainfall]])

    # Apply scaler only if available
    try:
        if scaler is not None:
            features = scaler.transform(features)
    except Exception as e:
        return render_template("index.html", error=f"Scaler error: {e}")

    # Predict
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {e}")

    return render_template("index.html",
                           prediction=prediction,
                           season=season,
                           temperature=round(projected_temp, 2),
                           humidity=humidity)


if __name__ == "__main__":
    app.run(debug=True)