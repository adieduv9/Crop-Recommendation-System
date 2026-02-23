from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime
from weather import get_weather

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("standscaler.pkl", "rb"))

def get_season(month):
    if month in [6,7,8,9]:
        return "Kharif"
    elif month in [10,11,12,1]:
        return "Rabi"
    else:
        return "Summer"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        try:
            city = request.form["city"]
            duration = int(request.form["duration"])

            nitrogen = float(request.form["nitrogen"])
            phosphorus = float(request.form["phosphorus"])
            potassium = float(request.form["potassium"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Get live weather
            weather = get_weather(city)
            if weather is None:
                return render_template("index.html", error="Invalid city or API error.")

            temperature, humidity = weather

            # Project future weather
            projected_temp = temperature + (0.4 * duration)

            current_month = datetime.now().month
            future_month = (current_month + duration - 1) % 12 + 1
            season = get_season(future_month)

            # Prepare input for model
            features = np.array([[nitrogen, phosphorus, potassium,
                                  projected_temp, humidity, ph, rainfall]])

            scaled = scaler.transform(features)
            prediction = model.predict(scaled)[0]

            return render_template("index.html",
                                   prediction=prediction,
                                   season=season,
                                   temperature=round(projected_temp,2),
                                   humidity=humidity)

        except Exception as e:
            return render_template("index.html", error="Something went wrong.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)