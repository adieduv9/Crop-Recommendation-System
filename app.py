from flask import Flask, request, render_template
import numpy as np
import pickle

# Create Flask app
app = Flask(__name__)

# Load model and scalersN
model_path = 'model.pkl'
scaler_stand_path = 'standscaler.pkl'
scaler_minmax_path = 'minmaxscaler.pkl'

model = pickle.load(open(model_path, 'rb'))
sc = pickle.load(open(scaler_stand_path, 'rb'))
ms = pickle.load(open(scaler_minmax_path, 'rb'))

# Define routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get input data from the form
    N = request.form.get('Nitrogen')
    P = request.form.get('Phosporus')
    K = request.form.get('Potassium')
    temp = request.form.get('Temperature')
    humidity = request.form.get('Humidity')
    ph = request.form.get('Ph')
    rainfall = request.form.get('Rainfall')

    # Create feature list and reshape
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Make prediction
    prediction = model.predict(final_features)

    # Map prediction to crop
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Get the crop name and image filename based on prediction
    crop = crop_dict.get(prediction[0], "Unknown")
    image = f"{prediction[0]}.jpg" if prediction[0] in crop_dict else None
    result = f"{crop} is the best crop to be cultivated right there" if crop != "Unknown" else "Sorry, we could not determine the best crop to be cultivated with the provided data."

    print(f"Prediction: {prediction[0]}")
    print(f"Crop: {crop}")
    print(f"Image: {image}")

    return render_template('index.html', result=result, image=image)

    

    # # Get the crop name based on prediction
    # crop = crop_dict.get(prediction[0], "Unknown")
    # result = f"{crop} is the best crop to be cultivated right there" if crop != "Unknown" else "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
