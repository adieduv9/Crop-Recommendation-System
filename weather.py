import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            return None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        return temperature, humidity

    except Exception:
        return None