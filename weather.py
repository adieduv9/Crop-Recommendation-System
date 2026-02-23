import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def search_city(query):
    url = f"https://api.openweathermap.org/geo/1.0/direct?q={query}&limit=5&appid={API_KEY}"
    response = requests.get(url, timeout=5)
    return response.json()

def get_weather_by_coords(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        return temperature, humidity
    except:
        return None