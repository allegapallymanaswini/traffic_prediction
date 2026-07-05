import requests
import time
import json
import random

# Mock Kafka Stream Producer
# This simulates real-time data arriving for prediction

BASE_URL = "http://127.0.0.1:5000"

def simulate_streaming():
    print("Starting Traffic Data Stream Simulation (Kafka Mock)...")

    # Get available segments from the API
    try:
        segments = requests.get(f"{BASE_URL}/api/segments").json()['segments']
    except:
        segments = ["RS-0175_Z1", "RS-0134_Z1"]

    while True:
        # Generate a realistic random traffic event with all required features
        event_data = {
            "Road_Segment_ID": random.choice(segments),
            "Weather_Condition": random.choice(["Sunny", "Rain", "Cloudy", "Heavy Rain", "Mist"]),
            "Hour": time.localtime().tm_hour,
            "DayOfWeek": time.localtime().tm_wday,
            "Month": time.localtime().tm_mon,
            "Historical_Average_Speed": random.uniform(30, 90),
            "Temperature_C": random.uniform(15, 35),
            "Humidity_Percent": random.uniform(40, 90),
            "Rainfall_mm": random.uniform(0, 15),
            "Visibility_km": random.uniform(1, 10),
            "Wind_Speed_kmph": random.uniform(0, 20),
            "Road_Type": random.choice(["Highway", "Residential", "Urban", "Rural"]),
            "Number_of_Lanes": random.choice([2, 3, 4, 6]),
            "Speed_Limit_kmph": random.choice([40, 50, 60, 80, 100]),
            "Traffic_Signals": random.choice(["Yes", "No"]),
            "Nearby_Intersections": random.randint(1, 10),
            "Event_Type": random.choice(["None", "Accident", "Construction", "Roadwork"]),
            "Event_Flag": random.choice([0, 1]),
            "prev_hour_traffic": random.randint(50, 400),
            "prev_2hour_traffic": random.randint(50, 400),
            "rolling_mean_3h": random.randint(50, 400)
        }

        print(f"Streaming Event: {event_data['Road_Segment_ID']} | Hour: {event_data['Hour']} | Weather: {event_data['Weather_Condition']}")

        # Send to API for real-time prediction
        try:
            response = requests.post(f"{BASE_URL}/api/predict", json=event_data)
            result = response.json()
            print(f"  Prediction: {round(result['prediction'])} | Congestion: {result['congestion_level']} | Risk: {result['accident_risk']}")
        except Exception as e:
            print(f"  Stream Error: {e}")

        time.sleep(5) # Stream every 5 seconds

if __name__ == "__main__":
    simulate_streaming()
