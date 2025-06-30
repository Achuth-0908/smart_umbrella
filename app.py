from flask import Flask, request, jsonify
from pymongo import MongoClient
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from flask_cors import CORS
import time
import random
import requests
import pytz
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  

# Load sensitive values from environment variables
MONGO_URI = os.getenv("MONGO_URI")
MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE_IP = os.getenv("DEVICE_IP")  # for /insertTestData requests

client = MongoClient(MONGO_URI)
db = client["SmartUmbrellaDB"]
collection = db["predictions"]

bundle = joblib.load("rain_predictor.pkl")
scaler = bundle["scaler"]
model = tf.keras.models.load_model(MODEL_PATH)
local_timezone = pytz.timezone("Asia/Kolkata")

@app.route("/sendData", methods=["POST"])
def receive_data():
    try:
        umbrella_id = request.form["umbrella_id"]
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])

        features = np.array([[humidity, temperature]])
        features_scaled = scaler.transform(features)

        prob = model.predict(features_scaled)[0][0]
        prediction = int(prob >= 0.5)

        data = {
            "umbrella_id": umbrella_id,
            "temperature": temperature,
            "humidity": humidity,
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "timestamp": datetime.now(local_timezone)
        }
        collection.insert_one(data)

        return jsonify({"prediction": prediction, "confidence": round(float(prob), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/getPrediction", methods=["GET"])
def get_prediction():
    umbrella_id = request.args.get('umbrella_id') 
    if not umbrella_id:
        return jsonify({"error": "umbrella_id is required"}), 400

    latest = collection.find({"umbrella_id": umbrella_id}).sort("timestamp", -1).limit(1)
    for doc in latest:
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"] + timedelta(hours=5, minutes=30)
        doc["timestamp"] = doc["timestamp"].astimezone(pytz.timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S %Z%z')
        return jsonify(doc)

    return jsonify({"error": "No data found for the given umbrella_id"}), 404

@app.route("/getHistoricalData", methods=["GET"])
def get_historical_data():
    umbrella_id = request.args.get('umbrella_id')
    if not umbrella_id:
        return jsonify({"error": "umbrella_id is required"}), 400

    historical_data = list(collection.find({"umbrella_id": umbrella_id}).sort("timestamp", -1).limit(10))
    
    data_points = []
    for doc in historical_data:
        doc["timestamp"] = doc["timestamp"] + timedelta(hours=5, minutes=30)
        time_str = doc["timestamp"].astimezone(pytz.timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S %Z%z')
        data_points.append({
            "umbrella_id": doc["umbrella_id"],
            "time": time_str,
            "temperature": doc["temperature"],
            "humidity": doc["humidity"]
        })
    
    return jsonify(data_points)


@app.route("/delete_all", methods=["GET", "DELETE"])
def delete_all_records():
    result = collection.delete_many({})
    return jsonify({
        "status": "success",
        "message": f"{result.deleted_count} documents deleted."
    })

@app.route("/insertTestData", methods=["GET"])
def insert_test_data():
    inserted_records = []

    for i in range(5):
        temperature = round(random.uniform(20.0, 35.0), 2)
        humidity = round(random.uniform(30.0, 80.0), 2)
        
        data = {
            "umbrella_id": "UMBRELLA_7CFA12B3",
            "temperature": str(temperature),
            "humidity": str(humidity)
        }

        try:
            response = requests.post(f"http://{DEVICE_IP}:5000/sendData", data=data)

            if response.status_code == 200:
                result = response.json()
                inserted_records.append({
                    "temperature": temperature,
                    "humidity": humidity,
                    "prediction": result.get("prediction"),
                    "probability": result.get("confidence"),
                    "timestamp": datetime.now(local_timezone)
                })
            else:
                inserted_records.append({
                    "error": f"Failed at record {i+1}",
                    "status_code": response.status_code,
                    "response": response.text
                })

        except Exception as e:
            inserted_records.append({
                "error": f"Exception at record {i+1}",
                "message": str(e)
            })

        time.sleep(3)

    return jsonify({
        "message": "Inserted 5 test records successfully.",
        "records": inserted_records
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
