# Smart Umbrella - Rain Prediction and Monitoring System

## Overview

The Smart Umbrella system is an end-to-end AI-powered weather-aware platform designed to predict rainfall based on humidity and temperature inputs. It leverages a trained neural network to make predictions and supports a live interface for IoT devices to log and retrieve real-time weather data.

This repository contains two major components:

* **Flask Backend**: Handles API requests, connects to MongoDB, and serves prediction and data visualization endpoints.
* **Rain Prediction Model**: A TensorFlow-based neural network trained on weather data, bundled with a StandardScaler for input normalization.

---

## Features

* Predicts whether it will rain based on humidity and temperature.
* Collects data from physical IoT devices like ESP32 or NodeMCU via API.
* Logs data with timestamps into MongoDB Atlas.
* Provides historical and real-time weather data retrieval.
* Secures sensitive keys using `.env` variables.
* Supports test data injection to simulate device behavior.

---

## Tech Stack

* **Backend**: Flask, TensorFlow, NumPy, PyMongo, joblib
* **Database**: MongoDB Atlas
* **Model**: TensorFlow (Neural Network)
* **Data**: weatherAUS.csv (real-world dataset)

---

## MongoDB Schema

Collection: `predictions`

```json
{
  "umbrella_id": "UMBRELLA_7CFA12B3",
  "temperature": 29.5,
  "humidity": 68.2,
  "prediction": 1,
  "probability": 0.83,
  "timestamp": "2025-06-30 20:00:00 IST+0530"
}
```

---

## API Endpoints

### `POST /sendData`

Receives sensor data and logs prediction to MongoDB.
**Payload:**

```json
{
  "umbrella_id": "UMBRELLA_7CFA12B3",
  "temperature": "30.5",
  "humidity": "70.1"
}
```

**Response:**

```json
{
  "prediction": 1,
  "confidence": 0.83
}
```

### `GET /getPrediction?umbrella_id=...`

Returns the latest prediction for a specific umbrella.

### `GET /getHistoricalData?umbrella_id=...`

Returns the last 10 records for a specific umbrella.

### `GET /delete_all`

Deletes all entries in the MongoDB collection.

### `GET /insertTestData`

Simulates 5 test data entries (used during development/testing).

---

## How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/smart-umbrella.git
cd smart-umbrella
```

### 2. Set Up Environment Variables

Create a `.env` file with the following content:

```env
MONGO_URI=your_mongodb_connection_string
MODEL_PATH=rain_model.h5
DEVICE_IP=127.0.0.1  # or your ESP32's IP if using real device
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Required libraries: Flask, TensorFlow, NumPy, joblib, python-dotenv, pymongo, pytz, requests

### 4. Train the Model (Optional)

You can run the model training script if you'd like to retrain using your own data:

```bash
python train_model.py
```

### 5. Start Flask Server

```bash
python app.py
```

The server will run on `http://0.0.0.0:5000`

---

## Security and Best Practices

* All credentials and device IPs are hidden in `.env` using `python-dotenv`.
* Never upload `.env` to GitHub. Add it to `.gitignore`.
* Model weights and scaler are bundled using `joblib` for reproducibility.

---

## Use Cases

* Weather-aware umbrella devices
* Smart home integration for outdoor alerts
* Agricultural decision support systems
* IoT-integrated weather prediction models

---

## Author

Developed by **Achuth G**. For questions or collaboration, feel free to connect via GitHub or [email](mailto:achuthganesh09@gmail.com).
