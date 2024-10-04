from fastapi import FastAPI
from kafka import KafkaConsumer, KafkaProducer
import json
import threading
from tensorflow.keras.models import load_model
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

app = FastAPI()

feature_columns = ['RMS', 'MAV', 'SSC', 'WL', 'MNF', 'MDF', 'IMDF', 'IMPF', 'PSD', 'MNP', 'ZC', 'stft_feature_1', 'stft_feature_2', 'stft_feature_3', 'stft_feature_4', 'stft_feature_5', 'stft_feature_6']

# Load the model and scaler
model = load_model('/app/Inference/lstm.keras')
with open('/app/Inference/scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
print("Scaler has been loaded from 'scaler.pkl'")

# Read Kafka configurations from environment variables
KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'localhost:9092')
NORMALIZED_DATA_TOPIC = os.environ.get('NORMALIZED_DATA_TOPIC', 'extracted_features')
PREDICTION_TOPIC = os.environ.get('PREDICTION_TOPIC', 'predictions')

consumer = KafkaConsumer(
    NORMALIZED_DATA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    group_id='200',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    auto_commit_interval_ms=1000,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def infer(input_data):
    start_time = time.time()
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=-1)
    result = np.ndarray.item(predicted_class)
    end_time = time.time()
    latency = end_time - start_time
    print(f'Predicted Class: {predicted_class[0]}')
    return result, latency

def consume_and_process():
    df_ary = pd.DataFrame()
    global request_counter
    request_counter = 0
    for message in consumer:
        start_time = time.time()
        request_counter += 1
        data = message.value
        df = pd.DataFrame(data)
        df_ary = pd.concat([df_ary, df], ignore_index=True)
        if len(df_ary) >= 30:
            X2 = df_ary.iloc[:30]
            X_scaled = loaded_scaler.transform(X2)
            sequence = X_scaled.reshape((1, 30, len(feature_columns)))
            result, Infer_latency = infer(sequence)
            print("Predicted Strain Level is:", result)
            df_ary = pd.DataFrame()

# Define an API endpoint to trigger data processing manually
@app.post("/start-processing/")
async def start_processing():
    thread = threading.Thread(target=consume_and_process)
    thread.start()
    return {"message": "Processing started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
