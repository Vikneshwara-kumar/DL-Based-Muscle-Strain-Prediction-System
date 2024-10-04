from fastapi import FastAPI
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import json
from json import dumps
import threading
from time import sleep
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from Feature_extractor import ProcessData  # Ensure this is correctly implemented

# InfluxDB configuration
token = "HPskNZN757ImNKb7CmOtEpfX4y2wD_2jiLNZ_TdXYMyBBUqkFBKMa26rCXaCpNuia-MbW7nSrhe9-DdFzQ_VvA=="
org = "iff"
url = "http://192.168.50.202:8086"
write_client = InfluxDBClient(url=url, token=token, org=org)

KAFKA_BROKER = '192.168.50.234:29093'
RAW_DATA_TOPIC = 'delsys-source-emg01'
EXTRACTED_FEATURES_TOPIC = 'extracted_features'

fs = 1000
FS = 2148
window_size = 300
window_shift = 150
feature_list = ['RMS', 'MAV', 'WL', 'ZC', 'MDF', 'MNF', 'MNP', 'SSC']  # List of required features
feature_columns = ['RMS', 'MAV', 'WL', 'SSC', 'MDF', 'MNF', 'MNP', 'PSD', 'stft_feature_1', 'stft_feature_2', 'stft_feature_3']

# Initialize FastAPI application
app = FastAPI()

consumer = KafkaConsumer(
    RAW_DATA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    group_id='200',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    auto_commit_interval_ms=2000,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def save_to_influx(bucket_name, df):
    json_body = []
    write_api = write_client.write_api(write_options=WritePrecision.NS)

    for index, row in df.iterrows():
        json_body.append({
            "measurement": "Feature_Extraction",
            "tags": {
                "topic": RAW_DATA_TOPIC  # Optional, add tags if needed
            },
            "time": pd.Timestamp.utcnow().isoformat(),
            "fields": row.to_dict()
        })

    try:
        write_api.write(bucket=bucket_name, org="iff", record=json_body)
        print("Data saved to InfluxDB successfully")
    except Exception as e:
        print(f"InfluxDB Write Error: {e}")


def save_to_influxT(bucket_name, response_time):
    json_body = [
        {
            "measurement": "Prediction_result",
            "tags": {},
            "time": pd.Timestamp.utcnow().isoformat(),
            "fields": {
                "Execution_time": response_time
            }
        }
    ]

    try:
        write_api = write_client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=bucket_name, org="iff", record=json_body)
        print("Data saved to InfluxDB successfully")
    except Exception as e:
        print(f"InfluxDB Write Error: {e}")


def process_data():
    for message in consumer:
        start_time = time.time()
        raw_data = message.value
        features = pd.DataFrame(raw_data)
        print(len(features))
        odp = ProcessData(features, FS, fs, window_size, window_shift, feature_list)
        processed_data = odp.process()
        print((processed_data))
        X1 = processed_data[feature_columns]

        df_json_str = X1.to_json(orient="records")
        df_json_obj = json.loads(df_json_str)

        producer.send(EXTRACTED_FEATURES_TOPIC, value=df_json_obj)
        response_time = time.time() - start_time
        print(response_time)

        # Save Processed data to InfluxDB
        save_to_influx("Microservice_Feature_Extraction", processed_data)
        save_to_influxT("Microservice_Feature_Extraction", response_time)


# API endpoint to start data processing manually (if needed)
@app.post("/start-processing")
def start_processing():
    thread = threading.Thread(target=process_data)
    thread.start()
    return {"message": "Data processing started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
