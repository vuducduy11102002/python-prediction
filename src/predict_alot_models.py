import mlflow
import os
import logging
import joblib
import yaml
import time
import numpy as np
import pandas as pd 
from config import Config

class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
            print(self.config)
            print("load config")

        logging.info(f"model-config: {self.config}")
        mlflow.set_tracking_uri(Config.MLFLOW_URI)

        # Lấy danh sách các mô hình.
        models = self.config["models"]

        # Tạo một danh sách các mô hình được tải.
        self.loaded_models = []

        # Tải từng mô hình.
        for model in models:
            model_uri = f"models:/{model['model_name']}@{model['model_version']}"
            self.loaded_models.append(mlflow.sklearn.load_model(model_uri))

        # Lưu tất cả các mô hình đã tải vào tệp.
        joblib.dump(self.loaded_models, open("models/all_models.pkl", 'wb'))
        logging.info(f"all models saved successfully")

        # Tải scaler.
        self.scaler = joblib.load("models/scaler.pkl")
        logging.info("scaler loaded")

    def predict(self, df: np.ndarray):
        start_time = time.time()
        df = self.scaler.transform(df)

        # Dự đoán bằng tất cả các mô hình.
        y_pred = []
        for model in self.loaded_models:
            y_pred.append(model.predict(df))

        end_pred = time.time()
        logging.info(f"predict time: {end_pred-start_time}")
        return y_pred


if __name__ == "__main__":
    val = pd.read_csv("data/val.csv")
    predictor = ModelPredictor("config/model.yaml")
    val_X = val.drop(columns=['Outcome'])
    val_y = val['Outcome']
    y_pred = predictor.predict(val_X)

    print(y_pred)
