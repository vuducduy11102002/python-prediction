import argparse
import logging
import os as pd
import pickle

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# import logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import Config


class ModelTrainer:
    EXPERIMENT_NAME = None

    dict_model = {
        "xgb": xgb.XGBClassifier(),
        "svm": SVC(),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(),
        "mlp": MLPClassifier(),
        "ada_boost": AdaBoostClassifier(),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(),
        "lightgbm": LGBMClassifier(),
        "logistic_regression": LogisticRegression(),
    }

    @staticmethod
    def train_model(model_name):
        ModelTrainer.EXPERIMENT_NAME = "diabetes_" + model_name
        logging.info(f"bắt đầu huấn luyện mô hình {model_name}")

        # init mlflow experiment
        mlflow.set_tracking_url = Config.MLFLOW_URI
        mlflow.set_experiment(ModelTrainer.EXPERIMENT_NAME)

        # load data
        df_train = pd.read_csv("data/train.csv")
        df_test = pd.read_csv("data/test.csv")

        # scaler
        scaler = StandardScaler()

        X_train = scaler.fit_transform(df_train.drop(columns=["Outcome"]))
        X_test = scaler.transform(df_test.drop(columns=["Outcome"]))

        # create model
        model = ModelTrainer.dict_model.get(model_name)

        # training model
        model.fit(X_train, df_train["Outcome"])

        # metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(df_test["Outcome"], y_pred)
        precision = precision_score(df_test["Outcome"], y_pred)
        recall = recall_score(df_test["Outcome"], y_pred)
        f1 = f1_score(df_test["Outcome"], y_pred)
        roc_auc = roc_auc_score(df_test["Outcome"], y_pred)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }
        logging.info(f"model {model_name} metrics: {metrics}")

        # mlflow logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.end_run()
        logging.info("hoàn thành huấn luyện mô hình")

    @staticmethod
    def train_all_models():
        for model_name in ModelTrainer.dict_model.keys():
            ModelTrainer.train_model(model_name)


if __name__ == "__main__":
    ModelTrainer.train_all_models()
