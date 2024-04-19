import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data, read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import json
import mlflow
from urllib.parse import urlparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids="3")
    runs = mlflow.search_runs([3])
    lowest=runs["metrics.rmse"].sort_values(ascending=True)[0]
    #lowest_run_id = runs[runs["metrics.rmse"]==lowest]["run_id"][0]
    #print(lowest)

    client=MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv=dict(mv)
        if mv["run_id"]==f621fd0c9eab4d418a25d08015706735:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)

            client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
        else:
            current_version=mv["version"]
            client.transition_model_version_stage(name=model_name, version=current_version, stage="Stagging")
            load_model = mlflow.pyfunc.log_model(logged_model)
            model_path = config["model_dirs"]
            os.system(f"attrib +h {model_path}")
            file_ = open(model_path, "w")
            joblib.dump(load_model, model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    log_production_model(config_path=parsed_args.config)