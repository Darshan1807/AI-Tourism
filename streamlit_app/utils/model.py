import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.preprocessing import (
    preprocess_pipeline, get_feature_columns, save_encoders, load_encoders,
    encode_features, engineer_features, apply_agg_stats
)
from utils.data_loader import load_raw_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_models(force_retrain: bool = False):
    demand_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    crowd_path = os.path.join(MODELS_DIR, "crowd_model.pkl")

    if not force_retrain and os.path.exists(demand_path) and os.path.exists(crowd_path):
        return load_models()

    df_raw = load_raw_data()
    df, encoders, lower_thresh, upper_thresh, agg_stats = preprocess_pipeline(df_raw)

    features = get_feature_columns()
    available_features = [f for f in features if f in df.columns]

    X = df[available_features].fillna(0)
    y_demand = df["Visitors_Count"]
    y_crowd = df["Overcrowding_Level"]

    crowd_encoder = LabelEncoder()
    y_crowd_enc = crowd_encoder.fit_transform(y_crowd)

    X_train, X_test, y_d_train, y_d_test, y_c_train, y_c_test = train_test_split(
        X, y_demand, y_crowd_enc, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    demand_model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.85,
        max_features="sqrt",
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    demand_model.fit(X_train_sc, y_d_train)
    d_pred = demand_model.predict(X_test_sc)
    d_r2 = r2_score(y_d_test, d_pred)
    d_rmse = np.sqrt(mean_squared_error(y_d_test, d_pred))
    demand_metrics = {"r2": d_r2, "rmse": d_rmse, "model_name": "Gradient Boosting Regressor"}

    crowd_model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.85,
        max_features="sqrt",
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    crowd_model.fit(X_train_sc, y_c_train)
    c_pred = crowd_model.predict(X_test_sc)
    c_acc = accuracy_score(y_c_test, c_pred)
    cm = confusion_matrix(y_c_test, c_pred)
    cr = classification_report(
        y_c_test, c_pred, target_names=crowd_encoder.classes_, output_dict=True
    )
    crowd_metrics = {"accuracy": c_acc, "model_name": "Gradient Boosting Classifier"}

    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(demand_path, "wb") as f:
        pickle.dump({
            "model": demand_model,
            "scaler": scaler,
            "metrics": demand_metrics,
            "features": available_features,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(crowd_path, "wb") as f:
        pickle.dump({
            "model": crowd_model,
            "scaler": scaler,
            "metrics": crowd_metrics,
            "crowd_encoder": crowd_encoder,
            "features": available_features,
            "confusion_matrix": cm,
            "classification_report": cr,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_encoders(encoders, lower_thresh, upper_thresh, agg_stats)

    return {
        "demand": {
            "model": demand_model, "scaler": scaler,
            "metrics": demand_metrics, "features": available_features,
        },
        "crowd": {
            "model": crowd_model, "scaler": scaler,
            "metrics": crowd_metrics, "crowd_encoder": crowd_encoder,
            "features": available_features,
            "confusion_matrix": cm, "classification_report": cr,
        }
    }


def load_models():
    demand_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    crowd_path = os.path.join(MODELS_DIR, "crowd_model.pkl")
    with open(demand_path, "rb") as f:
        demand_data = pickle.load(f)
    with open(crowd_path, "rb") as f:
        crowd_data = pickle.load(f)
    return {"demand": demand_data, "crowd": crowd_data}


def models_exist():
    demand_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    crowd_path = os.path.join(MODELS_DIR, "crowd_model.pkl")
    enc_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    return (
        os.path.exists(demand_path)
        and os.path.exists(crowd_path)
        and os.path.exists(enc_path)
    )


if __name__ == "__main__":
    print("Training models...")
    result = train_models(force_retrain=True)
    print(f"Demand: {result['demand']['metrics']}")
    print(f"Crowd:  {result['crowd']['metrics']}")
    print("Done.")
