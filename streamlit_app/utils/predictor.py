import numpy as np
import pandas as pd
from utils.preprocessing import (
    encode_features, engineer_features, apply_agg_stats,
    get_feature_columns, load_encoders
)
from utils.model import load_models, models_exist, train_models


def get_models_and_encoders():
    if not models_exist():
        train_models(force_retrain=False)
    models = load_models()
    encoders, lower_thresh, upper_thresh, agg_stats = load_encoders()
    return models, encoders, lower_thresh, upper_thresh, agg_stats


def build_input_row(
    month: str,
    year: int,
    season: str,
    day_of_week: str,
    is_weekend: int,
    weather_type: str,
    location_state: str,
    zone: str,
    place_type: str,
    significance: str,
    tourist_type: str,
    establishment_year: float,
    google_rating: float,
    review_count_lakhs: float,
    ticket_price: float,
    airport_within_50km: int,
    dslr_allowed: int,
    weekly_off: str,
) -> pd.DataFrame:
    row = {
        "Month": month,
        "Year": year,
        "Season": season,
        "Day_of_Week": day_of_week,
        "Is_Weekend": is_weekend,
        "Weather_Type": weather_type,
        "Location_State": location_state,
        "Zone": zone,
        "Place_Type": place_type,
        "Significance": significance,
        "Tourist_Type": tourist_type,
        "Establishment_Year": establishment_year,
        "Google_Rating": google_rating,
        "Review_Count_Lakhs": review_count_lakhs,
        "Ticket_Price": ticket_price,
        "Airport_Within_50km": airport_within_50km,
        "DSLR_Allowed": dslr_allowed,
        "Weekly_Off": weekly_off,
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    return df


def predict_visitors(input_df: pd.DataFrame, models: dict, encoders: dict, agg_stats: dict) -> float:
    df = apply_agg_stats(input_df.copy(), agg_stats)
    encoded = encode_features(df, fit=False, encoders=encoders)
    features = models["demand"]["features"]
    available = [f for f in features if f in encoded.columns]
    X = encoded[available].fillna(0)
    scaler = models["demand"]["scaler"]
    X_sc = scaler.transform(X)
    prediction = models["demand"]["model"].predict(X_sc)[0]
    return max(0, round(prediction, 2))


def predict_overcrowding(input_df: pd.DataFrame, models: dict, encoders: dict, agg_stats: dict) -> tuple:
    df = apply_agg_stats(input_df.copy(), agg_stats)
    encoded = encode_features(df, fit=False, encoders=encoders)
    features = models["crowd"]["features"]
    available = [f for f in features if f in encoded.columns]
    X = encoded[available].fillna(0)
    scaler = models["crowd"]["scaler"]
    X_sc = scaler.transform(X)
    crowd_model = models["crowd"]["model"]
    crowd_encoder = models["crowd"]["crowd_encoder"]
    pred_enc = crowd_model.predict(X_sc)[0]
    pred_label = crowd_encoder.inverse_transform([pred_enc])[0]
    proba_dict = None
    if hasattr(crowd_model, "predict_proba"):
        proba = crowd_model.predict_proba(X_sc)[0]
        proba_dict = {crowd_encoder.classes_[i]: round(float(p), 3) for i, p in enumerate(proba)}
    return pred_label, proba_dict


def classify_visitors_manual(visitors: float, lower_thresh: float, upper_thresh: float) -> str:
    if visitors <= lower_thresh:
        return "Low"
    elif visitors <= upper_thresh:
        return "Medium"
    else:
        return "High"
