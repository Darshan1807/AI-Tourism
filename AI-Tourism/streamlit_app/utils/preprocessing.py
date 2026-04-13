import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CURRENT_YEAR = 2025

_AGG_STATS = {}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Establishment_Year"] = pd.to_numeric(df["Establishment_Year"], errors="coerce")
    median_year = df["Establishment_Year"].median()
    df["Establishment_Year"] = df["Establishment_Year"].fillna(median_year)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Weekly_Off"] = df["Weekly_Off"].fillna("None")
    df["Weather_Type"] = df["Weather_Type"].fillna("Unknown")
    for col in ["Google_Rating", "Review_Count_Lakhs", "Ticket_Price", "Revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["Visitors_Count"] = pd.to_numeric(df["Visitors_Count"], errors="coerce")
    df["Visitors_Count"] = df["Visitors_Count"].fillna(df["Visitors_Count"].median())
    df["Is_Weekend"] = df["Is_Weekend"].map({"Yes": 1, "No": 0}).fillna(0)
    df["Airport_Within_50km"] = df["Airport_Within_50km"].map({"Yes": 1, "No": 0}).fillna(0)
    df["DSLR_Allowed"] = df["DSLR_Allowed"].map({"Yes": 1, "No": 0}).fillna(0)
    return df


def build_agg_stats(df: pd.DataFrame) -> dict:
    """Compute aggregated statistics from training data."""
    stats = {}
    stats["state_avg"] = df.groupby("Location_State")["Visitors_Count"].mean().to_dict()
    stats["place_type_avg"] = df.groupby("Place_Type")["Visitors_Count"].mean().to_dict()
    stats["season_avg"] = df.groupby("Season")["Visitors_Count"].mean().to_dict()
    stats["weather_avg"] = df.groupby("Weather_Type")["Visitors_Count"].mean().to_dict()
    stats["state_season_avg"] = (
        df.groupby(["Location_State", "Season"])["Visitors_Count"].mean()
        .reset_index().set_index(["Location_State", "Season"])["Visitors_Count"].to_dict()
    )
    stats["place_type_season_avg"] = (
        df.groupby(["Place_Type", "Season"])["Visitors_Count"].mean()
        .reset_index().set_index(["Place_Type", "Season"])["Visitors_Count"].to_dict()
    )
    stats["global_avg"] = df["Visitors_Count"].mean()
    stats["global_median"] = df["Visitors_Count"].median()
    return stats


def apply_agg_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    global_avg = stats.get("global_avg", 300)
    df["State_Avg_Visitors"] = df["Location_State"].map(
        stats.get("state_avg", {})
    ).fillna(global_avg)
    df["PlaceType_Avg_Visitors"] = df["Place_Type"].map(
        stats.get("place_type_avg", {})
    ).fillna(global_avg)
    df["Season_Avg_Visitors"] = df["Season"].map(
        stats.get("season_avg", {})
    ).fillna(global_avg)
    df["Weather_Avg_Visitors"] = df["Weather_Type"].map(
        stats.get("weather_avg", {})
    ).fillna(global_avg)
    df["State_Season_Avg"] = df.apply(
        lambda r: stats.get("state_season_avg", {}).get(
            (r["Location_State"], r["Season"]), global_avg
        ), axis=1
    )
    df["PlaceType_Season_Avg"] = df.apply(
        lambda r: stats.get("place_type_season_avg", {}).get(
            (r["Place_Type"], r["Season"]), global_avg
        ), axis=1
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Place_Age"] = (CURRENT_YEAR - df["Establishment_Year"]).clip(lower=0)
    df["Rating_Popularity"] = df["Google_Rating"] * df["Review_Count_Lakhs"]
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["Month_Num"] = df["Month"].map(month_map).fillna(1)
    season_peak = {"Summer": 1, "Winter": 1, "Monsoon": 0, "Autumn": 0}
    df["Is_Peak_Season"] = df["Season"].map(season_peak).fillna(0)
    df["Has_Ticket"] = (df["Ticket_Price"] > 0).astype(int)
    return df


def add_overcrowding_label(df: pd.DataFrame) -> tuple:
    df = df.copy()
    lower = df["Visitors_Count"].quantile(0.33)
    upper = df["Visitors_Count"].quantile(0.66)

    def classify(v):
        if v <= lower:
            return "Low"
        elif v <= upper:
            return "Medium"
        else:
            return "High"

    df["Overcrowding_Level"] = df["Visitors_Count"].apply(classify)
    return df, lower, upper


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    df = df.copy()
    categorical_cols = [
        "Month", "Season", "Day_of_Week", "Weather_Type",
        "Zone", "Place_Type", "Significance", "Tourist_Type",
        "Location_State", "Weekly_Off"
    ]
    if fit:
        encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        return df, encoders
    else:
        for col in categorical_cols:
            if col in df.columns and col in encoders:
                le = encoders[col]
                df[col + "_enc"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df


def get_feature_columns():
    return [
        "Month_enc", "Season_enc", "Day_of_Week_enc", "Weather_Type_enc",
        "Zone_enc", "Place_Type_enc", "Significance_enc", "Tourist_Type_enc",
        "Location_State_enc", "Weekly_Off_enc",
        "Is_Weekend", "Airport_Within_50km", "DSLR_Allowed",
        "Google_Rating", "Review_Count_Lakhs", "Ticket_Price",
        "Year", "Place_Age", "Rating_Popularity",
        "Month_Num", "Is_Peak_Season", "Has_Ticket",
        "State_Avg_Visitors", "PlaceType_Avg_Visitors",
        "Season_Avg_Visitors", "Weather_Avg_Visitors",
        "State_Season_Avg", "PlaceType_Season_Avg",
    ]


def preprocess_pipeline(df: pd.DataFrame):
    df = clean_data(df)
    df = engineer_features(df)
    agg_stats = build_agg_stats(df)
    df = apply_agg_stats(df, agg_stats)
    df, lower_thresh, upper_thresh = add_overcrowding_label(df)
    df, encoders = encode_features(df, fit=True)
    return df, encoders, lower_thresh, upper_thresh, agg_stats


def save_encoders(encoders: dict, lower_thresh: float, upper_thresh: float, agg_stats: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump({
            "encoders": encoders,
            "lower_thresh": lower_thresh,
            "upper_thresh": upper_thresh,
            "agg_stats": agg_stats,
        }, f)


def load_encoders():
    path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    if not os.path.exists(path):
        return None, None, None, {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (
        data["encoders"],
        data["lower_thresh"],
        data["upper_thresh"],
        data.get("agg_stats", {})
    )
