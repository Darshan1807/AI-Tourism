import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "travel_data.csv")


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_records": len(df),
        "total_places": df["Place_Name"].nunique(),
        "total_states": df["Location_State"].nunique(),
        "total_visitors": int(df["Visitors_Count"].sum()),
        "avg_visitors": round(df["Visitors_Count"].mean(), 2),
        "avg_rating": round(df["Google_Rating"].mean(), 2),
        "total_revenue": int(df["Revenue"].sum()),
        "place_types": df["Place_Type"].nunique(),
    }


def get_top_places(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("Place_Name")["Visitors_Count"]
        .mean()
        .reset_index()
        .sort_values("Visitors_Count", ascending=False)
        .head(n)
    )


def get_season_trend(df: pd.DataFrame) -> pd.DataFrame:
    season_order = ["Winter", "Summer", "Monsoon", "Autumn"]
    trend = (
        df.groupby("Season")["Visitors_Count"]
        .mean()
        .reset_index()
    )
    trend["Season"] = pd.Categorical(trend["Season"], categories=season_order, ordered=True)
    return trend.sort_values("Season")


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    trend = df.groupby("Month")["Visitors_Count"].mean().reset_index()
    trend["Month"] = pd.Categorical(trend["Month"], categories=month_order, ordered=True)
    return trend.sort_values("Month")


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Location_State")
        .agg(
            Avg_Visitors=("Visitors_Count", "mean"),
            Total_Visitors=("Visitors_Count", "sum"),
            Place_Count=("Place_Name", "nunique"),
            Avg_Rating=("Google_Rating", "mean"),
        )
        .reset_index()
        .sort_values("Total_Visitors", ascending=False)
    )


def get_place_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Place_Type")["Visitors_Count"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg_Visitors", "sum": "Total_Visitors", "count": "Records"})
        .sort_values("Total_Visitors", ascending=False)
    )


def get_weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Weather_Type")["Visitors_Count"]
        .mean()
        .reset_index()
        .sort_values("Visitors_Count", ascending=False)
    )


def get_alternatives(df: pd.DataFrame, state: str, current_place: str, zone: str = None) -> pd.DataFrame:
    query = (df["Location_State"] == state) & (df["Place_Name"] != current_place)
    if zone:
        zone_query = (df["Zone"] == zone) & (df["Place_Name"] != current_place)
        alts_zone = df[zone_query].groupby("Place_Name").agg(
            Avg_Visitors=("Visitors_Count", "mean"),
            Avg_Rating=("Google_Rating", "mean"),
            Place_Type=("Place_Type", "first"),
            Location_State=("Location_State", "first"),
            Zone=("Zone", "first"),
        ).reset_index()
        alts_state = df[query].groupby("Place_Name").agg(
            Avg_Visitors=("Visitors_Count", "mean"),
            Avg_Rating=("Google_Rating", "mean"),
            Place_Type=("Place_Type", "first"),
            Location_State=("Location_State", "first"),
            Zone=("Zone", "first"),
        ).reset_index()
        alts = pd.concat([alts_zone, alts_state]).drop_duplicates(subset="Place_Name")
    else:
        alts = df[query].groupby("Place_Name").agg(
            Avg_Visitors=("Visitors_Count", "mean"),
            Avg_Rating=("Google_Rating", "mean"),
            Place_Type=("Place_Type", "first"),
            Location_State=("Location_State", "first"),
            Zone=("Zone", "first"),
        ).reset_index()

    return alts.sort_values("Avg_Rating", ascending=False).head(5)
