import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import (
    load_raw_data, get_summary_stats, get_top_places,
    get_season_trend, get_monthly_trend, get_state_summary,
    get_place_type_distribution, get_weather_impact, get_alternatives
)
from utils.model import train_models, models_exist
from utils.predictor import (
    get_models_and_encoders, build_input_row,
    predict_visitors, predict_overcrowding, classify_visitors_manual
)
from utils.auth import register_user, login_user

st.set_page_config(
    page_title="Tourism AI - Demand & Overcrowding Intelligence",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C9A7, #6C63FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1A1D2E, #252838);
        border: 1px solid #2D3149;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00C9A7;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #9CA3AF;
        margin-top: 0.3rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #052e16, #064e3b);
        border: 1px solid #10B981;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #6EE7B7;
    }
    .risk-medium {
        background: linear-gradient(135deg, #451a03, #78350f);
        border: 1px solid #F59E0B;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #FCD34D;
    }
    .risk-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #FCA5A5;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E5E7EB;
        border-left: 4px solid #00C9A7;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .alert-box {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .alt-card {
        background: linear-gradient(135deg, #052e16, #064e3b);
        border: 1px solid #10B981;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #D1D5DB !important;
    }
    div[data-testid="stMetricValue"] {
        color: #00C9A7 !important;
        font-size: 1.5rem !important;
    }
    .auth-wrapper {
        max-width: 460px;
        margin: 3rem auto;
        padding: 2.5rem 2rem;
        background: linear-gradient(135deg, #1A1D2E, #252838);
        border: 1px solid #2D3149;
        border-radius: 18px;
    }
    .auth-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .auth-logo-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00C9A7, #6C63FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .auth-logo-sub {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .auth-tab-active {
        color: #00C9A7 !important;
        border-bottom: 2px solid #00C9A7 !important;
        font-weight: 700 !important;
    }
    .user-badge {
        background: linear-gradient(135deg, #1A1D2E, #252838);
        border: 1px solid #2D3149;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    return load_raw_data()


@st.cache_resource(show_spinner=False)
def load_trained_models():
    return get_models_and_encoders()


def render_risk_badge(level: str) -> str:
    if level == "Low":
        return f'<div class="risk-low">🟢 Low Risk — {level} Overcrowding</div>'
    elif level == "Medium":
        return f'<div class="risk-medium">🟡 Medium Risk — {level} Overcrowding</div>'
    else:
        return f'<div class="risk-high">🔴 High Risk — {level} Overcrowding</div>'


def page_dashboard():
    st.markdown('<div class="main-header">Tourism Demand Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-driven visitor forecasting and overcrowding risk assessment for Indian tourism destinations.</div>', unsafe_allow_html=True)

    df = load_data()
    stats = get_summary_stats(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with c2:
        st.metric("Unique Destinations", f"{stats['total_places']:,}")
    with c3:
        st.metric("States Covered", f"{stats['total_states']:,}")
    with c4:
        st.metric("Avg Visitors / Day", f"{int(stats['avg_visitors']):,}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Total Visitors", f"{stats['total_visitors']:,}")
    with c6:
        st.metric("Avg Google Rating", f"{stats['avg_rating']} ⭐")
    with c7:
        st.metric("Total Revenue (₹)", f"{stats['total_revenue']:,}")
    with c8:
        st.metric("Place Categories", f"{stats['place_types']}")

    st.markdown('<div class="section-title">Visitor Trends by Season & Month</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        season_data = get_season_trend(df)
        fig_season = px.bar(
            season_data, x="Season", y="Visitors_Count",
            color="Visitors_Count",
            color_continuous_scale=["#6C63FF", "#00C9A7"],
            title="Average Visitors by Season",
            labels={"Visitors_Count": "Avg Visitors"}
        )
        fig_season.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB", showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_season, use_container_width=True)

    with col2:
        monthly = get_monthly_trend(df)
        fig_monthly = px.line(
            monthly, x="Month", y="Visitors_Count",
            title="Monthly Visitor Trend",
            markers=True,
            labels={"Visitors_Count": "Avg Visitors"},
            color_discrete_sequence=["#00C9A7"]
        )
        fig_monthly.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 Destinations by Average Visitors</div>', unsafe_allow_html=True)
    top_places = get_top_places(df, 10)
    fig_top = px.bar(
        top_places.sort_values("Visitors_Count"),
        x="Visitors_Count", y="Place_Name",
        orientation="h",
        color="Visitors_Count",
        color_continuous_scale=["#6C63FF", "#00C9A7"],
        title="",
        labels={"Visitors_Count": "Avg Visitors", "Place_Name": "Destination"}
    )
    fig_top.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB", height=400, coloraxis_showscale=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('<div class="section-title">Overcrowding Level Distribution</div>', unsafe_allow_html=True)
    from utils.preprocessing import clean_data, add_overcrowding_label
    df_clean = clean_data(df)
    df_label, l, u = add_overcrowding_label(df_clean)
    oc_dist = df_label["Overcrowding_Level"].value_counts().reset_index()
    oc_dist.columns = ["Level", "Count"]
    color_map = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
    fig_oc = px.pie(
        oc_dist, names="Level", values="Count",
        color="Level", color_discrete_map=color_map,
        title="Overcrowding Level Distribution"
    )
    fig_oc.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E5E7EB"
    )
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.plotly_chart(fig_oc, use_container_width=True)
    with col_b:
        st.markdown(f"""
        **Overcrowding Thresholds (Percentile-Based)**  
        - 🟢 **Low**: Visitors ≤ {int(l):,}  
        - 🟡 **Medium**: {int(l):,} < Visitors ≤ {int(u):,}  
        - 🔴 **High**: Visitors > {int(u):,}

        These thresholds are computed dynamically from the 33rd and 66th percentiles of the overall dataset's `Visitors_Count`.
        """)
        weather_data = get_weather_impact(df)
        fig_weather = px.bar(
            weather_data, x="Weather_Type", y="Visitors_Count",
            color="Visitors_Count",
            color_continuous_scale=["#6C63FF", "#00C9A7"],
            title="Impact of Weather on Visitors",
            labels={"Visitors_Count": "Avg Visitors", "Weather_Type": "Weather"},
        )
        fig_weather.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB", coloraxis_showscale=False
        )
        st.plotly_chart(fig_weather, use_container_width=True)


def page_dataset():
    st.markdown('<div class="main-header">Dataset Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore raw and processed tourism data.</div>', unsafe_allow_html=True)

    df = load_data()

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Preprocessed Data", "Column Statistics"])

    with tab1:
        st.markdown(f"**{len(df):,} records** | {df.shape[1]} columns")
        filters = {}
        col1, col2, col3 = st.columns(3)
        with col1:
            states = ["All"] + sorted(df["Location_State"].dropna().unique().tolist())
            sel_state = st.selectbox("Filter by State", states, key="ds_state")
            if sel_state != "All":
                filters["Location_State"] = sel_state
        with col2:
            seasons = ["All"] + sorted(df["Season"].dropna().unique().tolist())
            sel_season = st.selectbox("Filter by Season", seasons, key="ds_season")
            if sel_season != "All":
                filters["Season"] = sel_season
        with col3:
            types = ["All"] + sorted(df["Place_Type"].dropna().unique().tolist())
            sel_type = st.selectbox("Filter by Place Type", types, key="ds_type")
            if sel_type != "All":
                filters["Place_Type"] = sel_type

        filtered = df.copy()
        for col, val in filters.items():
            filtered = filtered[filtered[col] == val]

        st.dataframe(filtered, use_container_width=True, height=450)
        st.caption(f"Showing {len(filtered):,} records")

    with tab2:
        from utils.preprocessing import clean_data, add_overcrowding_label
        df_clean = clean_data(df)
        df_label, l, u = add_overcrowding_label(df_clean)
        st.markdown(f"**Preprocessed: {len(df_label):,} records** with `Overcrowding_Level` column")
        st.dataframe(df_label[[
            "Date", "Place_Name", "Location_State", "Season", "Month",
            "Weather_Type", "Visitors_Count", "Overcrowding_Level",
            "Google_Rating", "Is_Weekend", "Airport_Within_50km"
        ]], use_container_width=True, height=450)

    with tab3:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown("**Numerical Column Statistics**")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)
        st.markdown("**Missing Values**")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(missing[missing["Missing Count"] > 0], use_container_width=True)


def _get_smart_defaults(df: pd.DataFrame, location_state: str, place_type: str, season: str) -> dict:
    season_month_map = {
        "Winter": "January", "Summer": "May",
        "Monsoon": "July", "Autumn": "October"
    }
    month = season_month_map.get(season, "January")
    zone_row = df[df["Location_State"] == location_state]["Zone"].mode()
    zone = zone_row.iloc[0] if not zone_row.empty else "Northern"
    sig_row = df[df["Place_Type"] == place_type]["Significance"].mode()
    significance = sig_row.iloc[0] if not sig_row.empty else "Historical"
    review_row = df[df["Place_Type"] == place_type]["Review_Count_Lakhs"]
    review_count = float(review_row.median()) if not review_row.empty else 0.2
    ticket_row = df[df["Place_Type"] == place_type]["Ticket_Price"]
    ticket_price = float(ticket_row.median()) if not ticket_row.empty else 0.0
    airport_row = df[df["Location_State"] == location_state]["Airport_Within_50km"].mode()
    airport = "Yes" if (not airport_row.empty and airport_row.iloc[0] in ["Yes", 1]) else "No"
    return {
        "month": month, "zone": zone, "significance": significance,
        "review_count": review_count, "ticket_price": ticket_price,
        "airport_50km": airport
    }


def page_forecasting():
    st.markdown('<div class="main-header">Visitor Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter 6 key details to predict expected tourist footfall using the AI model.</div>', unsafe_allow_html=True)

    df = load_raw_data()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Model Performance**")
        if models_exist():
            try:
                from utils.model import load_models
                models_data = load_models()
                dm = models_data["demand"]["metrics"]
                st.success(f"**{dm['model_name']}**")
                st.metric("R² Score", f"{dm['r2']:.4f}")
                st.metric("RMSE", f"{dm['rmse']:.2f}")
                st.caption("Gradient Boosting · 28 features · 3MB model")
            except Exception:
                pass

    st.markdown('<div class="section-title">Enter Destination Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1A1D2E;border:1px solid #2D3149;border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1.2rem;color:#9CA3AF;font-size:0.88rem;">
        Fill in the 6 key inputs below. Remaining parameters are automatically set using smart dataset-based defaults.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        location_state = st.selectbox(
            "📍 State / Location",
            sorted(df["Location_State"].dropna().unique().tolist()),
            help="The Indian state where the destination is located."
        )
        season = st.selectbox(
            "🌤️ Season",
            ["Winter", "Summer", "Monsoon", "Autumn"],
            help="Season of the planned visit."
        )
        place_type = st.selectbox(
            "🏛️ Place Type",
            sorted(df["Place_Type"].dropna().unique().tolist()),
            help="Category of the destination (e.g. Temple, Beach, Fort)."
        )

    with col2:
        weather_type = st.selectbox(
            "🌦️ Expected Weather",
            sorted(df["Weather_Type"].dropna().unique().tolist()),
            help="Expected weather conditions during the visit."
        )
        visit_day_type = st.radio(
            "📅 Visit Day",
            ["Weekday", "Weekend"],
            horizontal=True,
            help="Are you visiting on a weekday or weekend?"
        )
        google_rating = st.slider(
            "⭐ Google Rating of Place",
            min_value=1.0, max_value=5.0, value=4.3, step=0.1,
            help="The Google rating of the destination (1–5)."
        )

    is_weekend = 1 if visit_day_type == "Weekend" else 0
    day_of_week = "Saturday" if is_weekend else "Wednesday"

    defaults = _get_smart_defaults(df, location_state, place_type, season)

    with st.expander("🔧 Auto-filled Parameters (click to see defaults used)"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Month", defaults["month"])
        d2.metric("Zone", defaults["zone"])
        d3.metric("Significance", defaults["significance"])
        d4.metric("Review Count (L)", f"{defaults['review_count']:.2f}")
        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Ticket Price (₹)", f"{int(defaults['ticket_price'])}")
        d6.metric("Airport ≤ 50km", defaults["airport_50km"])
        d7.metric("Tourist Type", "Domestic")
        d8.metric("Est. Year", "1990")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Visitor Count", type="primary", use_container_width=True):
        with st.spinner("Running prediction model..."):
            try:
                models_res, encoders, lower_thresh, upper_thresh, agg_stats = load_trained_models()
                input_df = build_input_row(
                    month=defaults["month"],
                    year=2025,
                    season=season,
                    day_of_week=day_of_week,
                    is_weekend=is_weekend,
                    weather_type=weather_type,
                    location_state=location_state,
                    zone=defaults["zone"],
                    place_type=place_type,
                    significance=defaults["significance"],
                    tourist_type="Domestic",
                    establishment_year=1990.0,
                    google_rating=float(google_rating),
                    review_count_lakhs=defaults["review_count"],
                    ticket_price=defaults["ticket_price"],
                    airport_within_50km=1 if defaults["airport_50km"] == "Yes" else 0,
                    dslr_allowed=1,
                    weekly_off="None"
                )
                predicted = predict_visitors(input_df, models_res, encoders, agg_stats)
                risk_level = classify_visitors_manual(predicted, lower_thresh, upper_thresh)

                st.markdown("---")
                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Predicted Visitors", f"{int(predicted):,}")
                with c2:
                    st.metric("Low Threshold", f"≤ {int(lower_thresh):,}")
                with c3:
                    st.metric("High Threshold", f"> {int(upper_thresh):,}")

                st.markdown(render_risk_badge(risk_level), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                if risk_level == "High":
                    st.warning("High visitor demand detected. This destination may face overcrowding. Consider visiting on weekdays or during off-season.")
                elif risk_level == "Medium":
                    st.info("Moderate demand expected. Plan ahead and book tickets in advance.")
                else:
                    st.success("Low demand expected. Great time to visit with minimal crowds.")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)


def page_overcrowding():
    st.markdown('<div class="main-header">Overcrowding Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter 6 key details to classify overcrowding risk (Low / Medium / High) using the AI model.</div>', unsafe_allow_html=True)

    df = load_raw_data()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Classifier Performance**")
        if models_exist():
            try:
                from utils.model import load_models
                models_data = load_models()
                cm_data = models_data["crowd"]
                st.success(f"**{cm_data['metrics']['model_name']}**")
                st.metric("Accuracy", f"{cm_data['metrics']['accuracy']:.4f}")
                st.caption("3-class: Low / Medium / High · Gradient Boosting")
            except Exception:
                pass

    st.markdown('<div class="section-title">Enter Destination Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1A1D2E;border:1px solid #2D3149;border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1.2rem;color:#9CA3AF;font-size:0.88rem;">
        Fill in the 6 key inputs below. Smart defaults from the dataset are used for all other parameters automatically.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        location_state = st.selectbox(
            "📍 State / Location",
            sorted(df["Location_State"].dropna().unique().tolist()),
            key="oc_state",
            help="The Indian state where the destination is located."
        )
        season = st.selectbox(
            "🌤️ Season",
            ["Winter", "Summer", "Monsoon", "Autumn"],
            key="oc_season",
            help="Season of the planned visit."
        )
        place_type = st.selectbox(
            "🏛️ Place Type",
            sorted(df["Place_Type"].dropna().unique().tolist()),
            key="oc_pt",
            help="Category of the destination (e.g. Temple, Beach, Fort)."
        )

    with col2:
        weather_type = st.selectbox(
            "🌦️ Expected Weather",
            sorted(df["Weather_Type"].dropna().unique().tolist()),
            key="oc_wt",
            help="Expected weather conditions during the visit."
        )
        visit_day_type = st.radio(
            "📅 Visit Day",
            ["Weekday", "Weekend"],
            horizontal=True,
            key="oc_daytype",
            help="Are you visiting on a weekday or weekend?"
        )
        google_rating = st.slider(
            "⭐ Google Rating of Place",
            min_value=1.0, max_value=5.0, value=4.3, step=0.1,
            key="oc_gr",
            help="The Google rating of the destination (1–5)."
        )

    is_weekend = 1 if visit_day_type == "Weekend" else 0
    day_of_week = "Saturday" if is_weekend else "Wednesday"

    defaults = _get_smart_defaults(df, location_state, place_type, season)

    with st.expander("🔧 Auto-filled Parameters (click to see defaults used)"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Month", defaults["month"])
        d2.metric("Zone", defaults["zone"])
        d3.metric("Significance", defaults["significance"])
        d4.metric("Review Count (L)", f"{defaults['review_count']:.2f}")
        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Ticket Price (₹)", f"{int(defaults['ticket_price'])}")
        d6.metric("Airport ≤ 50km", defaults["airport_50km"])
        d7.metric("Tourist Type", "Domestic")
        d8.metric("Est. Year", "1990")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Overcrowding Risk", type="primary", use_container_width=True):
        with st.spinner("Classifying overcrowding risk..."):
            try:
                models_res, encoders, lower_thresh, upper_thresh, agg_stats = load_trained_models()
                input_df = build_input_row(
                    month=defaults["month"],
                    year=2025,
                    season=season,
                    day_of_week=day_of_week,
                    is_weekend=is_weekend,
                    weather_type=weather_type,
                    location_state=location_state,
                    zone=defaults["zone"],
                    place_type=place_type,
                    significance=defaults["significance"],
                    tourist_type="Domestic",
                    establishment_year=1990.0,
                    google_rating=float(google_rating),
                    review_count_lakhs=defaults["review_count"],
                    ticket_price=defaults["ticket_price"],
                    airport_within_50km=1 if defaults["airport_50km"] == "Yes" else 0,
                    dslr_allowed=1,
                    weekly_off="None"
                )
                pred_label, proba_dict = predict_overcrowding(input_df, models_res, encoders, agg_stats)

                st.markdown("---")
                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
                st.markdown(render_risk_badge(pred_label), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                if proba_dict:
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.markdown("**Confidence Breakdown**")
                        prob_df = pd.DataFrame([proba_dict]).T.reset_index()
                        prob_df.columns = ["Risk Level", "Probability"]
                        color_map = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
                        fig = px.bar(
                            prob_df, x="Risk Level", y="Probability",
                            color="Risk Level",
                            color_discrete_map=color_map,
                            title="Prediction Confidence",
                            text=prob_df["Probability"].apply(lambda x: f"{x:.1%}")
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#E5E7EB", showlegend=False,
                            yaxis=dict(tickformat=".0%", range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with res_col2:
                        st.markdown("**Confusion Matrix**")
                        from utils.model import load_models
                        crowd_data = load_models()["crowd"]
                        cm = crowd_data["confusion_matrix"]
                        crowd_enc = crowd_data["crowd_encoder"]
                        labels = crowd_enc.classes_
                        fig_cm = px.imshow(
                            cm, x=labels, y=labels,
                            color_continuous_scale="Teal",
                            labels=dict(x="Predicted", y="Actual"),
                            title="Model Confusion Matrix",
                            text_auto=True
                        )
                        fig_cm.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#E5E7EB"
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                if pred_label == "High":
                    st.error("High overcrowding risk detected! Visitor numbers are expected to be well above average. Consider visiting during off-season or on weekdays.")
                elif pred_label == "Medium":
                    st.warning("Moderate overcrowding risk. Expect above-average visitor numbers. Book tickets in advance.")
                else:
                    st.success("Low overcrowding risk. A comfortable and peaceful visit is expected.")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)


def page_insights():
    st.markdown('<div class="main-header">Insights & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into tourism patterns, state-wise analysis, and destination intelligence.</div>', unsafe_allow_html=True)

    df = load_raw_data()

    tab1, tab2, tab3, tab4 = st.tabs(["State Analysis", "Place Type Analysis", "Correlation", "Heatmap"])

    with tab1:
        state_data = get_state_summary(df)
        st.markdown("**Top States by Total Visitors**")
        fig_state = px.bar(
            state_data.head(15).sort_values("Total_Visitors"),
            x="Total_Visitors", y="Location_State",
            orientation="h",
            color="Avg_Rating",
            color_continuous_scale="Teal",
            title="Top 15 States — Total Visitors",
            labels={"Total_Visitors": "Total Visitors", "Location_State": "State", "Avg_Rating": "Avg Rating"}
        )
        fig_state.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB", height=500
        )
        st.plotly_chart(fig_state, use_container_width=True)

        st.dataframe(state_data.head(20).style.format({
            "Avg_Visitors": "{:.0f}",
            "Total_Visitors": "{:,}",
            "Avg_Rating": "{:.2f}"
        }), use_container_width=True)

    with tab2:
        pt_data = get_place_type_distribution(df)
        col1, col2 = st.columns(2)
        with col1:
            fig_pt = px.bar(
                pt_data.head(15).sort_values("Total_Visitors"),
                x="Total_Visitors", y="Place_Type",
                orientation="h",
                color="Avg_Visitors",
                color_continuous_scale=["#6C63FF", "#00C9A7"],
                title="Place Types by Total Visitors",
                labels={"Total_Visitors": "Total", "Place_Type": "Type"}
            )
            fig_pt.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E5E7EB", coloraxis_showscale=False
            )
            st.plotly_chart(fig_pt, use_container_width=True)
        with col2:
            fig_donut = px.pie(
                pt_data.head(10), names="Place_Type", values="Records",
                title="Distribution of Records by Place Type",
                hole=0.4
            )
            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E5E7EB"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with tab3:
        numeric_cols = ["Visitors_Count", "Google_Rating", "Review_Count_Lakhs",
                        "Ticket_Price", "Revenue"]
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Correlation Matrix — Key Numerical Features",
            text_auto=".2f"
        )
        fig_corr.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                df.sample(min(2000, len(df)), random_state=42),
                x="Google_Rating", y="Visitors_Count",
                color="Season",
                title="Google Rating vs Visitors Count",
                opacity=0.6,
                labels={"Visitors_Count": "Visitors", "Google_Rating": "Rating"}
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E5E7EB"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            fig_box = px.box(
                df, x="Season", y="Visitors_Count",
                color="Season",
                title="Visitor Distribution by Season",
                labels={"Visitors_Count": "Visitors"}
            )
            fig_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E5E7EB", showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.markdown("**Visitors Heatmap: State × Season**")
        heat_data = df.groupby(["Location_State", "Season"])["Visitors_Count"].mean().reset_index()
        heat_pivot = heat_data.pivot(index="Location_State", columns="Season", values="Visitors_Count").fillna(0)

        season_order = ["Winter", "Summer", "Monsoon", "Autumn"]
        heat_pivot = heat_pivot.reindex(columns=[c for c in season_order if c in heat_pivot.columns])

        top_states = df.groupby("Location_State")["Visitors_Count"].mean().nlargest(20).index
        heat_pivot = heat_pivot.loc[heat_pivot.index.isin(top_states)]

        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="Viridis",
            title="Average Visitors by State and Season (Top 20 States)",
            labels={"color": "Avg Visitors"}
        )
        fig_heat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB", height=600
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Visitors Heatmap: Place Type × Season**")
        heat_pt = df.groupby(["Place_Type", "Season"])["Visitors_Count"].mean().reset_index()
        heat_pt_pivot = heat_pt.pivot(index="Place_Type", columns="Season", values="Visitors_Count").fillna(0)
        heat_pt_pivot = heat_pt_pivot.reindex(columns=[c for c in season_order if c in heat_pt_pivot.columns])
        fig_heat_pt = px.imshow(
            heat_pt_pivot,
            color_continuous_scale="Teal",
            title="Average Visitors by Place Type and Season",
            labels={"color": "Avg Visitors"}
        )
        fig_heat_pt.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB", height=500
        )
        st.plotly_chart(fig_heat_pt, use_container_width=True)


def page_alerts():
    st.markdown('<div class="main-header">Alerts & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify overcrowded destinations and discover less-crowded alternatives nearby.</div>', unsafe_allow_html=True)

    df = load_raw_data()
    from utils.preprocessing import clean_data, add_overcrowding_label
    df_clean = clean_data(df)
    df_label, lower_thresh, upper_thresh = add_overcrowding_label(df_clean)

    high_risk = df_label[df_label["Overcrowding_Level"] == "High"].copy()
    medium_risk = df_label[df_label["Overcrowding_Level"] == "Medium"].copy()

    st.markdown('<div class="section-title">High-Risk Destinations (Current Dataset)</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1A1D2E, #252838); border: 1px solid #EF4444; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <h4 style="color:#FCA5A5; margin:0">⚠️ Overcrowding Alert Summary</h4>
        <p style="color:#9CA3AF; margin-top: 0.5rem;">
            🔴 <b>{high_risk['Place_Name'].nunique()} destinations</b> have been identified as <b>High Risk</b> (Visitors > {int(upper_thresh):,})<br>
            🟡 <b>{medium_risk['Place_Name'].nunique()} destinations</b> have <b>Medium Risk</b> (Visitors between {int(lower_thresh):,} and {int(upper_thresh):,})<br>
            Data-driven risk classification using percentile-based thresholds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["High Risk Destinations", "Medium Risk Destinations", "Find Alternatives"])

    with tab1:
        top_high = (
            high_risk.groupby(["Place_Name", "Location_State", "Zone", "Place_Type"])
            .agg(
                Avg_Visitors=("Visitors_Count", "mean"),
                Max_Visitors=("Visitors_Count", "max"),
                Avg_Rating=("Google_Rating", "mean"),
                Records=("Visitors_Count", "count")
            )
            .reset_index()
            .sort_values("Avg_Visitors", ascending=False)
        )

        for _, row in top_high.head(10).iterrows():
            with st.container():
                st.markdown(f"""
                <div class="alert-box">
                    <b style="color:#FCA5A5; font-size:1.1rem">🔴 {row['Place_Name']}</b>
                    <span style="color:#9CA3AF; font-size:0.85rem;"> — {row['Location_State']}, {row['Zone']}</span><br>
                    <span style="color:#E5E7EB">Type: <b>{row['Place_Type']}</b></span> &nbsp;|&nbsp;
                    <span style="color:#E5E7EB">Avg Visitors: <b>{int(row['Avg_Visitors']):,}</b></span> &nbsp;|&nbsp;
                    <span style="color:#E5E7EB">Peak: <b>{int(row['Max_Visitors']):,}</b></span> &nbsp;|&nbsp;
                    <span style="color:#E5E7EB">Rating: <b>⭐ {row['Avg_Rating']:.1f}</b></span>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        top_medium = (
            medium_risk.groupby(["Place_Name", "Location_State", "Zone", "Place_Type"])
            .agg(
                Avg_Visitors=("Visitors_Count", "mean"),
                Avg_Rating=("Google_Rating", "mean"),
            )
            .reset_index()
            .sort_values("Avg_Visitors", ascending=False)
        )
        st.dataframe(top_medium.head(20).style.format({
            "Avg_Visitors": "{:.0f}",
            "Avg_Rating": "{:.2f}"
        }), use_container_width=True)

    with tab3:
        st.markdown("**Search for Alternative Destinations**")
        st.markdown("Select a high-risk destination to discover recommended alternatives in the same state or zone.")

        col1, col2 = st.columns(2)
        with col1:
            selected_state = st.selectbox(
                "Select State",
                sorted(df["Location_State"].dropna().unique().tolist()),
                key="alt_state"
            )
        with col2:
            places_in_state = df[df["Location_State"] == selected_state]["Place_Name"].dropna().unique().tolist()
            selected_place = st.selectbox(
                "Select Overcrowded Destination",
                sorted(places_in_state),
                key="alt_place"
            )

        zone_val = df[df["Place_Name"] == selected_place]["Zone"].values[0] if selected_place else None

        if st.button("🔎 Find Alternatives", type="primary"):
            alternatives = get_alternatives(df_label, selected_state, selected_place, zone_val)
            if alternatives.empty:
                st.info("No alternatives found in this state. Try a different state.")
            else:
                st.markdown(f"**Top Alternatives to {selected_place}** (same state/zone, lower crowd, high rating):")
                for _, row in alternatives.iterrows():
                    oc_info = df_label[df_label["Place_Name"] == row["Place_Name"]]["Overcrowding_Level"].value_counts()
                    dominant_oc = oc_info.index[0] if len(oc_info) > 0 else "Unknown"
                    badge_color = "#10B981" if dominant_oc == "Low" else ("#F59E0B" if dominant_oc == "Medium" else "#EF4444")
                    st.markdown(f"""
                    <div class="alt-card">
                        <b style="color:#6EE7B7; font-size:1.05rem">✅ {row['Place_Name']}</b>
                        <span style="color:#9CA3AF; font-size:0.85rem;"> — {row['Location_State']}, {row['Zone']}</span><br>
                        Type: <b>{row['Place_Type']}</b> &nbsp;|&nbsp;
                        Avg Visitors: <b>{int(row['Avg_Visitors']):,}</b> &nbsp;|&nbsp;
                        Rating: <b>⭐ {row['Avg_Rating']:.1f}</b> &nbsp;|&nbsp;
                        Risk: <b style="color:{badge_color}">{dominant_oc}</b>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**General Recommendations for Overcrowded Destinations**")
        recommendations = [
            ("🕐 Visit during off-peak hours", "Early mornings (6-9 AM) or late evenings (5-7 PM) are significantly less crowded."),
            ("📅 Choose weekdays over weekends", "Saturday and Sunday typically see 30-50% more visitors at popular sites."),
            ("☁️ Prefer non-peak seasons", "Monsoon and Autumn seasons generally attract fewer visitors than Summer."),
            ("🎟️ Book tickets in advance", "Pre-booked entry reduces wait times and helps with crowd management."),
            ("🗺️ Explore nearby alternatives", "Use the 'Find Alternatives' tool above to discover less-crowded gems in the same region."),
            ("🚗 Consider travel timing", "Arriving early reduces traffic congestion and parking challenges near high-demand sites."),
        ]
        cols = st.columns(2)
        for i, (title, desc) in enumerate(recommendations):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: #1A1D2E; border: 1px solid #2D3149; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem;">
                    <b style="color:#00C9A7">{title}</b><br>
                    <span style="color:#9CA3AF; font-size:0.9rem">{desc}</span>
                </div>
                """, unsafe_allow_html=True)


def page_auth():
    st.markdown("""
    <div style="display:flex; justify-content:center; align-items:center; min-height:10vh; padding-top: 2rem;">
        <div style="text-align:center;">
            <div style="font-size:2.4rem;font-weight:800;background:linear-gradient(90deg,#00C9A7,#6C63FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🗺️ Tourism AI
            </div>
            <div style="color:#9CA3AF;font-size:1rem;margin-top:0.3rem;">
                Demand Forecasting & Overcrowding Intelligence
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    auth_mode = st.session_state.get("auth_mode", "login")

    col_l, col_c, col_r = st.columns([1, 1.6, 1])
    with col_c:
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            if st.button("Login", use_container_width=True,
                         type="primary" if auth_mode == "login" else "secondary"):
                st.session_state["auth_mode"] = "login"
                st.rerun()
        with tab_col2:
            if st.button("Sign Up", use_container_width=True,
                         type="primary" if auth_mode == "signup" else "secondary"):
                st.session_state["auth_mode"] = "signup"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if auth_mode == "login":
            st.markdown("""
            <div style="color:#E5E7EB;font-size:1.1rem;font-weight:600;margin-bottom:1rem;
                border-left:4px solid #00C9A7;padding-left:0.7rem;">
                Welcome Back
            </div>""", unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

                if submitted:
                    success, msg, user_info = login_user(username, password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user_info
                        st.success(f"Welcome back, {user_info['username']}!")
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("""
            <div style="text-align:center;color:#6B7280;font-size:0.85rem;margin-top:1rem;">
                Don't have an account? Click <b>Sign Up</b> above.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="color:#E5E7EB;font-size:1.1rem;font-weight:600;margin-bottom:1rem;
                border-left:4px solid #6C63FF;padding-left:0.7rem;">
                Create Account
            </div>""", unsafe_allow_html=True)

            with st.form("signup_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="Your full name (optional)")
                username = st.text_input("Username *", placeholder="Choose a username (min 3 chars)")
                email = st.text_input("Email *", placeholder="your@email.com")
                password = st.text_input("Password *", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Repeat your password")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if submitted:
                    if password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = register_user(username, email, password, full_name)
                        if success:
                            _, _, user_info = login_user(username, password)
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_info
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("""
            <div style="text-align:center;color:#6B7280;font-size:0.85rem;margin-top:1rem;">
                Already have an account? Click <b>Login</b> above.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;color:#374151;font-size:0.75rem;margin-top:2rem;border-top:1px solid #1F2937;padding-top:1rem;">
            Indian Tourism Dataset &nbsp;·&nbsp; AI-Powered Intelligence &nbsp;·&nbsp; 20K+ Records
        </div>
        """, unsafe_allow_html=True)


def main():
    if not st.session_state.get("authenticated", False):
        page_auth()
        return

    user = st.session_state.get("user", {})

    with st.sidebar:
        st.markdown("## 🗺️ Tourism AI")
        st.markdown("**Demand Forecasting & Overcrowding Prediction**")
        st.markdown("---")

        st.markdown(f"""
        <div class="user-badge">
            <div style="color:#00C9A7;font-weight:600;font-size:0.9rem;">
                👤 {user.get('username', 'User')}
            </div>
            <div style="color:#6B7280;font-size:0.75rem;margin-top:0.2rem;">
                {user.get('email', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            [
                "🏠 Dashboard",
                "📊 Dataset Viewer",
                "📈 Visitor Forecasting",
                "👥 Overcrowding Prediction",
                "💡 Insights & Analytics",
                "⚠️ Alerts & Recommendations"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Model Training**")
        if not models_exist():
            st.warning("Models not trained yet.")
        else:
            st.success("Models ready")

        if st.button("🔄 Train / Retrain Models", use_container_width=True):
            with st.spinner("Training models... this may take a moment."):
                try:
                    train_models(force_retrain=True)
                    st.cache_resource.clear()
                    st.success("Models trained!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = {}
            st.session_state["auth_mode"] = "login"
            st.rerun()

        st.markdown("""
        <div style="color: #6B7280; font-size: 0.75rem; margin-top: 0.5rem;">
            Dataset: Indian Tourism (20K records)<br>
            Model: Random Forest<br>
            Thresholds: Percentile-based
        </div>
        """, unsafe_allow_html=True)

    if not models_exist():
        with st.spinner("Training AI models for the first time... please wait."):
            try:
                train_models(force_retrain=False)
                st.rerun()
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")
                st.stop()

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "📊 Dataset Viewer":
        page_dataset()
    elif page == "📈 Visitor Forecasting":
        page_forecasting()
    elif page == "👥 Overcrowding Prediction":
        page_overcrowding()
    elif page == "💡 Insights & Analytics":
        page_insights()
    elif page == "⚠️ Alerts & Recommendations":
        page_alerts()


if __name__ == "__main__":
    main()
