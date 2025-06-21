"""
Web app demo visualizing irrigation and mulch strategies for yield optimization.
"""
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st


def make_clouds(fig, dates, num_curves: int, y_offset: float):
    """
    Graphs some fun clouds on the plotly figure.
    """
    for cloud in range(num_curves):
        x = np.linspace(0, len(dates), len(dates))
        y_base = np.sin(x) + np.cos(x / np.random.uniform(0.5, 2.0)) + y_offset
        noise = np.random.normal(scale=np.random.uniform(0.2, 0.5), size=len(dates))
        y = y_base + noise

        name = "a pretty cloud" if cloud == 0 else ""
        showlegend = cloud == 0
        fig.add_trace(go.Scatter(x=dates,
                                 y=y,
                                 mode="lines",
                                 line={"width": 2, "color": "white"},
                                 opacity=0.7,
                                 name=name,
                                 showlegend=showlegend,
                                 legendrank=0), row=1, col=1)


def get_cand_id_from_filters(data_df: pd.DataFrame, max_irr: int, max_mulch: int, weather_year: int) -> str:
    """
    Filters candidates by irrigation and mulch.
    First subsets candidates by those who mulch less than the threshold.
    Then finds the candidate that irrigates as close as possible to the max threshold without exceeding it.
    """
    # Filter candidates by irrigation and mulch
    year_df = data_df[data_df["year"] == weather_year]

    valid_mulches = year_df[year_df["mulch_pct"] <= max_mulch]["cand_id"].unique().tolist()
    filtered_df = year_df[year_df["cand_id"].isin(valid_mulches)]
    total_irrs = filtered_df.groupby("cand_id")["IrrDay"].sum()
    cand_id = total_irrs[total_irrs <= max_irr].idxmax()
    return cand_id


def plot_plotly(data_df: pd.DataFrame, weather_df: pd.DataFrame, max_irr: int, max_mulch: int, weather_year: int):
    """
    Creates the plotly figure showing a cloud, precipitation, yield, mulch, and irrigation strategy for a specific year
    filtered by the specified irrigation and mulch limits.
    """
    precip_range = (weather_df["Precipitation"].min(), weather_df["Precipitation"].max())

    cand_id = get_cand_id_from_filters(data_df, max_irr, max_mulch, weather_year)
    print(cand_id, weather_year)
    df = data_df[(data_df["year"] == weather_year) & (data_df["cand_id"] == cand_id)]
    weather = weather_df[weather_df["year"] == weather_year]

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.3, 0.7],
                        vertical_spacing=0.05,
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    yield_range = (data_df["DryYield"].min(), data_df["DryYield"].max())
    irr_range = (data_df["IrrDay"].min(), data_df["IrrDay"].max())

    fig.add_trace(go.Scatter(x=weather["Date"], y=weather["Precipitation"], line={"color": "#4FC3F7"}, name="precipitation"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=weather["Date"], y=df["IrrDay"], marker_color="#1976D2", name="irrigation"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=weather["Date"], y=df["DryYield"], line={"color": "#FFC107"}, name="maize yield"),
                  row=2, col=1, secondary_y=True)

    # We scale the mulch percentage to the irrigation range to show on the graph
    mulch_scaled = df['mulch_pct'].max() / 100 * irr_range[1]
    fig.add_trace(go.Scatter(x=weather["Date"], y=[mulch_scaled] * len(weather["Date"]), line={"color": "#66BB6A"}, name=f"{df['mulch_pct'].max():.0f}% mulch"), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=weather["Date"], y=[0] * len(df), line={"color": "#8D6E63"}, name="the ground"),
                  row=2, col=1, secondary_y=True)

    make_clouds(fig, weather["Date"], num_curves=5, y_offset=precip_range[1])

    fig.update_yaxes(title_text="Precipitation (mm)", range=precip_range, row=1, col=1)
    fig.update_yaxes(title_text="Irrigation (mm)", range=irr_range, row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Yield (tonnes/ha)", range=yield_range, row=2, col=1, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def main():
    np.random.seed(42)

    # Streamlit page configuration
    st.title("Yield and Irrigation Plot")
    # For now these do nothing
    region = st.sidebar.selectbox("Select Region", ["Champion, Nebraska", "Tunisia", "..."], index=0)
    crop_type = st.sidebar.selectbox("Crop Type", ["Maize", "Wheat", "Soybean", "..."], index=0)

    # Select the year
    weather_years = {"Oracle": 2018, "Last Year": 2017, "Low Precip": 1984, "High Precip": 2009}
    weather = st.sidebar.selectbox("Weather Prediction", ["Oracle", "Last Year", "Low Precip", "High Precip"], index=0)

    # Read the dataframe
    data_df = pd.read_csv("app/data.csv")
    results_df = pd.read_csv("results/rnn-subset/results.csv")
    results_df = results_df[results_df["gen"] == results_df["gen"].max()]
    results_df = results_df.sort_values(by="irrigation")

    weather_df = pd.read_csv("app/weather.csv")

    # Select a baseline
    total_irrs = data_df.groupby(["year", "cand_id"])["IrrDay"].sum()
    max_irr = st.slider("Max Irrigation",
                        min_value=total_irrs.min(),
                        max_value=total_irrs.max(),
                        value=total_irrs.max(),
                        step=0.1)
    max_mulch = st.slider("Max Mulch",
                          min_value=0.0,
                          max_value=100.0,
                          value=100.0,
                          step=0.1)

    plot_plotly(data_df, weather_df, max_irr, max_mulch, weather_years[weather])


if __name__ == "__main__":
    main()
