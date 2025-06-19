import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

def plot_matplotlib(filtered_df: pd.DataFrame, yield_range: tuple[float, float], irr_range: tuple[float, float]):
    # Plotting
    fig, (ax_weather, ax_main_1) = plt.subplots(2, 1,
                                                figsize=(15, 10),
                                                sharex=True,
                                                gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05})

    # Plot weather data
    precip_color = 'tab:cyan'
    ax_weather.plot(filtered_df["Date"], filtered_df["Precipitation"], color=precip_color)
    ax_weather.set_ylabel("Precipitation (mm)", color=precip_color)
    ax_weather.tick_params(axis='y', labelcolor=precip_color)
    ax_weather.tick_params(axis="x", labelbottom=False)

    # Plot Irrigation
    irr_color = 'tab:blue'
    ax_main_1.set_xlabel('Date')
    ax_main_1.set_ylabel('Irrigation (mm)', color=irr_color)
    ax_main_1.tick_params(axis='y', labelcolor=irr_color)
    ax_main_1.set_ylim(*irr_range)

    ax_main_1.bar(filtered_df['Date'], filtered_df["IrrDay"], color=irr_color)


    # Create a second y-axis for Yield
    ax_main_2 = ax_main_1.twinx()
    yield_color = 'tab:green'
    ax_main_2.set_ylabel('Yield (tonnes/ha)', color=yield_color)
    ax_main_2.tick_params(axis='y', labelcolor=yield_color)
    ax_main_2.set_ylim(*yield_range)

    ax_main_2.plot(filtered_df['Date'], filtered_df["DryYield"], color=yield_color)

    ax_main_1.xaxis.set_major_locator(mdates.MonthLocator())
    ax_main_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig.autofmt_xdate()  # auto-format dates

    st_plot = st.pyplot(fig)


def make_clouds(fig, dates, num_curves: int, y_offset: float):
    for cloud in range(num_curves):
        x = np.linspace(0, len(dates), len(dates))
        y_base = np.sin(x) + np.cos(x / np.random.uniform(0.5, 2.0)) + y_offset
        noise = np.random.normal(scale=np.random.uniform(0.2, 0.5), size=len(dates))
        y = y_base + noise

        name = "a pretty cloud" if cloud == 0 else ""
        showlegend = cloud == 0
        fig.add_trace(go.Scatter(x=dates, y=y, mode="lines", line={"width": 2, "color": "white"}, opacity=0.7, name=name, showlegend=showlegend, legendrank=0),
                      row=1, col=1)


def plot_plotly(df: pd.DataFrame, selected_baseline: int):
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.3, 0.7],
                        vertical_spacing=0.05,
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    yield_range = (df["DryYield"].min(), df["DryYield"].max())
    irr_range = (df["IrrDay"].min(), df["IrrDay"].max())

    # Filter data for the selected baseline
    filtered_df = df[df['baseline'] == selected_baseline].sort_values('Date')

    fig.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["Precipitation"], line={"color": "#4FC3F7"}, name="precipitation"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=filtered_df["Date"], y=filtered_df["IrrDay"], marker_color="#1976D2", name="irrigation"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["DryYield"], line={"color": "#FFC107"}, name="maize yield"),
                  row=2, col=1, secondary_y=True)

    max_df = df[df["baseline"] == 9].sort_values("Date")
    fig.add_trace(go.Scatter(x=max_df["Date"], y=max_df["DryYield"], line={"color": "#FFC107", "dash": "dash"}, opacity=0.7, name="max maize yield"),
                  row=2, col=1, secondary_y=True)

    min_df = df[df["baseline"] == 0].sort_values("Date")
    fig.add_trace(go.Scatter(x=min_df["Date"], y=min_df["DryYield"], line={"color": "#FFC107", "dash": "dash"}, opacity=0.7, name="min maize yield"),
                  row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=filtered_df["Date"], y=[0] * len(filtered_df), line={"color": "#8D6E63"}, name="the ground"),
                  row=2, col=1, secondary_y=True)

    make_clouds(fig, filtered_df["Date"], num_curves=5, y_offset=filtered_df["Precipitation"].max())

    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Irrigation (mm)", range=irr_range, row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Yield (tonnes/ha)", range=yield_range, row=2, col=1, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

def main():
    np.random.seed(42)

    # Streamlit page configuration
    st.title("Yield and Irrigation Plot")
    region = st.sidebar.selectbox("Select Region", ["Champion, Nebraska", "Tunisia", "..."], index=0)
    crop_type = st.sidebar.selectbox("Crop Type", ["Maize", "Wheat", "Soybean", "..."], index=0)
    weather_prediction = st.sidebar.selectbox("Weather Prediction", ["Oracle", "Last Year", "Low Precip", "High Precip"], index=0)

    # Read the dataframe
    df = pd.read_csv("data/one-season-data.csv")
    df = df.fillna(0)

    # Select a baseline
    selected_baseline = st.slider("Select Irrigation Level", min_value=0, max_value=9, value=0, step=1)

    plot_plotly(df, selected_baseline)


if __name__ == "__main__":
    main()
