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
        noise = np.random.normal(scale=np.random.uniform(0, 1), size=len(dates))
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

    if len(valid_mulches) == 0:
        return None

    filtered_df = year_df[year_df["cand_id"].isin(valid_mulches)]

    if filtered_df.empty:
        return None

    total_irrs = filtered_df.groupby("cand_id")["IrrDay"].sum()
    cand_id = total_irrs[total_irrs <= max_irr].idxmax()
    return cand_id


def plot_plotly(data_df: pd.DataFrame, weather_df: pd.DataFrame, cand_id: str, weather_year: int):
    """
    Creates the plotly figure showing a cloud, precipitation, yield, mulch, and irrigation strategy for a specific year
    filtered by the specified irrigation and mulch limits.
    """
    precip_range = (weather_df["Precipitation"].min(), weather_df["Precipitation"].max())

    df = data_df[(data_df["year"] == weather_year) & (data_df["cand_id"] == cand_id)]
    weather = weather_df[weather_df["year"] == weather_year]

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.3, 0.7],
                        vertical_spacing=0.05,
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    yield_range = (data_df["DryYield"].min(), data_df["DryYield"].max())
    irr_range = (data_df["IrrDay"].min(), data_df["IrrDay"].max())

    precip = go.Scatter(
        x=weather["Date"],
        y=weather["Precipitation"],
        line={"color": "#4FC3F7"},
        name="precipitation"
    )
    fig.add_trace(precip, row=1, col=1)
    irrigation = go.Bar(
        x=weather["Date"],
        y=df["IrrDay"],
        marker_color="#1976D2",
        name="irrigation"
    )
    fig.add_trace(irrigation, row=2, col=1, secondary_y=False)
    maize_yield = go.Scatter(
        x=weather["Date"],
        y=df["DryYield"],
        line={"color": "#FFC107"},
        name="yield"
    )
    fig.add_trace(maize_yield, row=2, col=1, secondary_y=True)

    # We scale the mulch percentage to the irrigation range to show on the graph
    mulch_scaled = df['mulch_pct'].max() / 100 * irr_range[1]
    mulch = go.Scatter(
        x=weather["Date"],
        y=[mulch_scaled] * len(weather["Date"]),
        line={"color": "#66BB6A"},
        name=f"{df['mulch_pct'].max():.0f}% mulch"
    )
    fig.add_trace(mulch, row=2, col=1, secondary_y=False)

    ground = go.Scatter(
        x=weather["Date"],
        y=[0] * len(weather["Date"]),
        line={"color": "#8D6E63"},
        name="the ground"
    )
    fig.add_trace(ground, row=2, col=1, secondary_y=True)

    make_clouds(fig, weather["Date"], num_curves=5, y_offset=precip_range[1])

    fig.update_yaxes(title_text="Precipitation (mm)", range=precip_range, row=1, col=1)
    fig.update_yaxes(title_text="Irrigation (mm)", range=irr_range, row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Yield (tonnes/ha)", range=yield_range, row=2, col=1, secondary_y=True)
    fig.update_xaxes(dtick="M1", tickformat="%b")

    st.plotly_chart(fig, use_container_width=True)


def display_strategy(data_df: pd.DataFrame, cand_id: str, weather_year: int):
    """
    Shows the overall irrigation and mulch used.
    """
    df = data_df[(data_df["year"] == weather_year) & (data_df["cand_id"] == cand_id)].copy()
    total_irr = df["IrrDay"].sum()
    mulch_pct = df["mulch_pct"].max()
    final_yield = df["DryYield"].max()
    strategy_df = pd.DataFrame({
        "Candidate ID": [cand_id],
        "Total Irrigation (mm)": [f"{total_irr:.2f}"],
        "Mulch Percentage": [f"{mulch_pct:.0f}%"],
        "Final Yield (tonnes/ha)": [f"{final_yield:.2f}"]
    })
    st.dataframe(strategy_df, hide_index=True)


def display_actions_table(data_df: pd.DataFrame, weather_df: pd.DataFrame, cand_id: str, weather_year: int):
    """
    Gets the days to irrigate and the amount of irrigation for a selected candidate and year.
    """
    df = data_df[(data_df["year"] == weather_year) & (data_df["cand_id"] == cand_id)].copy()
    irr_df = df[df["IrrDay"] > 0]
    if not irr_df.empty:
        st.subheader("Irrigation Timeline")
        start_date = pd.to_datetime(weather_df["Date"]).min()
        irr_df["Date"] = start_date + pd.to_timedelta(irr_df["time_step_counter"], unit="D")
        irr_df["Date"] = irr_df["Date"].dt.strftime("%B %d")
        irr_df["IrrDay"] = irr_df["IrrDay"].round(2)
        st.dataframe(irr_df[["Date", "IrrDay"]].rename(columns={"IrrDay": "Irrigation (mm)"}), hide_index=True)


def main():
    """
    Main logic to run Streamlit app.
    Gets inputs and loads data, then passes it into the plotting function.
    """
    np.random.seed(42)

    # Streamlit page configuration
    st.set_page_config(
        page_title="Irrigation Planner",
        page_icon="🌽",
        menu_items={
            "Get help": "mailto:daniel.young2@cognizant.com",
            "Report a bug": None,
            "About": """
            This app visualizes irrigation and mulch strategies for yield optimization. Optimization is done using NeuroAI from the Cognizant AI Lab. This project was done under Project Resilience: a nonprofit collaboration with the United Nations.
            """
        }
    )

    st.title("Yield Optimization")

    st.sidebar.markdown(
        """
        Select a context situation, then specify the max irrigation and mulch to use.
        """
    )

    # For now these do nothing
    _ = st.sidebar.selectbox("Select Region", ["Champion, Nebraska", "Tunisia", "..."], index=0)
    crop_select = st.sidebar.selectbox("Crop Type", ["Maize", "Potato", "..."], index=0)
    crop_path_map = {"Maize": "app/maize-data.csv", "Potato": "app/potato-data.csv"}

    # Select the year
    weather_years = {"Oracle": 2018, "Last Year": 2017, "Low Precip": 1984, "High Precip": 2009}
    weather = st.sidebar.selectbox("Weather Prediction", ["Oracle", "Last Year", "Low Precip", "High Precip"], index=0)

    # Read the dataframe
    data_df = pd.read_csv(crop_path_map[crop_select])

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
    weather_year = weather_years[weather]
    cand_id = get_cand_id_from_filters(data_df, max_irr, max_mulch, weather_year)
    if cand_id is None:
        st.error("No candidates found for selected irrigation and mulch limits.")
    else:
        plot_plotly(data_df, weather_df, cand_id, weather_year)
        display_strategy(data_df, cand_id, weather_year)
        display_actions_table(data_df, weather_df, cand_id, weather_year)


if __name__ == "__main__":
    main()
