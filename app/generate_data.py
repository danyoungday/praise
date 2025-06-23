"""
Takes models and pre-generates time series data to be used in the app.
"""
import pandas as pd
from tqdm import tqdm
import yaml

from evaluator import AquaCropEvaluator
from prescriptor import RNNPrescriptorFactory


def generate_data():
    """
    Takes in a results_dir and outputs a data.csv file with the full outputs of the candidates in the final generation.
    """
    results_dir = "results/rnn-seeded"
    with open(f"{results_dir}/config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["n_jobs"] = 1
    evaluator = AquaCropEvaluator(**config["eval_params"])
    factory = RNNPrescriptorFactory()
    population = factory.load_population(f"{results_dir}/population")

    results_df = pd.read_csv(f"{results_dir}/results.csv")
    results_df = results_df[results_df["gen"] == results_df["gen"].max()]

    results_df = results_df.sort_values(by="irrigation")

    save_cols = ["year", "time_step_counter", "DryYield", "IrrDay", "mulch_pct"]
    all_outputs = []
    for cand_id in tqdm(results_df["cand_id"]):
        candidate = population[cand_id]
        full_outputs = evaluator.run_candidate(candidate, detailed_output=True)
        full_outputs = full_outputs[save_cols]
        full_outputs["cand_id"] = cand_id
        all_outputs.append(full_outputs)

    all_outputs = pd.concat(all_outputs, axis=0)
    all_outputs.to_csv("app/data.csv", index=False)

    # Get the weather data for the simulation period over each year
    weather_df = evaluator.weather_data
    weather_df["monthday"] = weather_df["Date"].dt.month * 100 + weather_df["Date"].dt.day
    start_month, start_day = evaluator.sim_start_date.split("/")
    end_month, end_day = evaluator.sim_end_date.split("/")
    start_month, start_day, end_month, end_day = int(start_month), int(start_day), int(end_month), int(end_day)
    start_monthday = start_month * 100 + start_day
    end_monthday = end_month * 100 + end_day

    filtered_weather_df = weather_df[
        (weather_df["monthday"] >= start_monthday) & (weather_df["monthday"] <= end_monthday) &
        (weather_df["year"].isin(all_outputs["year"].unique()))
    ]
    filtered_weather_df = filtered_weather_df.drop(columns=["monthday"])
    filtered_weather_df.to_csv("app/weather.csv", index=False)


if __name__ == "__main__":
    generate_data()
