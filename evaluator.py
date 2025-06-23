"""
Evaluator that runs the Aquacrop simulator.
"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement, FieldMngt
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
from sklearn.preprocessing import MinMaxScaler
import torch

from prescriptor import RNNPrescriptor

# pylint: disable=protected-access


class AquaCropEvaluator(Evaluator):
    """
    Evaluator that runs the Aquacrop simulator.
    """
    def __init__(self, aquacrop_params: dict, device: str = "cpu", n_jobs: int = 1):
        """
        aquacrop params:
            sim_start_date: str
            sim_end_date: str
            weather_filepath: str
            soil_params:
                soil_type: str
            crop_params:
                crop_type: str
                planting_date: str
            init_wc_params:
                wc_type: str
                value: list[int]
        scale_data_path: optional path to a CSV file containing data to scale the features with
        """
        super().__init__(["yield", "irrigation", "mulch_pct"], n_jobs=n_jobs)

        # Set up AquaCrop model
        filepath = get_filepath(aquacrop_params["weather_filepath"])
        self.weather_data = prepare_weather(filepath)
        self.soil = Soil(**aquacrop_params["soil_params"])
        self.crop = Crop(**aquacrop_params["crop_params"])
        self.init_wc = InitialWaterContent(**aquacrop_params["init_wc_params"])

        self.sim_start_date = aquacrop_params['sim_start_date']
        self.sim_end_date = aquacrop_params['sim_end_date']

        self.device = device

        # Scaler
        # NOTE: We are technically cheating by looking into the future of weather here
        scaler = MinMaxScaler()
        scaler.fit(self.weather_data.drop(columns=["Date"]))

        self.weather_data["year"] = self.weather_data["Date"].dt.year
        weather_scaled = []
        for year in self.weather_data["year"].unique():
            year_data = self.weather_data[self.weather_data["year"] == year]
            year_data = year_data.sort_values(by="Date")
            scaled = scaler.transform(year_data.drop(columns=["Date", "year"]))
            if scaled.shape[0] > 365:
                scaled = scaled[:365]
            weather_scaled.append(scaled)

        weather_scaled = np.stack(weather_scaled, axis=0)
        self.weather_scaled = torch.from_numpy(weather_scaled).float().to(self.device)
        self.years = self.weather_data["year"].unique().tolist()

        # For now subset the weather data
        precips = self.weather_scaled[:, :, 2]
        avg_precips = precips.mean(axis=1)
        lowest_precip = torch.argmin(avg_precips).item()
        highest_precip = torch.argmax(avg_precips).item()
        subset_idxs = [lowest_precip, highest_precip, len(self.years) - 2, len(self.years) - 1]
        self.weather_scaled = self.weather_scaled[subset_idxs]
        self.years = [self.years[i] for i in subset_idxs]

    def update_predictor(self, _):
        pass

    def run_candidate(self, candidate: RNNPrescriptor, detailed_output: bool = False) -> pd.DataFrame:
        """
        Runs a candidate to get actions for each context. (Policy for each year of weather)
        Then runs each policy through the AquaCrop model for each year.
        Returns a DataFrame with the results. If detailed_output is True, each row corresponds to a time step in each
        simulation. If False, returns a summary DataFrame where each row corresponds to a year.
        """
        # Batched torch inference
        with torch.no_grad():
            candidate.eval()
            candidate.to(self.device)
            policies = candidate.forward(self.weather_scaled)
            policies = policies.cpu().numpy()

        all_results_dfs = []
        for policy, year in zip(policies, self.years):
            irrigation_management = IrrigationManagement(irrigation_method=1,
                                                         SMT=policy[0:4].tolist())
            field_management = FieldMngt(mulches=True, mulch_pct=policy[4])
            aquacrop_input = {
                "irrigation_management": irrigation_management,
                "field_management": field_management
            }
            results_df = self.run_aquacrop(aquacrop_input, year, detailed_output=detailed_output)
            all_results_dfs.append(results_df)

        # Return detailed output or summary results
        results_df = pd.concat(all_results_dfs, axis=0)
        return results_df

    def run_aquacrop(self, aquacrop_input: dict, year: int, detailed_output: bool = False) -> pd.DataFrame:
        """
        Runs aquacrop on a given candidate strategy.
        Returns a DataFrame with the results:
            yield
            irrigation
            mulch_pct
        If detailed_output is True, returns a DataFrame with the entire time series of results for every year.
        """

        # Run policy through AquaCrop model in given year
        model = AquaCropModel(sim_start_time=f"{year}/{self.sim_start_date}",
                              sim_end_time=f"{year}/{self.sim_end_date}",
                              weather_df=self.weather_data,
                              soil=self.soil,
                              crop=self.crop,
                              initial_water_content=self.init_wc,
                              **aquacrop_input)
        model.run_model(till_termination=True)

        # Process results based on detailed_output flag
        mulch_pct = aquacrop_input["field_management"].mulch_pct
        if detailed_output:
            results_df = pd.concat([model._outputs.water_flux,
                                    model._outputs.water_storage,
                                    model._outputs.crop_growth], axis=1)
            results_df = results_df.loc[:, ~results_df.columns.duplicated()]
            results_df["mulch_pct"] = mulch_pct
            results_df["year"] = year
        else:
            results_df = model._outputs.final_stats
            results_df = results_df.rename(columns={
                "Dry yield (tonne/ha)": "yield",
                "Seasonal irrigation (mm)": "irrigation"
            })
            results_df["mulch_pct"] = mulch_pct
            results_df["year"] = year

        return results_df

    def evaluate_candidate(self, candidate: RNNPrescriptor) -> tuple[np.ndarray, int]:
        results_df = self.run_candidate(candidate)
        return np.array([-1 * results_df["yield"].mean(),
                         results_df["irrigation"].mean(),
                         results_df["mulch_pct"].mean()]), 0


# def main():
#     import yaml
#     with open("config.yml", "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#     config["eval_params"]["n_jobs"] = 1

#     evaluator = AquaCropEvaluator(**config["eval_params"])
#     dummy_prescriptor = RNNPrescriptor()

#     results = evaluator.run_aquacrop(dummy_prescriptor)
#     print(results)

# if __name__ == "__main__":
#     main()
