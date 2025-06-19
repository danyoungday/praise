"""
Evaluator that runs the Aquacrop simulator.
"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
from presp.prescriptor import Prescriptor
from sklearn.preprocessing import MinMaxScaler

from utils import trim_post_harvest

# pylint: disable=protected-access


class AquaCropEvaluator(Evaluator):
    """
    Evaluator that runs the Aquacrop simulator.
    """
    def __init__(self, aquacrop_params: dict, features: list[str], scale_data_path: str = None, n_jobs: int = 1):
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
        super().__init__(["yield", "irrigation"], n_jobs=n_jobs)

        # Set up AquaCrop model
        filepath = get_filepath(aquacrop_params["weather_filepath"])
        self.weather_data = prepare_weather(filepath)
        self.soil = Soil(**aquacrop_params["soil_params"])
        self.crop = Crop(**aquacrop_params["crop_params"])
        self.init_wc = InitialWaterContent(**aquacrop_params["init_wc_params"])

        self.sim_start_date = aquacrop_params['sim_start_date']
        self.sim_end_date = aquacrop_params['sim_end_date']

        self.features = list(features)

        # Scaler
        # NOTE: We are technically cheating by looking into the future of weather here
        self.scaler = None
        self.all_feature_names = None
        if scale_data_path:
            self.scaler = MinMaxScaler()
            scaler_data = pd.read_csv(scale_data_path)
            scaler_data = trim_post_harvest(scaler_data)

            self.all_feature_names = scaler_data.columns.tolist()
            self.all_feature_names.remove("baseline")
            self.all_feature_names.remove("depth")
            scaler_data = scaler_data[features]
            self.scaler.fit(scaler_data)

    def update_predictor(self, _):
        pass

    def construct_features(self, model: AquaCropModel, t: int) -> np.ndarray:
        """
        Extracts features from the AquaCropModel at time step t.
        Take weather from time t and outputs from time t-1.
        """
        weather = model._weather[t]  # column 4 is the date
        water_flux = model._outputs.water_flux[t-1]
        # columns 0 and 2 are duplicates
        water_storage_cols = [1] + list(range(3, model._outputs.water_storage.shape[1]))
        water_storage = model._outputs.water_storage[t-1, water_storage_cols]
        # columns 0, 1 and 2 are duplicates
        crop_growth = model._outputs.crop_growth[t-1, 3:]

        feature_data = np.concatenate((weather, water_flux, water_storage, crop_growth))

        if self.scaler:
            df_row = pd.DataFrame([feature_data], columns=self.all_feature_names)
            feature_data = self.scaler.transform(df_row[self.features])

        feature_data = feature_data.astype(np.float32)
        return feature_data

    def run_aquacrop(self, model: AquaCropModel, candidate: Prescriptor) -> pd.DataFrame:
        """
        Runs the passed AquaCrop model with irrigation depths prescribed by a candidate.
        If no candidate is passed, just runs the model to termination with no custom irrigation strategy.
        Returns a dict:
            depths: list[float] - depths prescribed at each time step
            final_stats: pd.DataFrame - final stats from the run
            water_flux: pd.DataFrame - water flux data at each time step
            water_storage: pd.DataFrame - water storage data at each time step
            crop_growth: pd.DataFrame - crop growth data at each time step
        """
        t = 0
        depths = []
        if candidate is not None:
            model._initialize()
            while model._clock_struct.model_is_finished is False:
                features = self.construct_features(model, t)
                t += 1
                # get depth to apply
                depth = candidate.forward(features)
                depths.append(depth)
                model._param_struct.IrrMngt.depth = depth
                model.run_model(initialize_model=False)
        
        else:
            model.run_model(till_termination=True)

        results_df = pd.concat([model._outputs.water_flux,
                                model._outputs.water_storage,
                                model._outputs.crop_growth], axis=1)
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        depths_col = np.zeros(len(results_df))
        depths_col[:len(depths)] = depths
        results_df["depth"] = depths_col
        return results_df

    def evaluate_candidate(self, candidate: Prescriptor) -> tuple[np.ndarray, int]:
        # combine into aquacrop model and specify start and end simulation date
        model = AquaCropModel(sim_start_time=self.sim_start_date,
                              sim_end_time=self.sim_end_date,
                              weather_df=self.weather_data,
                              soil=self.soil,
                              crop=self.crop,
                              initial_water_content=self.init_wc,
                              irrigation_management=IrrigationManagement(irrigation_method=5))

        results_df = self.run_aquacrop(model, candidate)
        dry_yield = results_df["DryYield"].max()
        seasonal_irrigation = results_df["IrrDay"].sum()

        return np.array([-1 * dry_yield, seasonal_irrigation]), 0


# def main():
#     import yaml
#     from prescriptor import AquaCropPrescriptor, ReservoirPrescriptor
#     with open("config.yml", "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
#     config["prescriptor_params"]["input_size"] = 5
#     config["prescriptor_params"]["model_params"][0]["in_features"] = 64
#     config["eval_params"]["n_jobs"] = 1

#     evaluator = AquaCropEvaluator(**config["eval_params"])
#     dummy_prescriptor = ReservoirPrescriptor(**config["prescriptor_params"])

#     model = AquaCropModel(sim_start_time=evaluator.sim_start_date,
#                           sim_end_time=evaluator.sim_end_date,
#                           weather_df=evaluator.weather_data,
#                           soil=evaluator.soil,
#                           crop=evaluator.crop,
#                           initial_water_content=evaluator.init_wc,
#                           irrigation_management=IrrigationManagement(irrigation_method=5))

#     results_df = evaluator.run_aquacrop(model, dummy_prescriptor)
#     final_stats = model._outputs.final_stats
#     print(final_stats)

# if __name__ == "__main__":
#     main()