"""
Evaluator that runs the Aquacrop simulator.
"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import numpy as np
import pandas as pd
from presp.evaluator import Evaluator
from presp.prescriptor import Prescriptor

# pylint: disable=protected-access


class AquaCropEvaluator(Evaluator):
    """
    Evaluator that runs the Aquacrop simulator.
    """
    def __init__(self, aquacrop_params: dict, n_jobs: int):
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
        """
        super().__init__(["yield", "irrigation"], n_jobs=n_jobs)

        filepath = get_filepath(aquacrop_params["weather_filepath"])
        self.weather_data = prepare_weather(filepath)
        self.soil = Soil(**aquacrop_params["soil_params"])
        self.crop = Crop(**aquacrop_params["crop_params"])
        self.init_wc = InitialWaterContent(**aquacrop_params["init_wc_params"])

        self.sim_start_date = aquacrop_params['sim_start_date']
        self.sim_end_date = aquacrop_params['sim_end_date']

    def update_predictor(self, _):
        pass

    def construct_features(self, model: AquaCropModel, t: int) -> np.ndarray:
        """
        Extracts features from the AquaCropModel at time step t.
        Take weather from time t and outputs from time t-1.
        Convert to torch tensor for the NNPrescriptor.
        """
        total_t = model._outputs.crop_growth.shape[0]

        # Weather features: low and high temp, precipitation, evapotranspiration, time
        weather = model._weather[t, 0:4]  # weather[:, 4] is the time, not needed
        weather_mins = [0.0, 0.0, 0.0, 0.0]
        weather_maxes = [50.0, 50.0, 100.0, 10.0]

        # exclude water_flux[:, 4] because it's always nan
        # Water flux features: Water requirement, surface storage, irrigation per day, infiltration, runoff,
        # deep percolation, capillary rise, groundwater inflow, soil evaporation, potential soil evaporation,
        # crop transpiration, potential crop transpiration
        mask = np.ones(model._outputs.water_flux.shape[1], dtype=bool)
        mask[[1, 4]] = False
        water_flux = model._outputs.water_flux[t-1, mask]
        water_flux_mins = [0.0] * 14
        water_flux_maxes = [total_t, total_t, 400.0, 10.0, 50.0, 50.0, 50.0, 30.0, 5.0, 5.0, 8.0, 8.0, 8.0, 8.0]

        # Water storage features: time, growing season, days after planting, water content in each compartment (1-12)
        water_storage = model._outputs.water_storage[t-1, 3:]
        water_storage_mins = [0.0] * 12
        water_storage_maxes = [0.5] * 12  # guessing from chatgpt

        # Crop growth features: time, season, dap, growing degree days + cum, root depth, canopy cover + no stress,
        # biomass + no stress, harvest index + adjusted, dry + fresh yield + potential 
        crop_growth = model._outputs.crop_growth[t-1, 3:]
        crop_growth_mins = [0.0] * 12
        # We're cheating a bit here by knowing how long the crops are going to grow for
        # TODO: yield scaling is hard-coded
        crop_growth_maxes = [25.0, 25.0 * total_t, 2.0, 1.0, 1.0, 2000.0, 2000.0, 0.6, 0.6, 20.0, 20.0, 20.0]

        # Normalize features
        weather = (weather - np.array(weather_mins)) / (np.array(weather_maxes) - np.array(weather_mins))
        water_flux = (water_flux - np.array(water_flux_mins)) / (np.array(water_flux_maxes) - np.array(water_flux_mins))
        water_storage = (water_storage - np.array(water_storage_mins)) / (np.array(water_storage_maxes) - np.array(water_storage_mins))  # noqa
        crop_growth = (crop_growth - np.array(crop_growth_mins)) / (np.array(crop_growth_maxes) - np.array(crop_growth_mins))  # noqa

        features = np.concatenate((weather, water_flux, water_storage, crop_growth))
        features = features.astype(np.float32)
        features = features.reshape(1, -1)
        return features

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
        model._initialize()
        t = 0
        depths = []
        while model._clock_struct.model_is_finished is False:
            # If we have a candidate, construct features for it and then get the depth to apply
            if candidate is not None:
                features = self.construct_features(model, t)
                t += 1
                # get depth to apply
                depth = candidate.forward(features)
                depths.append(depth)
                model._param_struct.IrrMngt.depth = depth
            # If we have no candidate, keep the default irrigation strategy and log the depth
            else:
                depths.append(model._param_struct.IrrMngt.depth)

            model.run_model(initialize_model=False)

        results_df = pd.concat([model._outputs.water_flux,
                                model._outputs.water_storage,
                                model._outputs.crop_growth], axis=1)
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        results_df["depth"] = depths
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

        results = self.run_aquacrop(model, candidate)
        final_stats = results["final_stats"]
        return np.array([-1 * final_stats["Dry yield (tonne/ha)"].mean(),
                         final_stats["Seasonal irrigation (mm)"].mean()]), 0
