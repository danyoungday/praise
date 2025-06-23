"""
Code to run the baseline optimization strategy using the simplex algorithm
"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement, FieldMngt
from aquacrop.utils import prepare_weather, get_filepath

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from tqdm import tqdm
import yaml

from evaluator import AquaCropEvaluator


class BaselineRunner:
    """
    Runner that keeps track of AquaCrop params to run the model in optimization.
    """
    def __init__(self, aquacrop_params: dict, mulch_pct: int, year: int):
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
        year: the year to run the model for
        """
        filepath = get_filepath(aquacrop_params["weather_filepath"])
        self.weather_data = prepare_weather(filepath)
        self.soil = Soil(**aquacrop_params["soil_params"])
        self.crop = Crop(**aquacrop_params["crop_params"])
        self.init_wc = InitialWaterContent(**aquacrop_params["init_wc_params"])

        self.sim_start_date = f"{year}/{aquacrop_params['sim_start_date']}"
        self.sim_end_date = f"{year}/{aquacrop_params['sim_end_date']}"

        self.field_mngt = FieldMngt(mulches=True, mulch_pct=mulch_pct)

    def run_model(self, smts, max_irr_season):
        """
        funciton to run model and return results for given set of soil moisture targets
        """
        irrmngt = IrrigationManagement(irrigation_method=1, SMT=smts, MaxIrrSeason=max_irr_season)  # define irrigation management

        # create and run model
        model = AquaCropModel(sim_start_time=self.sim_start_date,
                              sim_end_time=self.sim_end_date,
                              weather_df=self.weather_data,
                              soil=self.soil,
                              crop=self.crop,
                              initial_water_content=self.init_wc,
                              irrigation_management=irrmngt,
                              field_management=self.field_mngt)

        model.run_model(till_termination=True)
        return model.get_simulation_results()

    def evaluate(self, smts, max_irr_season, test=False):
        """
        funciton to run model and calculate reward (yield) for given set of soil moisture targets
        """
        # run model
        out = self.run_model(smts, max_irr_season)
        # get yields and total irrigation
        yld = out['Dry yield (tonne/ha)'].mean()
        tirr = out['Seasonal irrigation (mm)'].mean()

        reward = yld

        # return either the negative reward (for the optimization)
        # or the yield and total irrigation (for analysis)
        if test:
            return yld, tirr, reward
        else:
            return -reward

    def get_starting_point(self, num_smts, max_irr_season, num_searches):
        """
        find good starting threshold(s) for optimization
        """
        # get random SMT's
        x0list = np.random.rand(num_searches, num_smts)*100
        rlist = []
        # evaluate random SMT's
        for xtest in x0list:
            r = self.evaluate(xtest, max_irr_season,)
            rlist.append(r)

        # save best SMT
        x0 = x0list[np.argmin(rlist)]

        return x0

    def optimize(self, num_smts, max_irr_season, num_searches=100):
        """ 
        optimize thresholds to be profit maximising
        """
        # get starting optimization strategy
        x0 = self.get_starting_point(num_smts, max_irr_season, num_searches)
        # run optimization
        res = fmin(self.evaluate, x0, disp=0, args=(max_irr_season,))
        # reshape array
        smts = res.squeeze()
        # evaluate optimal strategy
        return smts


def main(config_path: str, baseline_save_path: str):
    """
    Given a config we use in evolution, run the baseline optimization strategy on the aquacrop configuration
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluator = AquaCropEvaluator(**config["eval_params"])
    years = evaluator.years

    mulch_pcts = [0, 50, 100]
    rows = []
    for mulch_pct in mulch_pcts:
        for year in tqdm(years, desc="Getting baseline for years"):
            baseline_runner = BaselineRunner(config["eval_params"]["aquacrop_params"], mulch_pct, year)
            for max_irr in tqdm(range(0, 500, 50), leave=False):
                row = {"year": year, "mulch_pct": mulch_pct, "max_irrigation": max_irr}
                # find optimal thresholds and save to list
                smts = baseline_runner.optimize(4, max_irr)
                for i, smt in enumerate(smts):
                    row[f"SMT-{i+1}"] = smt

                # save the optimal yield and total irrigation
                yld, tirr, _ = baseline_runner.evaluate(smts, max_irr, True)
                row["yield"] = yld
                row["irrigation"] = tirr

    results_df = pd.DataFrame(rows)
    results_df.to_csv(baseline_save_path, index=False)


if __name__ == "__main__":
    main("config.yml", "baselines/full-baseline.csv")
