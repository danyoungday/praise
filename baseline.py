"""
Code to run the baseline optimization strategy using the simplex algorithm
"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin
from tqdm import tqdm
import yaml


class BaselineRunner:
    """
    Runner that keeps track of AquaCrop params to run the model in optimization.
    """
    def __init__(self, aquacrop_params: dict):
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
        filepath = get_filepath(aquacrop_params["weather_filepath"])
        self.weather_data = prepare_weather(filepath)
        self.soil = Soil(**aquacrop_params["soil_params"])
        self.crop = Crop(**aquacrop_params["crop_params"])
        self.init_wc = InitialWaterContent(**aquacrop_params["init_wc_params"])

        self.sim_start_date = aquacrop_params['sim_start_date']
        self.sim_end_date = aquacrop_params['sim_end_date']

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
                              irrigation_management=irrmngt)

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

    baseline_runner = BaselineRunner(config["eval_params"]["aquacrop_params"])

    opt_smts = []
    yld_list = []
    tirr_list = []
    for max_irr in tqdm(range(0, 500, 50)):
        # find optimal thresholds and save to list
        smts = baseline_runner.optimize(4, max_irr)
        opt_smts.append(smts)

        # save the optimal yield and total irrigation
        yld, tirr, _ = baseline_runner.evaluate(smts, max_irr, True)
        yld_list.append(yld)
        tirr_list.append(tirr)

    results_df = pd.DataFrame()
    for i in range(len(opt_smts[0])):
        results_df[f"SMT-{i+1}"] = [smt[i] for smt in opt_smts]
    results_df["yield"] = yld_list
    results_df["irrigation"] = tirr_list
    results_df["max_irrigation"] = list(range(0, 500, 50))
    results_df.to_csv(baseline_save_path, index=False)

    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))

    # plot results
    ax.scatter(tirr_list, yld_list)
    ax.plot(tirr_list, yld_list)

    # labels
    ax.set_xlabel('Total Irrigation (ha-mm)', fontsize=18)
    ax.set_ylabel('Yield (tonne/ha)', fontsize=18)
    ax.set_xlim([-20, 600])
    ax.set_ylim([2, 15.5])

    # annotate with optimal thresholds
    bbox = dict(boxstyle="round", fc="1")
    offset = [15, 15, 15, 15, 15, -125, -100,  -5, 10, 10]
    yoffset = [0, -5, -10, -15, -15,  0,  10, 15, -20, 10]
    for i, smt in enumerate(opt_smts):
        smt = smt.clip(0, 100)
        ax.annotate('(%.0f, %.0f, %.0f, %.0f)' % (smt[0], smt[1], smt[2], smt[3]),
                    (tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
                    bbox=bbox, fontsize=12)

    plt.show()


if __name__ == "__main__":
    main("config.yml", "baselines/one-season.csv")
