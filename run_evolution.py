"""
Main running logic
"""
from pathlib import Path
import shutil

import pandas as pd
from presp.evolution import Evolution
from presp.prescriptor import NNPrescriptorFactory
import wandb
import yaml  # pylint: disable=wrong-import-order

from evaluator import AquaCropEvaluator
from data.generate import DataGenerator
from prescriptor import AquaCropPrescriptor, ReservoirPrescriptor


def main():
    """
    Main logic to run evolution.
    """
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    save_path = Path(config["evolution_params"]["save_path"])

    # wandb logging
    run = wandb.init(project="praise", name=save_path.name, config=config)
    run.log_code(exclude_fn=lambda path: "baseline.py" in path)

    # Clear save dir if it exists
    if save_path.exists():
        delete = input("Directory already exists. Delete? (y/n):")
        if delete.lower() == "y":
            shutil.rmtree(save_path)
        else:
            print("Exiting without running evolution.")
            return
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2("config.yml", save_path / "config.yml")

    # If data path does not exist, run data generation
    data_path = config["eval_params"]["scale_data_path"]
    if not Path(data_path).exists():
        print("Generating data...")
        generator = DataGenerator(config["eval_params"]["aquacrop_params"], config["eval_params"]["features"])
        all_results_df = generator.generate_data("baselines/one-season.csv")
        all_results_df.to_csv(data_path, index=False)

    # Set up and run evolution
    evaluator = AquaCropEvaluator(**config["eval_params"])
    presc_cls = AquaCropPrescriptor if config["prescriptor_type"] == "aquacrop" else ReservoirPrescriptor
    prescriptor_factory = NNPrescriptorFactory(presc_cls, **config["prescriptor_params"])
    evolution = Evolution(evaluator=evaluator,
                          prescriptor_factory=prescriptor_factory,
                          **config["evolution_params"])
    evolution.run_evolution()

    # More wandb logging
    results_df = pd.read_csv(save_path / "results.csv")
    results_table = wandb.Table(dataframe=results_df)
    run.log({"results": results_table})

    pop_artifact = wandb.Artifact("population", type="population")
    pop_artifact.add_file(save_path / "population")
    run.log_artifact(pop_artifact)

    run.finish()


if __name__ == "__main__":
    main()
