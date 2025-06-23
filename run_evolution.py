"""
Main running logic
"""
from pathlib import Path
import shutil

import pandas as pd
from presp.evolution import Evolution
import wandb
import yaml  # pylint: disable=wrong-import-order

from evaluator import AquaCropEvaluator
from prescriptor import RNNPrescriptorFactory


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

    # Set up and run evolution
    evaluator = AquaCropEvaluator(**config["eval_params"])
    prescriptor_factory = RNNPrescriptorFactory()

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
