"""
Data processing util functions
"""
import pandas as pd

def trim_post_harvest(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each baseline in the results_df, remove all the post-harvest data.
    """
    all_dfs = []
    for baseline in results_df["baseline"].unique():
        baseline_df = results_df[results_df["baseline"] == baseline]
        harvest_idx = baseline_df[baseline_df["DryYield"] != 0].last_valid_index()
        baseline_df = baseline_df.loc[:harvest_idx]
        all_dfs.append(baseline_df)
    return pd.concat(all_dfs)