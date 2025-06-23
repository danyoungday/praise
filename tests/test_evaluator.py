"""
Evaluator unittests
"""
import unittest

from aquacrop import IrrigationManagement, FieldMngt
import pandas as pd
import yaml

from evaluator import AquaCropEvaluator


class TestEvaluator(unittest.TestCase):
    """
    Tests the AquaCropEvaluator to make sure it runs AquaCrop correctly and returns the expected results.
    """
    def setUp(self):
        with open("tests/test.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.evaluator = AquaCropEvaluator(**config["eval_params"])

    def test_output_size(self):
        """
        Ensures the results DataFrame has the expected shape either in base or detailed output mode.
        """
        irrigation_management = IrrigationManagement(irrigation_method=0)
        field_management = FieldMngt(mulches=False, mulch_pct=0)
        aquacrop_input = {
            "irrigation_management": irrigation_management,
            "field_management": field_management
        }
        base_results_df = self.evaluator.run_aquacrop(aquacrop_input,
                                                      year=2018,
                                                      detailed_output=False)

        detailed_results_df = self.evaluator.run_aquacrop(aquacrop_input,
                                                          year=2018,
                                                          detailed_output=True)

        self.assertEqual(base_results_df.shape, (1, 10))

        start_date = f"2018/{self.evaluator.sim_start_date}"
        end_date = f"2018/{self.evaluator.sim_end_date}"
        delta = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        n_days = delta.days + 1
        self.assertEqual(detailed_results_df.shape, (n_days, 43))
