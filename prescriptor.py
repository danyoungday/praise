"""
Prescriptor implementation for AquaCrop.
"""
import numpy as np
from presp.prescriptor import Prescriptor, NNPrescriptor
import torch


class AquaCropPrescriptor(NNPrescriptor):
    """
    Prescriptor that uses an NN to prescribe depth of irrigation based on features from the AquaCrop model.
    """
    def __init__(self, depth_scale: float, model_params: list[dict], device: str = "cpu"):
        super().__init__(model_params, device)
        self.depth_scale = depth_scale

    def forward(self, context: np.ndarray) -> float:
        """
        Forward pass, then scale the output.
        """
        x = torch.from_numpy(context).float().to(self.device)
        output = super().forward(x).item()
        return output * self.depth_scale


class HeuristicPrescriptor(Prescriptor):
    """
    Heuristic prescriptor from the AquaCrop examples.
    """
    def forward(self, context: np.ndarray) -> float:
        pass
