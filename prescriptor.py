"""
Prescriptor implementation for AquaCrop.
"""
import numpy as np
from presp.prescriptor import NNPrescriptor
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


class ReservoirPrescriptor(NNPrescriptor):
    """
    Contains a fixed LSTM used as a reservoir for reservoir computing.
    NOTE: In this case, the input size is the true number of features. The input size in the model_params is actually
    the hidden size of the LSTM.
    """
    def __init__(self, depth_scale: float, input_size: int, model_params: list[dict], device: str = "cpu"):
        super().__init__(model_params, device)
        self.depth_scale = depth_scale

        self.hidden_size = model_params[0]["in_features"]
        self.reservoir = torch.nn.LSTM(input_size=input_size,
                                       hidden_size=model_params[0]["in_features"],
                                       batch_first=True)
        self.reservoir.to(self.device)

        self.hc = (torch.zeros(1, 1, self.hidden_size).to(self.device),
                   torch.zeros(1, 1, self.hidden_size).to(self.device))

    def forward(self, context: np.ndarray) -> float:
        """
        Forward pass through reservoir, then retrieve the output by passing the hidden state through the NN.
        """
        x = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        out, self.hc = self.reservoir(x, self.hc)
        out = out.squeeze(1)
        depth = super().forward(out).item() * self.depth_scale
        return depth

    def reset(self):
        """
        Resets hidden and cell states of the reservoir
        """
        self.hc = (torch.zeros(1, 1, self.hidden_size).to(self.device),
                   torch.zeros(1, 1, self.hidden_size).to(self.device))
