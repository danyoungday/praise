"""
Prescriptor implementation for AquaCrop.
"""
import copy

import numpy as np
from presp.prescriptor import Prescriptor, PrescriptorFactory, NNPrescriptor
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


class RNNPrescriptor(torch.nn.Module, Prescriptor):
    """
    Prescriptor that uses RNN
    """
    def __init__(self):
        torch.nn.Module.__init__(self)
        Prescriptor.__init__(self)
        self.rnn = torch.nn.RNN(input_size=4, hidden_size=64, batch_first=True)
        self.fc = torch.nn.Linear(64, 5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Context: N x L x Context
        Output: N x Actions
        """
        out, _ = self.rnn(context)
        presc = self.fc(out[:, -1, :])
        scaled = self.sigmoid(presc) * 100.0
        return scaled


class RNNPrescriptorFactory(PrescriptorFactory):
    """
    Prescriptor factory for the RNN Prescriptor. Same as the NNPrescriptorFactory but we separate the RNN and FC.
    """
    def random_init(self) -> RNNPrescriptor:
        """
        Note sure how to specially randomly initialize the RNNPrescriptor so we just let PyTorch do its default
        behavior.
        """
        return RNNPrescriptor()

    def crossover(self, parents: list[RNNPrescriptor]) -> list[RNNPrescriptor]:
        """
        NOTE: Now that RNNPrescriptor is a torch.nn.Module, we don't have to iterate over the parameters in the
        separate modules RNN and FC, we can just iterate over the parameters of the whole model. However, I'm keeping
        it like this for now because I'm afraid it may have unintended consequences.
        """
        with torch.no_grad():
            child = RNNPrescriptor()
            parent1_rnn, parent2_rnn = parents[0].rnn, parents[1].rnn

            child.rnn = copy.deepcopy(parent1_rnn)
            for child_param, parent2_param in zip(child.rnn.parameters(), parent2_rnn.parameters()):
                mask = torch.rand(child_param.data.shape) < 0.5
                child_param.data[mask] = parent2_param.data[mask]

            parent1_fc, parent2_fc = parents[0].fc, parents[1].fc
            child.fc = copy.deepcopy(parent1_fc)
            for child_param, parent2_param in zip(child.fc.parameters(), parent2_fc.parameters()):
                mask = torch.rand(child_param.data.shape) < 0.5
                child_param.data[mask] = parent2_param.data[mask]

            return [child]

    def mutation(self, candidate: RNNPrescriptor, mutation_rate: float, mutation_factor: float):
        with torch.no_grad():
            for param in candidate.rnn.parameters():
                mutate_mask = torch.rand(param.shape) < mutation_rate
                noise = torch.normal(0,
                                     mutation_factor,
                                     param[mutate_mask].shape,
                                     dtype=param.dtype)
                param[mutate_mask] *= (1 + noise)

    def save_population(self, population: list[RNNPrescriptor], path: str):
        pop_dict = {cand.cand_id: (cand.rnn.state_dict(), cand.fc.state_dict()) for cand in population}
        torch.save(pop_dict, path)

    def load_population(self, path: str) -> dict[str, RNNPrescriptor]:
        pop_dict = torch.load(path)
        population = {}
        for cand_id, (rnn_state, fc_state) in pop_dict.items():
            prescriptor = RNNPrescriptor()
            prescriptor.rnn.load_state_dict(rnn_state)
            prescriptor.fc.load_state_dict(fc_state)
            prescriptor.cand_id = cand_id
            population[cand_id] = prescriptor
        return population
