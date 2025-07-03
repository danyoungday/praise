"""
Trains seeds with backpropagation to acquire desired behavior across a set of contexts. These can be injected into the
initial population in Evolution.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import yaml

from evaluator import AquaCropEvaluator
from prescriptor import RNNPrescriptorFactory, RNNPrescriptor


def train_seed(prescriptor: RNNPrescriptor,
               evaluator: AquaCropEvaluator,
               label: torch.Tensor,
               device: str) -> RNNPrescriptor:
    """
    Trains a single seed using backpropagation. Uses the evaluator to get the context data to train on. Takes a label
    of desired behavior for all contexts in the evaluator.
    """
    prescriptor.to(device)

    weather_scaled = evaluator.weather_scaled.to(device)
    label = label.repeat(weather_scaled.shape[0], 1).to(device)

    dataset = TensorDataset(weather_scaled, label)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(prescriptor.parameters())
    criterion = torch.nn.MSELoss()

    with tqdm(range(1000), desc="Training Seed") as pbar:
        avg_loss = float("inf")
        for _ in pbar:
            prescriptor.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                pred = prescriptor.forward(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(weather_scaled)
            pbar.set_postfix(loss=f"{avg_loss:.2e}")

    return prescriptor


def main():
    """
    Main logic to train seeds. We get all combinations of (no mulch, full mulch) and (no irrigation, full irrigation)
    """
    device = "mps"
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    factory = RNNPrescriptorFactory()

    evaluator = AquaCropEvaluator(**config["eval_params"])

    seeds = []

    irr_seed_values = [[0.0, 0.0, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]]
    mulch_seed_values = [[0.0], [100.0]]
    for irr_value in irr_seed_values:
        for mulch_value in mulch_seed_values:
            label = torch.tensor(irr_value + mulch_value, dtype=torch.float32)
            prescriptor = factory.random_init()
            prescriptor = train_seed(prescriptor, evaluator, label, device)
            prescriptor.cand_id = f"0_{len(seeds)}"
            seeds.append(prescriptor)

    factory.save_population(seeds, config["evolution_params"]["seed_path"])


if __name__ == "__main__":
    main()
