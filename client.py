import json
from collections import OrderedDict
from glob import glob

import flwr as fl
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y


class SyntheticDataset(Dataset):
    def __init__(self, path, cid, split):
        data = json.load(open(glob(f"{path}/{split}/*.json")[0]))
        users = data["users"]
        self.user_data = data["user_data"][users[cid]]

    def __len__(self):
        return len(self.user_data["x"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "x": torch.tensor(self.user_data["x"][idx]),
            "y": torch.tensor(self.user_data["y"][idx]),
        }

        return sample


def load_synthetic_data(config: dict):
    loaders = []
    for split in ["train", "test"]:
        dataset = SyntheticDataset(config.data_path, config.cid, split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True if split == "train" else False,
            )
        )

    return loaders


def train(net, trainloader, epochs, lr):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(batch["x"].to(DEVICE)), batch["y"].to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            outputs = net(batch["x"].to(DEVICE))
            labels = batch["y"].to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


class Client(fl.client.NumPyClient):
    def __init__(self, config):
        self.net = Net()
        self.local_epochs = config.local_epochs
        self.lr = config.lr
        self.trainloader, self.testloader = load_synthetic_data(config)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


@hydra.main(config_path="config", config_name="synthetic_client")
def main(config):
    fl.client.start_numpy_client(
        server_address=config.client_address, client=Client(config)
    )


if __name__ == "__main__":
    main()
