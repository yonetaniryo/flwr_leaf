"""modified version of https://github.com/adap/flower/blob/main/examples/quickstart_pytorch/client.py"""

from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..data.synthetic import load_synthetic_data
from .net import MLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(batch["x"].to(DEVICE)), batch["y"].to(DEVICE)).backward()
            optimizer.step()


def test(net: nn.Module, testloader: DataLoader) -> list:
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


class SyntheticClient(fl.client.NumPyClient):
    def __init__(self, config):
        self.net = MLP(config.net.input_dim, config.net.num_classes)
        self.local_epochs = config.optim.local_epochs
        self.lr = config.optim.lr
        self.trainloader, self.testloader = load_synthetic_data(
            config.data_path, config.cid, config.optim.batch_size
        )

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
