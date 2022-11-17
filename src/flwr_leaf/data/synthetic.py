import json
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset


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


def load_synthetic_data(data_path: str, cid: int, batch_size: int):
    loaders = []
    for split in ["train", "test"]:
        dataset = SyntheticDataset(data_path, cid, split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if split == "train" else False,
            )
        )

    return loaders
