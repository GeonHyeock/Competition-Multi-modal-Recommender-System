from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FMDataset(Dataset):
    def __init__(self, data, image, text):
        self.data = data
        self.image = image
        self.text = text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, rating = self.data[idx]

        return {
            "x": np.array([user, item], dtype=np.int32),
            "image": self.image[int(item)],
            "text": self.text[int(item)],
            "y": rating,
        }


if __name__ == "__main__":
    data = pd.read_csv("data/raw/train.csv").to_numpy()
    image = np.load("data/raw/image.npy")
    text = np.load("data/raw/text.npy")
    dataset = FMDataset(data, image, text)
