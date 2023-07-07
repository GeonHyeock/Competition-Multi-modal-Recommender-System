import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict
from tqdm import tqdm


def submission(model, batch=5):
    model.eval()
    item_id = []

    with torch.no_grad():
        for u in tqdm(user):
            data = torch.vstack([u.repeat(len(item)), item]).T
            pred = model((data, None, None)).argsort(descending=True)[:50].cpu().numpy()

            item_id.extend(pred)

    submit = pd.DataFrame(
        {"user_id": user.cpu().numpy().repeat(50), "item_id": item_id}
    )
    submit.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from FM.model import FactorizationMachineModel

    train = pd.read_csv("data/raw/train.csv")
    user = torch.tensor(np.load("data/raw/user_label.npy")).type(torch.int32).to("cuda")
    item = torch.tensor(np.load("data/raw/item_label.npy")).type(torch.int32).to("cuda")
    model = FactorizationMachineModel([192403, 63001], 256)
    ckpt = torch.load("logs/train/runs/FM_base_141/checkpoints/epoch_141.ckpt")
    ckpt = {".".join(k.split(".")[1:]): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(ckpt)
    submission(model.to("cuda"))
