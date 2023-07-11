import torch
import yaml
import argparse
import os
import pandas as pd
from tqdm import tqdm
from utils.dataset import RecDataset
from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader

os.chdir("./MMRec")


PATH = "./saved/BM3_best_a.pt"
model = torch.load(PATH).to("cuda")
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="BM3", help="name of models")
parser.add_argument(
    "--dataset", "-d", type=str, default="Inha", help="name of datasets"
)
args, _ = parser.parse_known_args()

with open("./src/configs/overall.yaml") as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

with open(f"./src/configs/model/{args.model}.yaml") as f:
    config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

with open(f"./src/configs/dataset/{args.dataset}.yaml") as f:
    config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

config = Config(args.model, args.dataset, config_dict)


dataset = RecDataset(config)
train, _, _ = dataset.split()
print(dataset)
print(train)
eval_data = EvalDataLoader(
    config,
    dataset,
    additional_dataset=train,
    batch_size=128,
)
model.eval()
submission_user, submission_item = [], []
with tqdm(total=len(eval_data)) as pbar:
    for batch_idx, batched_data in tqdm(enumerate(eval_data)):
        # predict: interaction without item ids
        scores = model.full_sort_predict(batched_data)
        masked_items = batched_data[1]
        # mask out pos items
        scores[masked_items[0], masked_items[1]] = -1e10
        # rank and get top-k
        _, topk_index = torch.topk(scores, 50, dim=-1)  # nusers x topk
        for user in batched_data[0]:
            submission_user.extend(user.repeat(50).cpu().numpy())
        submission_item.extend(topk_index.view(-1).cpu().numpy())
        pbar.update(1)

submission = pd.DataFrame({"user_id": submission_user, "item_id": submission_item})

i_id_mapping = pd.read_csv("./data/Inha/i_id_mapping.csv", sep="\t").to_dict()
u_id_mapping = pd.read_csv("./data/Inha/u_id_mapping.csv", sep="\t").to_dict()

submission.user_id = submission.user_id.map(u_id_mapping["user_id"])
submission.item_id = submission.item_id.map(i_id_mapping["asin"])

submission.to_csv("./my_submission.csv", index=False)
print("Done")
