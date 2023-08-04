import torch
import yaml
import argparse
import os
import pandas as pd
import time
import re
from tqdm import tqdm
from utils.dataset import RecDataset
from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def submission(args):
    model = torch.load(os.path.join("saved", args.model, args.Saved_Model)).to("cuda")
    with open("./src/configs/overall.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(f"./src/configs/model/{args.model}.yaml") as f:
        config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

    with open(f"./src/configs/dataset/{args.dataset}.yaml") as f:
        config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

    config = Config(args.model, args.dataset, config_dict)
    if args.Saved_Model.find("fold") >= 0:
        idx = args.Saved_Model.find("fold")
        config["inter_splitting_label"] = args.Saved_Model[idx : idx + 6]
    dataset = RecDataset(config)
    train, _, _ = dataset.split()
    print(dataset)
    print(train)
    eval_data = EvalDataLoader(
        config,
        dataset,
        additional_dataset=train,
        batch_size=1028,
    )
    model.eval()
    submission_user, submission_item = [], []
    with tqdm(total=len(eval_data)) as pbar:
        start = time.time()
        for _, batched_data in tqdm(enumerate(eval_data)):
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
    end = time.time()

    submission = pd.DataFrame({"user_id": submission_user, "item_id": submission_item})
    i_id_mapping = pd.read_csv("./data/Inha/i_id_mapping.csv", sep="\t").to_dict()
    u_id_mapping = pd.read_csv("./data/Inha/u_id_mapping.csv", sep="\t").to_dict()

    submission.user_id = submission.user_id.map(u_id_mapping["user_id"])
    submission.item_id = submission.item_id.map(i_id_mapping["asin"])

    submission.to_csv(f"../submission/{args.model}/{args.Saved_Model}.csv", index=False)

    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BM3", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Inha", help="name of datasets"
    )
    parser.add_argument(
        "--Saved_Model", "-S", type=str, default="saved/BM3", help="model.pt"
    )
    args, _ = parser.parse_known_args()

    createDirectory(os.path.join(f"../submission/{args.model}"))

    p = re.compile(r"[0-9]\.?[0-9]*")
    n_layer, embedding_size, feat_embed_dim, inference_time = [], [], [], []
    for pt in os.listdir(args.Saved_Model):
        if "inter" in pt:
            args.Saved_Model = pt
            layer, _, emb, feat, _ = p.findall(pt)
            inf_time = submission(args)

            n_layer.append(layer)
            embedding_size.append(emb)
            feat_embed_dim.append(feat)
            inference_time.append(inf_time)
    pd.DataFrame(
        {
            "n_layer": n_layer,
            "embedding_size": embedding_size,
            "feat_embed_dim": feat_embed_dim,
            "inference_time": inference_time,
        }
    ).to_csv("../logs/infer_time.csv", index=False)
    print("Done")
