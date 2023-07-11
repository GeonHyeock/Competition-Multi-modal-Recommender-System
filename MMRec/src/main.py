# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
import wandb
from utils.quick_start import quick_start

os.environ["NUMEXPR_MAX_THREADS"] = "48"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BM3", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Inha", help="name of datasets"
    )

    with open("./src/configs/overall.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    args, _ = parser.parse_known_args()

    with open(f"./src/configs/model/{args.model}.yaml") as f:
        config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

    with open(f"./src/configs/dataset/{args.dataset}.yaml") as f:
        config_dict.update(yaml.load(f, Loader=yaml.FullLoader))

    wandb.login()
    quick_start(
        model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True
    )
