import world
import utils
from world import cprint
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import Procedure
import os
from os.path import join

# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
    wandb.login()
    wandb.init(
        project="inha-Competition",
        group=world.simple_model,
    )


try:
    for epoch in range(world.TRAIN_epochs):
        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
        )
        print(f"EPOCH[{epoch}/{world.TRAIN_epochs}] {output_information}")
        torch.save(Recmodel.state_dict(), weight_file)
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config["multicore"])
finally:
    if world.tensorboard:
        w.close()
print("end")
