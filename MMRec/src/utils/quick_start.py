# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import wandb
import torch.nn as nn
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_seed(config["seed"])
    init_logger(config)
    logger = getLogger()

    # print config infor
    logger.info("██Server: \t" + platform.node())
    logger.info("██Dir: \t" + os.getcwd() + "\n")
    logger.info(config)

    # load data
    dataset = RecDataset(config)

    # print dataset statistics
    logger.info(str(dataset))
    train_dataset, valid_dataset, _ = dataset.split()
    logger.info("\n====Training====\n" + str(train_dataset))
    logger.info("\n====Validation====\n" + str(valid_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config["train_batch_size"], shuffle=True
    )
    valid_data = EvalDataLoader(
        config,
        valid_dataset,
        additional_dataset=train_dataset,
        batch_size=config["eval_batch_size"],
    )

    ############ Dataset loadded, run model
    hyper_ret, idx = [], 0
    logger.info("\n\n=================================\n\n")

    # hyper-parameters & combinations
    hyper_ls = [config[i] or [None] for i in config["hyper_parameters"]]
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        for j, k in zip(config["hyper_parameters"], hyper_tuple):
            config[j] = k
        hyper = {i: config[i] for i in config["hyper_parameters"]}
        wandb.init(
            project="inha-Competition",
            group=config["model"],
            name=str(hyper),
            config=hyper,
        )

        logger.info(
            "========={}/{}: Parameters:{}={}=======".format(
                idx + 1, total_loops, config["hyper_parameters"], hyper_tuple
            )
        )

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config["model"])(config, train_data).to(config["device"])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        _, best_valid_result, _ = trainer.fit(train_data, valid_data=valid_data)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result))

        logger.info("best valid result: {}".format(dict2str(best_valid_result)))
        wandb.run.finish()

    # log info
    logger.info("\n============All Over=====================")
    for p, k in hyper_ret:
        logger.info(
            "Parameters: {}={},\n best valid: {},".format(
                config["hyper_parameters"], p, dict2str(k)
            )
        )

    logger.info("\n\n█████████████ BEST ████████████████")
    logger.info(
        "\tParameters: {}={},\nValid: {}".format(
            config["hyper_parameters"],
            hyper_ret[0][0],
            dict2str(hyper_ret[0][1]),
        )
    )
