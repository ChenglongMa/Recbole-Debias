# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn
import gc
import logging
from logging import getLogger

import pandas as pd
import torch
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color

from recbole_debias.config import Config
from recbole_debias.data import create_dataset, data_preparation
from recbole_debias.utils import get_model, get_trainer
from recbole_debias.utils.case_study import full_sort_topk


def run_recbole_debias(model=None, model_file=None, dataset=None, config_file_list=None,
                       config_dict=None, saved=True, to_evaluate=True,
                       batch_size=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    if config['single_spec']:
        config['device'] = torch.device("cuda") if config['use_gpu'] and torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer: Trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = None, None
    if model_file is None:
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved, show_progress=config['show_progress'])
    else:
        # When calculate ItemCoverage metrics, we need to run this code for set item_nums in eval_collector.
        trainer.eval_collector.data_collect(train_data)

    if to_evaluate:
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved, model_file=model_file, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
        topk_result = None
    else:
        test_result = None
        topk_result = full_predict(model, dataset, config, test_data, batch_size=batch_size)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'topk_result': topk_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data


@torch.no_grad()
def full_predict(model, dataset, config, test_data, batch_size=None) -> pd.DataFrame:
    # uid_series = None
    # uid_series = dataset.token2id(dataset.uid_field, ["196", "186"])
    # if uid_series is None:
    uid_series = torch.arange(1, test_data.dataset.user_num)

    topk_score, topk_iid_list = full_sort_topk(
        model, test_data, uid_series=uid_series, k=max(config['topk']), device=config["device"], batch_size=batch_size
    )
    # print(topk_score)  # scores of top 10 items
    # print(topk_iid_list)  # internal id of top 10 items
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    external_user_list = dataset.id2token(dataset.uid_field, uid_series)
    topk_result = pd.DataFrame.from_dict(dict(zip(external_user_list, external_item_list))).melt(var_name=dataset.uid_field,
                                                                                                 value_name=dataset.iid_field)
    topk_result['score'] = topk_score.cpu().flatten()
    return topk_result
