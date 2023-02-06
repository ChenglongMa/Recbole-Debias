import logging
from logging import getLogger

import torch
from recbole.data import construct_transform
from recbole.utils import init_logger, init_seed, set_color, get_flops
from recbole_debias.config import Config
from recbole_debias.data import create_dataset, data_preparation
from recbole_debias.quick_start import load_data_and_model
from recbole_debias.utils import get_model, get_trainer

if __name__ == '__main__':
    model_file = "saved/MACR-Feb-07-2023_02-43-09.pth"
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    test_result = trainer.evaluate(test_data, load_best_model=True, model_file=model_file, show_progress=config['show_progress'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    # return {
    #     'best_valid_score': best_valid_score,
    #     'valid_score_bigger': config['valid_metric_bigger'],
    #     'best_valid_result': best_valid_result,
    #     'test_result': test_result
    # }
