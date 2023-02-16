# @Time   : 2022/3/22
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse
import os
import time

import pandas as pd

from recbole_debias.quick_start import run_recbole_debias
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')
    parser.add_argument('--model_file', '-f', type=str, default=None, help='model checkpoint file')
    parser.add_argument('--no-evaluate', '-n', action=argparse.BooleanOptionalAction,
                        help='DO NOT evaluate but just generate top-k or prediction score')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Batch size for full sort prediction')

    args, _ = parser.parse_known_args()
    to_evaluate = not args.no_evaluate
    batch_size = args.batch_size
    model = args.model
    dataset = args.dataset

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    mf = args.model_file
    if mf is not None and mf.split('-')[0].lower() != args.model.lower():
        raise TypeError("Checkpoint file does not match model name")

    model_file = f"saved/{mf}.pth" if mf is not None else None
    start = time.time()
    result = run_recbole_debias(model=model, model_file=model_file,
                                dataset=dataset, config_file_list=config_file_list,
                                to_evaluate=to_evaluate, batch_size=batch_size)
    elapse = (time.time() - start) / 60  # unit: s

    test_result = result['test_result']
    topk_result = result['topk_result']

    os.makedirs('./result/', exist_ok=True)
    now = time.strftime("%y%m%d%H%M%S")
    if isinstance(test_result, dict):
        res = []
        for metric, value in test_result.items():
            res.append([model, metric, value, elapse])
        res = pd.DataFrame(res, columns=['model', 'metric', 'value', 'elapse(mins)'])
        res.to_csv(f'./result/result_{model}_{dataset}_{now}.csv', index=False)
        print(res.head())
    if isinstance(topk_result, pd.DataFrame):
        topk_result.to_csv(f'./result/topk_{model}_{dataset}_{now}.csv', index=False)

    # @2302051603: replace extractor with equivalent GRU cell
    # @------1632: again, replace u_inputs with r_inputs
