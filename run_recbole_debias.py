# @Time   : 2022/3/22
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse
import os
import time

import pandas as pd
import torch

from recbole_debias.quick_start import run_recbole_debias

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')
    parser.add_argument('--model_file', '-f', type=str, default=None, help='model checkpoint files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    mf = args.model_file
    if mf.split('-')[0].lower() != args.model.lower():
        raise TypeError("Checkpoint file does not match model name")

    model_file = f"saved/{mf}.pth" if mf is not None else None
    start = time.time()
    result = run_recbole_debias(model=args.model, dataset=args.dataset, config_file_list=config_file_list, model_file=model_file)
    elapse = (time.time() - start) / 60  # unit: s

    res = []
    for metric, value in result['test_result'].items():
        res.append([args.model, metric, value, elapse])
    res = pd.DataFrame(res, columns=['model', 'metric', 'value', 'elapse(mins)'])
    os.makedirs('./result/', exist_ok=True)
    now = time.strftime("%y%m%d%H%M%S")
    res.to_csv(f'./result/result_{now}.csv', index=False)
    print(res.head())
    # @2302051603: replace extractor with equivalent GRU cell
    # @------1632: again, replace u_inputs with r_inputs
