# @Time   : 2022/3/22
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse
import time

import pandas as pd

from recbole_debias.quick_start import run_recbole_debias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    start = time.time()
    result = run_recbole_debias(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elapse = (time.time() - start) / 60  # unit: s

    res = []
    for metric, value in result['test_result'].items():
        res.append([args.model, metric, value, elapse])
    res = pd.DataFrame(res, columns=['model', 'metric', 'value', 'elapse(s)'])
    res.to_csv('./result.csv', index=False)
    print(res.head())
