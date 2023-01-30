import argparse

import pandas as pd

from recbole_debias.quick_start import run_recbole_debias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', '-m', nargs='+', type=str, default=['MF'], help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    res = []

    for model in args.models:
        result = run_recbole_debias(model=model, dataset=args.dataset, config_file_list=config_file_list)
        for k, v in result['test_result'].items():
            res.append([model, k, v])
    res = pd.DataFrame(res, columns=['model', 'metric', 'value'])
    res.to_csv('./result.csv', index=False)
    print(res.head())

