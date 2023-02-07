import argparse
import os
import time

import pandas as pd

from recbole_debias.quick_start import run_recbole_debias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', '-m', nargs='+', type=str, default=['MF'], help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='kwai', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')
    parser.add_argument('--model_files', '-f', nargs='+', type=str, default=None, help='model checkpoint files')

    args, _ = parser.parse_known_args()
    models = args.models
    mfs = args.model_files
    model_and_files = []
    for i in range(len(models)):
        mf = mfs[i] if i < len(mfs) else None
        model = models[i]
        if mf.split('-')[0].lower() != model.lower():
            raise TypeError("Checkpoint file does not match model name")
        model_and_files.append((model, f"saved/{mf}.pth" if mf is not None else None))

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    res = []

    for model, model_file in model_and_files:
        start = time.time()
        try:
            result = run_recbole_debias(model=model, dataset=args.dataset, config_file_list=config_file_list, model_file=model_file)
            elapse = (time.time() - start) / 60  # unit: s
            for metric, value in result['test_result'].items():
                res.append([model, metric, value, elapse])
        except:
            print('Error')
            pass

    res = pd.DataFrame(res, columns=['model', 'metric', 'value', 'elapse(mins)'])
    os.makedirs('./result/', exist_ok=True)
    now = time.strftime("%y%m%d%H%M%S")
    res.to_csv(f'./result/result_{now}.csv', index=False)
    print(res.head())
