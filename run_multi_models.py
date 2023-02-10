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
    parser.add_argument('--no-evaluate', '-n', action=argparse.BooleanOptionalAction,
                        help='DO NOT evaluate but just generate top-k or prediction score')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Batch size for full sort prediction')

    args, _ = parser.parse_known_args()
    to_evaluate = not args.no_evaluate
    batch_size = args.batch_size

    models = args.models
    mfs = args.model_files
    model_and_files = []
    for i in range(len(models)):
        mf = mfs[i] if i < len(mfs) else None
        model = models[i]
        if mf is not None and mf.split('-')[0].lower() != model.lower():
            raise TypeError("Checkpoint file does not match model name")
        model_and_files.append((model, f"saved/{mf}.pth" if mf is not None else None))

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    eval_res = []
    os.makedirs('./result/', exist_ok=True)
    now = time.strftime("%y%m%d%H%M%S")
    for model, model_file in model_and_files:
        start = time.time()
        try:
            result = run_recbole_debias(model=model, model_file=model_file,
                                        dataset=args.dataset, config_file_list=config_file_list,
                                        to_evaluate=to_evaluate, batch_size=batch_size)
            elapse = (time.time() - start) / 60  # unit: s
            test_result = result['test_result']
            topk_result = result['topk_result']
            if topk_result is not None:
                topk_result.to_csv(f'./result/topk_result_for_{model}_{now}.csv', index=False)
            if test_result is not None:
                for metric, value in test_result.items():
                    eval_res.append([model, metric, value, elapse])
        except:
            print('Error')
            pass

    if len(eval_res) > 0:
        eval_res = pd.DataFrame(eval_res, columns=['model', 'metric', 'value', 'elapse(mins)'])
        eval_res.to_csv(f'./result/result_{now}.csv', index=False)
        print(eval_res.head())
    print('Done')
