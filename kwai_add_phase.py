import pandas as pd

if __name__ == '__main__':
    dir_name = './dataset'
    sep = ','

    phases = ['train', 'valid', 'test']
    for phase in phases:
        filename = f'{dir_name}/kwai/kwai.{phase}.inter'
        df = pd.read_csv(filename, sep=sep)
        df['phase:token'] = phase
        df.to_csv(filename, index=False)

    print('Done')
