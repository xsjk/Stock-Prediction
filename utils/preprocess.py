import pandas as pd
import numpy as np
import os
from typing import Callable, Iterable
from functools import reduce


def get_technical_indicators(data: pd.Series) -> pd.DataFrame:
    dataset = pd.DataFrame(index=data.index)

    # Create 7 and 21 days Moving Average
    dataset['ma7'] = data.rolling(window=7).mean()
    dataset['ma21'] = data.rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = data.ewm(span=26).mean()
    dataset['12ema'] = data.ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = data.rolling(window=20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = data.ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = data-1
    # dataset['log_momentum'] = np.log(dataset['momentum'])
    
    return dataset
    

def get_fourier_components(data: pd.Series, n_components: list[int]=[3, 6, 9, 27, 81, 100]) -> pd.DataFrame:
    dataset = pd.DataFrame(index=data.index)

    fft = np.fft.fft(data)
    for n in n_components:
        fft_ = fft.copy()
        fft_[n: -n] = 0
        dataset[f'FT {n} components'] = np.real(np.fft.ifft(fft_))

    return dataset


def get_wavelet_features(data: pd.Series) -> pd.DataFrame:
    dataset = pd.DataFrame(index=data.index)

    # Create wavelet features
    import pywt
    cA, cD = pywt.dwt(data, 'db1')
    cA_ext = pywt.upcoef('a', cA, 'db1', take=len(data))
    cD_ext = pywt.upcoef('d', cD, 'db1', take=len(data))

    dataset['cA'] = cA_ext
    dataset['cD'] = cD_ext
    
    return dataset

 
def get_stft_features(data: pd.Series, N: int=10) -> pd.DataFrame:
    # Create STFT features
    from scipy import signal
    f, t, Zxx = signal.stft(data, nperseg=N*2+1, noverlap=N*2)
    dataset = pd.DataFrame(np.abs(Zxx.T), index=data.index).add_prefix('STFT')
    return dataset.iloc[N:-N]


def get_news_features(data: pd.Series) -> pd.DataFrame:
    if data.name == 'Apple Close':
        news = pd.read_csv('data/Apple News.csv', index_col=0, parse_dates=True)
        news['Evaluation'] = news['Evaluation'].map({'negative': 0, 'neutral': 0.5, 'positive': 1})
        news_evaluation = pd.Series({date: d['Evaluation'].dot(d['Prob']) / d['Prob'].sum() for date, d in news.groupby('Date')}).rename('News')
        return news_evaluation.to_frame()
    else:
        return pd.DataFrame(index=data.index)

def preprocess(
        dataset: pd.DataFrame, 
        target: str | Iterable[str], 
        processors: Iterable[Callable[[pd.Series], pd.DataFrame]] = [
            get_technical_indicators, 
            get_fourier_components, 
            get_wavelet_features, 
            get_stft_features,
            get_news_features
        ]
    ) -> pd.DataFrame:

    if isinstance(target, str):
        features = reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
            map(lambda getter: getter(dataset[f'{target} Close']), processors)
        )
        print(f'{target} {features.shape}')

        return pd.merge(dataset, features.add_prefix(f'{target} '), left_index=True, right_index=True).sort_index(axis=1)
    else:
        for t in target:
            dataset = preprocess(dataset, t, processors)

        return dataset

def merge(raw_data_dict: dict[str, pd.DataFrame]):
    dataset = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), map(lambda x: x[1].add_prefix(f'{x[0]} '), raw_data_dict.items()))
    return dataset.loc[dataset.index.duplicated(keep="first") == False].dropna()

if __name__ == "__main__":
    import argparse
    from rich import print
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, nargs='+', default=[])
    parser.add_argument('--processors', type=str, nargs='+', default=['technical_indicators', 'fourier_components', 'wavelet_features', 'stft_features', 'news_features'], choices=['technical_indicators', 'fourier_components', 'wavelet_features', 'stft_features', 'news_features'])
    parser.add_argument('-o', '--save-path', type=str, default='data/processed_dataset.pkl')
    args = parser.parse_args()

    print(f"Preprocessing {args.targets} with {args.processors}")
    if args.targets == []:
        args.targets = [name[:-16] for name in os.listdir('data') if name.endswith('(2017-2023).csv')]
    args.processors = [globals()[f'get_{name}'] for name in args.processors]

    raw_data_dict = {name[:-16]: pd.read_csv(f'data/{name}', index_col=0, parse_dates=True) for name in os.listdir('data') if name[:-16] in args.targets}
    dataset = merge(raw_data_dict)
    dataset = preprocess(dataset, raw_data_dict.keys(), args.processors)
    dataset.dropna(inplace=True)
    dataset.to_pickle(args.save_path)
    print(f"Result saved to {args.save_path}. Shape: {dataset.shape}")