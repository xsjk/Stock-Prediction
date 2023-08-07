import pandas as pd
import numpy as np

def get_technical_indicators(data):
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
    dataset['log_momentum'] = np.log(dataset['momentum'])
    
    return dataset
    

def get_fourier_components(data, n_components):
    dataset = pd.DataFrame(index=data.index)

    fft = np.fft.fft(data)
    for n in n_components:
        fft_ = fft.copy()
        fft_[n: -n] = 0
        dataset[f'FT {n} components'] = np.real(np.fft.ifft(fft_))

    return dataset


def get_wavelet_features(data):
    dataset = pd.DataFrame(index=data.index)

    # Create wavelet features
    import pywt
    cA, cD = pywt.dwt(data, 'db1')
    cA_ext = pywt.upcoef('a', cA, 'db1', take=len(data))
    cD_ext = pywt.upcoef('d', cD, 'db1', take=len(data))

    dataset['cA'] = cA_ext
    dataset['cD'] = cD_ext
    
    return dataset

 
def get_stft_features(data, N=3):
    # Create STFT features
    from scipy import signal
    f, t, Zxx = signal.stft(data, nperseg=N*2+1, noverlap=N*2)
    dataset = pd.DataFrame(np.abs(Zxx.T), index=data.index).add_prefix('STFT')
    return dataset.iloc[N:-N]