from torch.utils.data import Dataset
import pandas as pd

class StockDataSet(Dataset):
    
    def __init__(self, df: pd.DataFrame, target='Apple'):
        self.df = df
        self.length, self.dim = self.df.shape

        from sklearn.preprocessing import StandardScaler
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X = self.x_scaler.fit_transform(self.df.values)
        self.Y = self.y_scaler.fit_transform(self.df[f'{target} Close'].values.reshape(-1, 1))


    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.length
    
    def inverse_transform(self, y):
        return self.y_scaler.inverse_transform(y)
    
    @classmethod
    def from_preprocessed(cls, path='data/processed_dataset.pkl', target='Apple'):
        return cls(pd.read_pickle(path).astype('float32'), target)

    @classmethod
    def from_raw(cls, folder_path='data', target='Apple'):
        import os
        from .preprocess import merge, preprocess
        df = merge({name[:-16]: pd.read_csv(f'{folder_path}/{name}', index_col=0, parse_dates=True) 
                    for name in os.listdir(folder_path) if name.endswith('(2017-2023).csv')})
        df = preprocess(df, target).dropna().astype('float32')
        return cls(df, target)
    
    
