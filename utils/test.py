from utils.train import GAN
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser


def config_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    parser.add_argument('ckpt_path', type=str, help='path to the checkpoint to be tested')
    return parser

if __name__ == '__main__':
    from rich import print
    from rich import traceback
    traceback.install()
    import warnings
    warnings.filterwarnings("ignore")

    parser = config_parser()
    args = parser.parse_args()
    model = GAN.load_from_checkpoint(args.ckpt_path)
    print(model.hparams)
    target = model.hparams.target

    model.eval()
    model.freeze()
    X, Y = model.dataset[:]
    y_pred = model(X.to(model.device))
    y_true = model.raw_dataset.y_scaler.inverse_transform(Y[:, model.num_days_for_predict].cpu()).flatten()
    y_pred = model.raw_dataset.y_scaler.inverse_transform(y_pred.cpu()).flatten()


    split = int(len(y_pred)*model.hparams.train_size)
    y_train_pred, y_test_pred = y_pred[:split], y_pred[split:]
    y_train, y_test = y_true[:split], y_true[split:]
    train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
    test_RMSE = mean_squared_error(y_test, y_test_pred, squared=False)
    standard_RMSE = mean_squared_error(y_true[1:], y_true[:-1], squared=False)
    train_standard_RMSE = mean_squared_error(y_train[1:], y_train[:-1], squared=False)
    test_standard_RMSE = mean_squared_error(y_test[1:], y_test[:-1], squared=False)
    print('target:', target)
    print("train RMSE:", train_RMSE)
    print("test RMSE:", test_RMSE)
    print("standard RMSE:", standard_RMSE)
    print("train standard RMSE:", train_standard_RMSE)
    print("test standard RMSE:", test_standard_RMSE)
    print("train RMSE / standard RMSE:", train_RMSE/train_standard_RMSE)
    print("test RMSE / standard RMSE:", test_RMSE/test_standard_RMSE)


    df = model.raw_dataset.df
    df[f'{target} Close Pred'] = None
    df.iloc[-len(y_pred):, -1] = y_pred
    df[f'{target} Close'].plot(figsize=(16,8),label=f'True', color='#536897')
    df[f'{target} Close Pred'].plot(figsize=(16,8),label=f'Pred', color='#E17D81', rot=0)

    split = df.index[int(len(df) * model.hparams.train_size)]
    plt.plot((split, split), (df[f'{target} Close'].min(), df[f'{target} Close'].max()), linestyle='--', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{target} Close Price')
    plt.savefig(f'images/{target} Close Price Prediction.png', dpi=300, bbox_inches='tight')
    print(f'saved prediction plot to "images/{target} Close Price Prediction.png"')