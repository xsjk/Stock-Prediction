
import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import mean_squared_error

from .model import Discriminator, Generator
from .data import StockDataSet


class GAN(LightningModule):
    def __init__(
            self, 
            num_days_for_predict, 
            num_days_to_predict, 
            target='Apple',
            learning_rate=0.00002, 
            momentum=None,
            num_workers=1, 
            batch_size=128, 
            train_size=0.8, 
            val_size=0.1,
            optimizer=torch.optim.Adam,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_days_for_predict = num_days_for_predict
        self.num_days_to_predict = num_days_to_predict
        self.target = target
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.optimizer = optimizer
        self.prepare_data()
        self.G = Generator(
            input_size = self.raw_dataset.dim,
            output_size = num_days_to_predict
        )
        self.D = Discriminator(
            input_size = num_days_for_predict+num_days_to_predict
        )
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

    def prepare_data(self) -> None:
        self.raw_dataset = StockDataSet.from_preprocessed(target=self.target)
        X, Y = self.raw_dataset[:]
        X = torch.from_numpy(np.array([X[i:i+self.num_days_for_predict] for i in range(len(X)-self.num_days_for_predict-self.num_days_to_predict+1)]))
        Y = torch.from_numpy(np.array([Y[i:i+self.num_days_for_predict+self.num_days_to_predict] for i in range(len(Y)-self.num_days_for_predict-self.num_days_to_predict+1)]))
        self.dataset = TensorDataset(X, Y)

    def setup(self, stage: str) -> None:
        i1 = int(len(self.dataset) * self.train_size)
        i2 = int(len(self.dataset) * (self.train_size + self.val_size))
        match stage:
            case "fit":
                self.train_dataset = TensorDataset(*self.dataset[:i1])
                self.val_dataset = TensorDataset(*self.dataset[i1:i2])
            case "test":
                self.test_dataset = TensorDataset(*self.dataset[i2:])
            case "predict":
                pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False, num_workers=self.num_workers)
    
    def forward(self, x):
        return self.G(x)
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        optG, optD = self.optimizers()

        fake_data = self.G(x).reshape(-1, self.num_days_to_predict, 1)
        fake_data = torch.cat([y[:, :self.num_days_for_predict, :], fake_data], axis = 1)

        real_output = self.D(y)
        fake_output = self.D(fake_data)
        real_labels = torch.ones_like(real_output, device=self.device)
        fake_labels = torch.zeros_like(fake_output, device=self.device)

        lossD = self.criterion(real_output, real_labels) \
              + self.criterion(fake_output, fake_labels)

        optD.zero_grad()
        lossD.backward(retain_graph=True)
        optD.step()

        fake_output = self.D(fake_data)
        lossG = self.criterion(fake_output, real_labels)
        optG.zero_grad()
        lossG.backward()
        optG.step()
        
        self.log_dict({"lossG": lossG, "lossD": lossD, "lossG+lossD": lossG+lossD})

    def validation_step(self, val_batch, batch_idx):
        # calculate RMSE
        x, y = val_batch
        y_true = self.raw_dataset.inverse_transform(y[:, self.num_days_for_predict].cpu()).flatten()
        y_pred = self.raw_dataset.inverse_transform(self.G(x).cpu()).flatten()
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        self.log("val_RMSE", rmse)
        return rmse
        
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.raw_dataset.inverse_transform(self.G(x).cpu()).flatten()
        return y_pred

    def configure_optimizers(self):
        if self.momentum is None:
            return (self.optimizer(self.G.parameters(), lr=self.learning_rate),
                    self.optimizer(self.D.parameters(), lr=self.learning_rate))
        else:
            return (self.optimizer(self.G.parameters(), lr=self.learning_rate, momentum=self.momentum),
                    self.optimizer(self.D.parameters(), lr=self.learning_rate, momentum=self.momentum))


checkpoint_callback = ModelCheckpoint(
    monitor='val_RMSE',
    dirpath='./model_checkpoint',
    save_top_k=3,
)

early_stop_callback = EarlyStopping(
    monitor='val_RMSE',
    min_delta=0.00,
    patience=100,
    verbose=False,
    mode='min'
)

def config_parser(
        parser: ArgumentParser,
        targets: list[str] = None,
        optimizers: list[str] = None,
):

    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    new_parser = subparsers.add_parser('new', help='Train a new model')
    resume_parser = subparsers.add_parser('resume', help='Resume training a model')

    
    GAN_parser = new_parser.add_argument_group("GAN", "Arguments for GAN")
    GAN_parser.add_argument("target", type=str, default="Apple", choices=targets, help="Target stock to predict")
    GAN_parser.add_argument("--num-days-for-predict", type=int, default=10, help="Number of days used for prediction")
    GAN_parser.add_argument("--num-days-to-predict", type=int, default=1, help="Number of days to predict")
    GAN_parser.add_argument("--learning-rate", type=float, default=0.00002, help="Learning rate for both generator and discriminator")
    GAN_parser.add_argument("--momentum", type=float, default=None, help="Momentum for both generator and discriminator")
    GAN_parser.add_argument("--optimizer", type=str, default="adam", choices=optimizers, help="Optimizer for both generator and discriminator")
    GAN_parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    GAN_parser.add_argument("--batch-size", type=int, default=128, help="Batch size for dataloader")
    
    resume_parser = resume_parser.add_argument_group("Resume")
    resume_parser.add_argument("checkpoint_path", type=str, default=None)
    
    for p in (GAN_parser, resume_parser):
        trainer_parser = p.add_argument_group("Trainer", "Arguments for Trainer")
        trainer_parser.add_argument("--min-epochs", type=int, default=100, help="Minimum number of epochs")
        trainer_parser.add_argument("--max-epochs", type=int, default=-1, help="Maximum number of epochs")
        trainer_parser.add_argument("--early-stop", action="store_true", default=False, help="Whether to use early stopping")
        trainer_parser.add_argument("--early-stop-patience", type=int, default=100, help="Patience for early stopping")

    return parser

if __name__ == "__main__":

    from rich import traceback
    traceback.install()
    import warnings
    warnings.filterwarnings("ignore")

    optimizer_map = {
        "adam": torch.optim.Adam,
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }

    parser = config_parser(
        ArgumentParser(),
        targets = sorted(name[:-16] for name in os.listdir('data') if name.endswith('(2017-2023).csv')),
        optimizers = sorted(optimizer_map.keys())
    )
    args = parser.parse_args()
    args.optimizer = optimizer_map[args.optimizer]

    callbacks = [checkpoint_callback]
    if args.early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_RMSE',
            min_delta=0.00,
            patience=args.early_stop_patience,
            verbose=False,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(
        max_epochs = args.max_epochs,
        min_epochs = args.min_epochs,
        log_every_n_steps=10,
        callbacks = callbacks,
    )

    checkpoint_callback.dirpath = os.path.join(
        trainer.logger.log_dir,
        'checkpoints'
    )

    
    match args.subcommand:
        case "new":                    
            model = GAN(
                num_days_for_predict = args.num_days_for_predict,
                num_days_to_predict = args.num_days_to_predict,
                learning_rate = args.learning_rate,
                momentum = args.momentum,
                num_workers = args.num_workers,
                batch_size=args.batch_size,
                optimizer=args.optimizer,
            )
            trainer.fit(model)
        case "resume":
            print("resume from", args.checkpoint_path)
            model = GAN.load_from_checkpoint(args.checkpoint_path)
            trainer.fit(model, ckpt_path=args.checkpoint_path)
        
    trainer.save_checkpoint(os.path.join(checkpoint_callback.dirpath, "last.ckpt"))

