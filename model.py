import os
import torch
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from typing import Tuple, Optional
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


class CryptocurrencyForecast:
    """
    Example forecasting prediction model using Fusion Transformer. Largely based on the tutorial
    from the great PyTorch Forecasting package; [1] lightly adapted to use a cryptocurrency prices
    dataset.
    
    References
    ----------
    [1] https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
    """
    def __init__(self, data_path:str, max_prediction_length:int = 7, max_encoder_length: int = 30, learning_rate: float = 0.03,
                 hidden_size: int = 16, target_prediction_variable: str = "Close", batch_size:int = 32, num_workers: int = 0,
                 gpus: Optional[int] = None, max_epochs:int = 10, model_path: Optional[str] = None):
        
        pl.seed_everything(42)

        self.data_path = data_path
        self.model_path = model_path
        
        # load dataset CSV as Pandas DataFrame and convert columns to correct types
        self.data = pd.read_csv(data_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data["Symbol"] = self.data.Symbol.astype("category")

        # if user passes `gpus` parameter use it; otherwise, use the
        # `_PL_TRAINER_GPUS` environment variable automatically populated by
        # Grid indicating how many GPUs are availale
        if gpus:
            self.gpus = gpus
        else:
            self.gpus = os.getenv("_PL_TRAINER_GPUS", 0)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.hidden_size = hidden_size

        # used to predict a continuous variable in dataset
        self.target_prediction_variable = target_prediction_variable

        # configures horizon of predictions
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.training_cutoff = self.data["time_idx"].max() - self.max_prediction_length

    @property
    def dataset(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Creates training and validation instances of TimeSeriesDataSet"""
        # create training dataset
        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= self.training_cutoff],
            time_idx="time_idx",
            target=self.target_prediction_variable,
            group_ids=["Symbol"],
            min_encoder_length=self.max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["Symbol"],
            time_varying_known_reals=["High", "Low", "Open", "Volume", "Marketcap"],
            target_normalizer=GroupNormalizer(
                groups=["Symbol"], transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missings=True,
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation = TimeSeriesDataSet.from_dataset(training, self.data, predict=True, stop_randomization=True)

        return (training, validation)

    @property
    def dataloaders(self) -> Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]:
        """Creates train and validation DataLoders"""
        training, validation = self.dataset
        train = training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=self.num_workers)
        val = validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=self.num_workers)

        return (train, val)

    def train(self) -> None:
        """Train network using Lightning"""
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus,
            weights_summary="top",
            gradient_clip_val=0.1,
            limit_train_batches=30,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        training, _ = self.dataset
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),

            # changing both parameters to any other values raises exceptions; that's a bug in PyTorch Forecasting
            log_interval=0,
            log_val_interval=0,
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        # fit network
        train_dataloader, val_dataloader = self.dataloaders
        trainer.fit(
            tft,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def plot_predictions(self, n:int = 10) -> None:
        """Plots predictions against the validation dataset"""
        model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)

        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        # TODO: there's a bug with plot_predictions it looks like so I can't generate
        # plots correctly.
        _, val_dataloader = self.dataloaders
        raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True)
        for idx in range(n):
            try:
                model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
            except RuntimeError:
                print("Missing certain indices. Pre-process prediction data to contain all indices.")
