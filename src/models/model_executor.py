from pprint import pprint

import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef


import numpy as np
import src.config as co
import wandb

class NNEngine(pl.LightningModule):

    def __init__(
        self,
        model_type,
        neural_architecture,
        optimizer,
        lr,
        weight_decay=0,
        loss_weights=None,
        remote_log=None
    ):
        super().__init__()
        assert optimizer == co.Optimizers.ADAM.value or optimizer == co.Optimizers.RMSPROP.value

        self.remote_log = remote_log

        self.model_type = model_type
        self.neural_architecture = neural_architecture

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        out = self.neural_architecture(x)
        logits = self.softmax(out)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val

    def training_epoch_end(self, validation_step_outputs):
        losses = [el["loss"].item() for el in validation_step_outputs]
        sum_losses = float(np.sum(losses))
        self.log(co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value, sum_losses, prog_bar=True)

        if self.remote_log is not None:
            self.remote_log.log({co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value: sum_losses})

    def validation_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.VALIDATION)

    def test_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.TESTING)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y = batch
        prediction = self(x)
        loss_val = self.loss_fn(prediction, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(prediction, dim=1)

        return prediction_ind, y, loss_val

    def __validation_and_testing_end(self, validation_step_outputs, model_step):

        predictions, ys, loss_vals = [], [], []
        for prediction, y, loss_val in validation_step_outputs:
            predictions += prediction.tolist()
            ys += y.tolist()
            loss_vals += [loss_val.item()]

        self.__compute_cm(ys, predictions, model_step, 'ALL')

        val_dict = self.__compute_metrics(ys, predictions, model_step, loss_vals, 'ALL')

        df = pd.DataFrame(
            list(zip(stock_names, predictions, ys)),
            columns=['stock_names', 'predictions', 'ys']
        )

        if co.CHOSEN_STOCKS[co.STK_OPEN.TEST] == co.Stocks.ALL:
            # computing metrics per stock
            for si in co.CHOSEN_STOCKS[co.STK_OPEN.TEST].value:
                df_si = df[df['Name'] == si]
                ys = df_si['ys'].to_numpy()
                predictions = df_si['predictions'].to_numpy()
                val_dict.update(self.__compute_metrics(ys, predictions, model_step, loss_vals, si))
                self.__compute_cm(ys, predictions, model_step, si)

        # for saving best model
        self.log(model_step.value + co.Metrics.F1.value, f1score, prog_bar=True)

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)
            self.remote_log.log({model_step.value + "_conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=co.CLASS_NAMES,
                title=model_step.value + "_conf_mat")}
            )

    def configure_optimizers(self):

        if self.model_type == co.Models.DAIN:
            return torch.optim.RMSprop([
                {'params': self.neural_architecture.base.parameters()},
                {'params': self.neural_architecture.dean.mean_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.mean_lr},
                {'params': self.neural_architecture.dean.scaling_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.scale_lr},
                {'params': self.neural_architecture.dean.gating_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.gate_lr},
            ], lr=self.lr)

        if self.optimizer == co.Optimizers.ADAM.value:
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == co.Optimizers.RMSPROP.value:
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
