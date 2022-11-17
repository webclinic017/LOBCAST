import torch.cuda

from pprint import pprint
from collections import Counter


from src.data_preprocessing.FI.FIDataBuilder import FIDataBuilder
from src.data_preprocessing.FI.FIDataset import FIDataset
from src.data_preprocessing.FI.FIDataModule import FIDataModule
from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from src.data_preprocessing.LOB.LOBDataModule import LOBDataModule
from src.data_preprocessing.LOB.LOBDataset import LOBDataset
import src.models.model_callbacks as cbk
import numpy as np
from pytorch_lightning import Trainer

from src.models.model_executor import NNEngine

from src.models.mlp.mlp_param_search import hyperparameters_mlp
from src.models.lstm.lstm_param_search import hyperparameters_lstm
from src.models.deeplob.dlb_param_search import hyperparameters_dlb
from src.data_preprocessing.FI.fi_param_search import hyperparameters_fi
from src.data_preprocessing.LOB.lobster_param_search import hyperparameters_lobster

import src.config as co
import wandb

from src.models.mlp.mlp import MLP
from src.models.lstm.lstm import LSTM
from src.models.deeplob.deeplob import DeepLob

# def parser_cl_arguments():
#     """ Parses the arguments for the command line. """
#
#     parser = argparse.ArgumentParser(description='PyTorch Training')
#
#     parser.add_argument('--model',    default=co.Models.MLP.value)
#     parser.add_argument('--data',     default=co.DatasetFamily.LOBSTER.value)
#     parser.add_argument('--is_sweep', default=True)
#     parser.add_argument('--back_win', default=co.BACKWARD_WINDOW)
#     parser.add_argument('--forw_win', default=co.FORWARD_WINDOW)
#
#     return parser


SWEEP_CONF_DICT_MODEL = {
    co.Models.MLP:  hyperparameters_mlp,
    co.Models.LSTM: hyperparameters_lstm,
    co.Models.DEEPLOB: hyperparameters_dlb,
}

SWEEP_CONF_DICT_DATA = {
    co.DatasetFamily.FI:  hyperparameters_fi,
    co.DatasetFamily.LOBSTER:  hyperparameters_lobster,
}


def prepare_data_FI():

    fi_train = FIDataBuilder(
        co.DATA_SOURCE + co.DATASET_FI,
        dataset_type=co.DatasetType.TRAIN,
        horizon=co.HORIZON,
        window=co.BACKWARD_WINDOW
    )

    fi_val = FIDataBuilder(
        co.DATA_SOURCE + co.DATASET_FI,
        dataset_type=co.DatasetType.TEST,
        horizon=co.HORIZON,
        window=co.BACKWARD_WINDOW
    )

    train_set = FIDataset(x=fi_train.get_samples_x(), y=fi_train.get_samples_y())
    val_set = FIDataset(x=fi_val.get_samples_x(), y=fi_val.get_samples_y())

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set))
    print()

    fi_dm = FIDataModule(train_set, val_set, co.BATCH_SIZE)
    return fi_dm



def prepare_data_LOBSTER():

    # train
    lo_train = LOBSTERDataBuilder(
        co.DATASET_LOBSTER,
        co.DatasetType.TRAIN,
        start_end_trading_day=("2021-08-11", "2021-08-17"),  # ("2022-03-01", "2022-03-07")  5 days
        crop_trading_day_by=60 * 30,
        window_size_forward=co.FORWARD_WINDOW,
        window_size_backward=co.BACKWARD_WINDOW,
        label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
        is_shuffle=co.IS_SHUFFLE_INPUT,
        is_data_preload=co.IS_DATA_PRELOAD
    )

    # use the same
    mu, sigma, train_lab_threshold_pos, train_lab_threshold_neg = lo_train.normalization_means, lo_train.normalization_stds, lo_train.label_threshold_pos, lo_train.label_threshold_neg

    # validation
    lo_val = LOBSTERDataBuilder(
        co.DATASET_LOBSTER,
        co.DatasetType.VALIDATION,
        start_end_trading_day=("2021-08-18", "2021-08-24"),  # ("2022-03-01", "2022-03-07")  5 days
        crop_trading_day_by=60 * 30,
        window_size_forward=co.FORWARD_WINDOW,
        window_size_backward=co.BACKWARD_WINDOW,
        normalization_mean=mu,
        normalization_std=sigma,
        label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
        #label_threshold_pos=train_lab_threshold_pos,
        #label_threshold_neg=train_lab_threshold_neg,
        is_shuffle=co.IS_SHUFFLE_INPUT,
        is_data_preload=co.IS_DATA_PRELOAD
    )

    # test
    lo_test = LOBSTERDataBuilder(
        co.DATASET_LOBSTER,
        co.DatasetType.TEST,
        start_end_trading_day=("2021-08-25", "2021-08-27"),  # ("2022-03-02", "2022-03-03") 3 test
        crop_trading_day_by=60 * 30,
        window_size_forward=co.FORWARD_WINDOW,
        window_size_backward=co.BACKWARD_WINDOW,
        normalization_mean=mu,
        normalization_std=sigma,
        label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
        #label_threshold_pos=train_lab_threshold_pos,
        #label_threshold_neg=train_lab_threshold_neg,
        is_shuffle=co.IS_SHUFFLE_INPUT,
        is_data_preload=co.IS_DATA_PRELOAD
    )

    train_set = LOBDataset(x=lo_train.get_samples_x(), y=lo_train.get_samples_y())
    val_set = LOBDataset(x=lo_val.get_samples_x(),   y=lo_val.get_samples_y())
    test_set = LOBDataset(x=lo_test.get_samples_x(),  y=lo_test.get_samples_y())

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    lob_dm = LOBDataModule(train_set, val_set, test_set, co.BATCH_SIZE)
    return lob_dm



def lunch_training():
    print("Lunching the execution of {} on {} dataset.".format(co.CHOSEN_MODEL, co.CHOSEN_DATASET))

    remote_log = None
    if co.IS_WANDB:

        wandb.init(project=co.PROJECT_NAME, entity="fin-di-sapienza")
        remote_log = wandb

        co.BATCH_SIZE = wandb.config.batch_size
        co.LEARNING_RATE = wandb.config.lr
        co.IS_SHUFFLE_INPUT = wandb.config.is_shuffle

        if co.DATASET_FAMILY == co.DatasetFamily.LOBSTER:
            co.BACKWARD_WINDOW = wandb.config.window_size_backward
            co.FORWARD_WINDOW = wandb.config.window_size_forward
            co.LABELING_SIGMA_SCALER = wandb.config.labeling_sigma_scaler
        elif co.DATASET_FAMILY == co.DatasetFamily.FI:
            co.BACKWARD_WINDOW = wandb.config.window_size_backward
            co.HORIZON = wandb.config.fi_horizon_k

        if co.CHOSEN_MODEL == co.Models.MLP or co.CHOSEN_MODEL == co.Models.MLP_FI:
            co.MLP_HIDDEN = wandb.config.hidden_mlp
            co.P_DROPOUT = wandb.config.p_dropout
        elif co.CHOSEN_MODEL == co.Models.LSTM:
            co.LSTM_HIDDEN = wandb.config.lstm_hidden
            co.LSTM_N_HIDDEN = wandb.config.lstm_n_hidden

    data_module = pick_dataset(co.CHOSEN_DATASET)

    model = pick_model(co.CHOSEN_MODEL, data_module, remote_log)

    trainer = Trainer(
        gpus=co.DEVICE,
        check_val_every_n_epoch=co.VALIDATE_EVERY,  # val_check_interval
        max_epochs=co.EPOCHS,
        callbacks=[
            cbk.callback_save_model(co.CHOSEN_MODEL.value),
            cbk.early_stopping()
        ]
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path="best")

    wandb.finish()


def lunch_training_sweep():
    # 🐝 STEP: initialize sweep by passing in config

    sweep_id = wandb.sweep(
        sweep={
            'name': co.SWEEP_NAME,
            'method': co.SWEEP_METHOD,
            'metric': co.SWEEP_METRIC,
            'parameters': {
                **SWEEP_CONF_DICT_DATA[co.CHOSEN_DATASET],
                **SWEEP_CONF_DICT_MODEL[co.CHOSEN_MODEL]
            }
        },
        project=co.PROJECT_NAME
    )
    wandb.agent(sweep_id, function=lunch_training)  # count=4 max trials
    # wandb agent -p lob-adversarial-attacks-22 -e matteoprata rygxo9ti  to run sweeps in parallel

def pick_dataset(datasetFamily):
    if datasetFamily == co.DatasetFamily.LOBSTER:
        return prepare_data_LOBSTER()
    elif datasetFamily == co.DatasetFamily.FI:
        return prepare_data_FI()


def pick_model(chosen_model, data_module, remote_log):
    net_architecture = None
    if chosen_model == co.Models.MLP or chosen_model == co.Models.MLP_FI:
        net_architecture = MLP(
            x_shape=np.prod(data_module.x_shape),  # 40 * wind
            y_shape=data_module.y_shape,
            hidden_layer_dim=co.MLP_HIDDEN,
            p_dropout=co.P_DROPOUT
        )

    elif chosen_model == co.Models.LSTM:
        net_architecture = LSTM(
            x_shape=data_module.x_shape[1],  # 40, wind is the time
            y_shape=data_module.y_shape,
            hidden_layer_dim=co.LSTM_HIDDEN,
            num_layers=co.LSTM_N_HIDDEN
        )

    elif chosen_model == co.Models.DEEPLOB:
        net_architecture = DeepLob(y_shape=data_module.y_shape)

    return NNEngine(net_architecture, lr=co.LEARNING_RATE, remote_log=remote_log).to(co.DEVICE_TYPE)


if __name__ == "__main__":

    # test set: Counter({1: 47392, 0: 6347, 2: 5633})
    # y_pred = [1]*59372 # hp: the model predicts always 1
    # y_true = [0]*6347 + [1]*47392 + [2]*5633
    # print(f1(y_true, y_pred, average='micro')) # = 0.798
    # print(f1(y_true, y_pred, average='macro')) # = 0.295
    # print(f1(y_true, y_pred, average='weighted')) # = 0.709
    # exit()

    if co.IS_WANDB:
        lunch_training_sweep()
    else:
        lunch_training()
