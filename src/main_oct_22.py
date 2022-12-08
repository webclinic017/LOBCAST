# UTILS
from pprint import pprint
from collections import Counter
import numpy as np
import src.config as co
import wandb

# TORCH
from pytorch_lightning import Trainer
import src.models.model_callbacks as cbk
from src.models.model_executor import NNEngine

# DATASETS
from src.data_preprocessing.FI.FIDataBuilder import FIDataBuilder
from src.data_preprocessing.FI.FIDataset import FIDataset
from src.data_preprocessing.FI.FIDataModule import FIDataModule
from src.data_preprocessing.FI.fi_param_search import hyperparameters_fi
from src.data_preprocessing.LOB.LOBDataModule import LOBDataModule
from src.data_preprocessing.LOB.LOBDataset import LOBDataset
from src.data_preprocessing.LOB.lobster_param_search import hyperparameters_lobster

# MODELS
from src.models.mlp.mlp import MLP
from src.models.mlp.mlp_param_search import hyperparameters_mlp
from src.models.lstm.lstm import LSTM
from src.models.lstm.lstm_param_search import hyperparameters_lstm
from src.models.deeplob.deeplob import DeepLob
from src.models.deeplob.dlb_param_search import hyperparameters_dlb
from src.models.cnn1.cnn1 import CNN1
from src.models.cnn1.cnn1_param_search import hyperparameters_cnn1
from src.models.cnn2.cnn2 import CNN2
from src.models.cnn2.cnn2_param_search import hyperparameters_cnn2
from src.models.cnnlstm.cnnlstm import CNNLSTM
from src.models.cnnlstm.cnnlstm_param_search import hyperparameters_cnnlstm
from src.models.dain.dain import DAIN
from src.models.dain.dain_param_search import hyperparameters_dain
from src.models.translob.translob import TransLob
from src.models.translob.tlb_param_search import hyperparameters_tlb
from src.models.tabl.ctabl import CTABL
from src.models.tabl.weight_constraint import WeightConstraint
from src.models.tabl.tabl_param_search import hyperparameters_tabl

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
    co.Models.CNN1: hyperparameters_cnn1,
    co.Models.CNN2: hyperparameters_cnn2,
    co.Models.LSTM: hyperparameters_lstm,
    co.Models.CNNLSTM: hyperparameters_cnnlstm,
    co.Models.DAIN: hyperparameters_dain,
    co.Models.DEEPLOB: hyperparameters_dlb,
    co.Models.TRANSLOB: hyperparameters_tlb,
    co.Models.CTABL: hyperparameters_tabl,
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
        dataset_type=co.DatasetType.VALIDATION,
        horizon=co.HORIZON,
        window=co.BACKWARD_WINDOW
    )

    fi_test = FIDataBuilder(
        co.DATA_SOURCE + co.DATASET_FI,
        dataset_type=co.DatasetType.TEST,
        horizon=co.HORIZON,
        window=co.BACKWARD_WINDOW
    )

    train_set = FIDataset(x=fi_train.get_samples_x(), y=fi_train.get_samples_y())
    val_set = FIDataset(x=fi_val.get_samples_x(), y=fi_val.get_samples_y())
    test_set = FIDataset(x=fi_test.get_samples_x(), y=fi_test.get_samples_y())

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    fi_dm = FIDataModule(train_set, val_set, test_set, co.BATCH_SIZE, co.IS_SHUFFLE_INPUT)
    return fi_dm



def prepare_data_LOBSTER():

    train_set = LOBDataset(
        dataset_type=co.DatasetType.TRAIN,
        stocks=co.CHOSEN_STOCKS['train'].value,
        start_end_trading_day=co.CHOSEN_PERIOD.value['train']
    )

    stockName2mu, stockName2sigma = train_set.stockName2mu, train_set.stockName2sigma

    val_set = LOBDataset(
        dataset_type=co.DatasetType.VALIDATION,
        stocks=co.CHOSEN_STOCKS['train'].value,
        start_end_trading_day=co.CHOSEN_PERIOD.value['val'],
        stockName2mu=stockName2mu, stockName2sigma=stockName2sigma
    )
    test_set = LOBDataset(
        dataset_type=co.DatasetType.TEST,
        stocks=co.CHOSEN_STOCKS['test'].value,
        start_end_trading_day=co.CHOSEN_PERIOD.value['test'],
        stockName2mu=stockName2mu, stockName2sigma=stockName2sigma
    )

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    lob_dm = LOBDataModule(train_set, val_set, test_set, co.BATCH_SIZE, co.IS_SHUFFLE_INPUT)
    return lob_dm


def lunch_training():
    lunch_string = "Lunching the execution of {} on {} dataset.".format(co.CHOSEN_MODEL, co.CHOSEN_DATASET)
    if co.CHOSEN_DATASET == co.DatasetFamily.LOBSTER:
        lunch_string += ' Period: {}, Train stock: {}, Test stock: {}.'.format(co.CHOSEN_PERIOD, co.CHOSEN_STOCKS['train'], co.CHOSEN_STOCKS['test'])
    print(lunch_string)


    remote_log = None
    run_name = ''

    if co.IS_WANDB:

        wandb.init(project=co.PROJECT_NAME, entity="fin-di-sapienza")
        remote_log = wandb
        run_name = wandb.run.name

        co.BATCH_SIZE = wandb.config.batch_size
        co.IS_SHUFFLE_INPUT = wandb.config.is_shuffle
        co.OPTIMIZER = wandb.config.optimizer
        co.LEARNING_RATE = wandb.config.lr

        if co.CHOSEN_DATASET == co.DatasetFamily.LOBSTER:
            co.BACKWARD_WINDOW = wandb.config.window_size_backward
            co.FORWARD_WINDOW = wandb.config.window_size_forward
            co.LABELING_SIGMA_SCALER = wandb.config.labeling_sigma_scaler
        elif co.CHOSEN_DATASET == co.DatasetFamily.FI:
            co.BACKWARD_WINDOW = wandb.config.window_size_backward
            co.HORIZON = wandb.config.fi_horizon_k

        if co.CHOSEN_MODEL == co.Models.MLP:
            co.MLP_HIDDEN = wandb.config.hidden_mlp
            co.P_DROPOUT = wandb.config.p_dropout

        elif co.CHOSEN_MODEL == co.Models.LSTM or co.CHOSEN_MODEL == co.Models.CNNLSTM:
            co.LSTM_HIDDEN = wandb.config.lstm_hidden
            co.LSTM_N_HIDDEN = wandb.config.lstm_n_hidden
            co.MLP_HIDDEN = wandb.config.hidden_mlp
            co.P_DROPOUT = wandb.config.p_dropout

        elif co.CHOSEN_MODEL == co.Models.DAIN:
            co.DAIN_LAYER_MODE = wandb.config.dain_layer_mode
            co.MLP_HIDDEN = wandb.config.hidden_mlp
            co.P_DROPOUT = wandb.config.p_dropout

        elif co.CHOSEN_MODEL == co.Models.TRANSLOB:
            co.WEIGHT_DECAY = wandb.config.weight_decay

        elif co.CHOSEN_MODEL == co.Models.CTABL:
            co.EPOCHS = wandb.config.epochs

    data_module = pick_dataset(co.CHOSEN_DATASET)

    model = pick_model(co.CHOSEN_MODEL, data_module, remote_log)

    trainer = Trainer(
        accelerator=co.DEVICE_TYPE,
        devices=co.NUM_GPUS,
        check_val_every_n_epoch=co.VALIDATE_EVERY,
        max_epochs=co.EPOCHS,
        callbacks=[
            cbk.callback_save_model(co.CHOSEN_MODEL.value, run_name),
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
    loss_weights = None

    if chosen_model == co.Models.MLP:
        net_architecture = MLP(
            num_features=np.prod(data_module.x_shape),  # 40 * wind
            num_classes=data_module.num_classes,
            hidden_layer_dim=co.MLP_HIDDEN,
            p_dropout=co.P_DROPOUT
        )

    elif chosen_model == co.Models.CNN1:
        net_architecture = CNN1(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif chosen_model == co.Models.CNN2:
        net_architecture = CNN2(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif chosen_model == co.Models.LSTM:
        net_architecture = LSTM(
            x_shape=data_module.x_shape[1],  # 40, wind is the time
            num_classes=data_module.num_classes,
            hidden_layer_dim=co.LSTM_HIDDEN,
            hidden_mlp=co.MLP_HIDDEN,
            num_layers=co.LSTM_N_HIDDEN,
            p_dropout=co.P_DROPOUT
        )

    elif chosen_model == co.Models.CNNLSTM:
        net_architecture = CNNLSTM(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
            batch_size=co.BATCH_SIZE,
            seq_len=data_module.x_shape[0],
            hidden_size=co.LSTM_HIDDEN,
            num_layers=co.LSTM_N_HIDDEN,
            hidden_mlp=co.MLP_HIDDEN,
            p_dropout=co.P_DROPOUT
        )

    elif chosen_model == co.Models.DAIN:
        net_architecture = DAIN(
            backward_window=data_module.x_shape[0],
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
            mlp_hidden=co.MLP_HIDDEN,
            p_dropout=co.P_DROPOUT,
            mode='adaptive_avg',
            mean_lr=1e-06,
            gate_lr=1e-02,
            scale_lr=1e-02
        )

    elif chosen_model == co.Models.DEEPLOB:
        net_architecture = DeepLob(num_classes=data_module.num_classes)

    elif chosen_model == co.Models.TRANSLOB:
        net_architecture = TransLob()

    elif chosen_model == co.Models.CTABL:
        raise NotImplementedError
        net_architecture = CTABL(60, 40, 10, 10, 120, 5, 3, 1)
        constraint = WeightConstraint()
        net_architecture.apply(constraint)
        # TODO: compute loss_weights

    return NNEngine(
        model_type=chosen_model,
        neural_architecture=net_architecture,
        optimizer=co.OPTIMIZER,
        lr=co.LEARNING_RATE,
        weight_decay=co.WEIGHT_DECAY,
        loss_weights=loss_weights,
        remote_log=remote_log).to(co.DEVICE_TYPE)


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
