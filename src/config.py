
import numpy as np
import os

from src.constants import LearningHyperParameter
import src.constants as cst
from src.metrics.metrics_log import Metrics
from datetime import date, datetime

np.set_printoptions(suppress=True)


class Settings:
    def __init__(self):

        self.SEED: int = 0
        self.IS_HPARAM_SEARCH: bool = False  # else preload from json file
        self.IS_CPU: bool = False

        self.DATASET_NAME: cst.DatasetFamily = cst.DatasetFamily.FI
        self.STOCK_TRAIN_VAL: str = "AAPL"
        self.STOCK_TEST: str = "AAPL"

        self.N_TRENDS = 3
        self.PREDICTION_MODEL = cst.ModelsClass.MLP.value
        self.PREDICTION_HORIZON_UNIT: cst.UnitHorizon = cst.UnitHorizon.EVENTS
        self.PREDICTION_HORIZON_FUTURE: int = 5
        self.PREDICTION_HORIZON_PAST: int = 10
        self.OBSERVATION_PERIOD: int = 10
        self.IS_SHUFFLE_TRAIN_SET = None

        self.TRAIN_SET_PORTION = .8

        self.MODE = 'train'
        self.MODEL_PATH: str = ""
        self.DEVICE = ""


class ConfigHP:
    def add_hyperparameters(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_hyperparameter(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        return self.__dict__


class ConfigHPTuned(ConfigHP):
    pass


class ConfigHPTunable(ConfigHP):
    def __init__(self):
        self.BATCH_SIZE = [64, 55]
        self.LEARNING_RATE = [0.01]
        self.EPOCHS_UB = [50]
        self.OPTIMIZER = ["SGD"]


class Configuration(Settings):
    """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    def __init__(self, run_name_prefix=None):
        super().__init__()

        self.TUNABLE = ConfigHPTunable()
        self.TUNED = ConfigHPTuned()

        # assigns values to
        for key, value in self.PREDICTION_MODEL.tunable_parameters.items():
            self.TUNABLE.add_hyperparameter(key, value)

        for key, _ in self.TUNABLE.__dict__.items():
            self.TUNED.add_hyperparameter(key, None)

        # DO TUNING
        for key, value in self.TUNABLE.__dict__.items():
            self.TUNED.__setattr__(key, value[0])

        print(self.TUNED.__dict__)
        # exit()

        self.IS_DEBUG = False
        self.IS_TEST_ONLY = False

        self.RUN_NAME_PREFIX = self.assign_prefix(prefix=run_name_prefix, is_debug=self.IS_DEBUG)
        self.setup_all_directories(self.RUN_NAME_PREFIX, self.IS_DEBUG, self.IS_TEST_ONLY)

        self.RANDOM_GEN_DATASET = None
        self.VALIDATE_EVERY = 1

        self.IS_DATA_PRELOAD = True
        self.INSTANCES_LOWER_BOUND = 1000  # under-sampling must have at least INSTANCES_LOWER_BOUND instances

        self.TRAIN_SPLIT_VAL = .8  # FI only
        self.META_TRAIN_VAL_TEST_SPLIT = (.7, .15, .15)  # META Only

        self.CHOSEN_PERIOD = cst.Periods.FI

        self.CHOSEN_STOCKS = {
            cst.STK_OPEN.TRAIN: cst.Stocks.FI,
            cst.STK_OPEN.TEST: cst.Stocks.FI
        }

        self.IS_WANDB = 0

        self.SWEEP_METHOD = 'grid'  # 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.WANDB_SWEEP_NAME = None

        self.SWEEP_METRIC = {
            'goal': 'maximize',
            'name': None
        }

        self.TARGET_DATASET_META_MODEL = cst.DatasetFamily.LOB
        self.JSON_DIRECTORY = ""

        self.EARLY_STOPPING_METRIC = None

        self.METRICS_JSON = Metrics(self)
        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.01
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS_UB] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.SGD.value
        self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0
        self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08  # default value for ADAM
        self.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM] = 0.9

        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_SNAPSHOTS] = 100
        # LOB way to label to measure percentage change LOB = HORIZON
        self.HYPER_PARAMETERS[LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.NONE.value
        self.HYPER_PARAMETERS[LearningHyperParameter.FORWARD_WINDOW] = cst.WinSize.NONE.value
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True
        self.HYPER_PARAMETERS[LearningHyperParameter.LABELING_SIGMA_SCALER] = .9
        self.HYPER_PARAMETERS[LearningHyperParameter.FI_HORIZON] = cst.FI_Horizons.K10.value  # in FI = FORWARD_WINDOW  = k in papers

        self.HYPER_PARAMETERS[LearningHyperParameter.MLP_HIDDEN] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.RNN_HIDDEN] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.META_HIDDEN] = 16

        self.HYPER_PARAMETERS[LearningHyperParameter.RNN_N_HIDDEN] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.DAIN_LAYER_MODE] = 'full'
        self.HYPER_PARAMETERS[LearningHyperParameter.P_DROPOUT] = 0
        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_RBF_NEURONS] = 16

    def dynamic_config_setup(self):
        # sets the name of the metric to optimize
        self.SWEEP_METRIC['name'] = "{}_{}_{}".format(cst.ModelSteps.VALIDATION_MODEL.value, self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)
        self.EARLY_STOPPING_METRIC = "{}_{}_{}".format(cst.ModelSteps.VALIDATION_EPOCH.value, self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)

        self.WANDB_SWEEP_NAME = self.cf_name_format().format(
            self.PREDICTION_MODEL.name,
            self.SEED,
            self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
            self.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
            self.DATASET_NAME.value,
            self.CHOSEN_PERIOD.name,
            self.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
            self.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
            self.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        )

        if not self.IS_HPARAM_SEARCH and not self.IS_WANDB:
            self.WANDB_RUN_NAME = self.WANDB_SWEEP_NAME

    @staticmethod
    def cf_name_format(ext=""):
        return "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}" + ext

    @staticmethod
    def setup_all_directories(prefix, is_debug, is_test):
        """
        Creates two folders:
            (1) data.experiments.LOB-CLASSIFIERS-(PREFIX) for the jsons with the stats
            (2) data.saved_models.LOB-CLASSIFIERS-(PREFIX) for the models
        """

        if not is_test:
            cst.PROJECT_NAME = cst.PROJECT_NAME.format(prefix)
            cst.DIR_SAVED_MODEL = cst.DIR_SAVED_MODEL.format(prefix) + "/"
            cst.DIR_EXPERIMENTS = cst.DIR_EXPERIMENTS.format(prefix) + "/"

            # create the paths for the simulation if they do not exist already
            paths = ["data", cst.DIR_SAVED_MODEL, cst.DIR_EXPERIMENTS]
            for p in paths:
                if not os.path.exists(p):
                    os.makedirs(p)

    @staticmethod
    def assign_prefix(prefix, is_debug):
        if is_debug:
            return "debug"
        elif prefix is not None:
            return prefix
        else:
            return datetime.now().strftime("%Y-%m-%d+%H-%M-%S")
