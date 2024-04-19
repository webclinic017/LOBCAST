# LOBCAST — Stock Price Trend Forecasting with Python

## 📈 LOBCAST 
LOBCAST is a Python-based open-source framework developed for stock market trend forecasting using Limit Order Book (LOB) 
data. The framework enables users to test deep learning models for the task of Stock Price Trend Prediction (SPTP). 
It serves as the official repository for the paper titled __LOB-Based Deep Learning Models for Stock Price Trend Prediction: 
A Benchmark Study__ [[paper](https://link.springer.com/article/10.1007/s10462-024-10715-4)].

The paper formalizes the SPTP task and the structure of LOB data. 
In the following sections, we elaborate on downloading LOB data, running stock predictions using LOBCAST with your new DL model,
model evaluation and comparison.

#### About mini-LOBCAST
This main branch represents a newer version of LOBCAST named mini-LOBCAST. It enables benchmarking models on the standard 
LOB dataset used in the literature, specifically FI-2010 [[dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)]. 
This version will be expanded to include more datasets with procedures for handling data consistently for benchmarking. 
These procedures are already available in the branch LOBCAST-v1, which will be integrated soon. We encourage the use of this version, 
while also recommending a glance at the other branch for additional implemented models and functions.

## Installing LOBCAST 

You can install LOBCAST by cloning the repository and navigating into the directory:

```
git clone https://github.com/matteoprata/LOBCAST.git
cd LOBCAST
```

Install all the required dependencies:
```
pip install -r requirements.txt
```
### Downloading LOB Dataset 
To download the FI-2010 Dataset [[dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)], follow these instructions:

1. Download the dataset into `data/datasets` by running:
```
mkdir data/datasets
cd data/datasets
wget --content-disposition "https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```
2. Unzip the file 
3. Run:
```
mv data/datasets/published data/datasets/FI-2010
```
4. Unzip `data/datasets/FI-2010/BenchmarkDatasets/BenchmarkDatasets.zip` into `data/datasets/FI-2010/BenchmarkDatasets`.

Ensure that this path exists to execute LOBCAST on this dataset:
```
data/datasets/FI-2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/*
```

### Running
Run LOBCAST locally with an MLP model and FI-2010 dataset using default settings in `src.settings`:
```
python -m src.run 
```

To customize parameters:
```
python -m src.run --SEED 42 --PREDICTION_MODEL BINCTABL --OBSERVATION_PERIOD 10 --EPOCHS_UB 20 --IS_WANDB 0
```
This will execute LOBCAST with seed 42 on the FI-2010 dataset, using the BINCTABL model, with an observation period of 10 events, for 20 epochs, running locally (not on WANDB).

The `run.py` file allows adjusting the following arguments, which are all attributes of the class `src.settings.Settings`.
```
LOBCAST 
optional arguments:
  -h, --help            show this help message and exit
  --SEED 
  --DATASET_NAME 
  --N_TRENDS 
  --PREDICTION_MODEL 
  --PREDICTION_HORIZON_UNIT 
  --PREDICTION_HORIZON_FUTURE 
  --PREDICTION_HORIZON_PAST 
  --OBSERVATION_PERIOD 
  --IS_SHUFFLE_TRAIN_SET 
  --EPOCHS_UB 
  --TRAIN_SET_PORTION 
  --VALIDATION_EVERY 
  --IS_TEST_ONLY 
  --TEST_MODEL_PATH 
  --DEVICE 
  --N_GPUs 
  --N_CPUs 
  --DIR_EXPERIMENTS 
  --IS_WANDB 
  --WANDB_SWEEP_METHOD 
  --IS_SANITY_CHECK 
```
At the end of the execution, json files containing all the statistics of the simulation and a PDF showing the performance 
of the model will be created at `data/experiments`.

### Settings
To set up a simulation in terms of randomness, choice of dataset, choice of model, observation frame of models, and 
whether to log the metrics locally or on WANDB, LOBCAST allows setting all these parameters by accessing the 
`Lobcast().SETTINGS` object. These parameters are set at the beginning of the simulation and overwritten by the arguments 
passed from the command-line interface (CLI).

### Hyperparameters
To find the right learning parameters of the model, hyperparameters can be specified in `src.hyperparameters.HPTunable`. 
By default, it contains the batch size, learning rate, and optimizer, but it can be extended by the user to specify other 
parameters. Keep in this class all the hyperparameters common to all the models. In the following we will see how to add 
model specific parameters. 

You can specify all the values that the parameters can take as ```{'values': [1, 2, 3]}```, or the min-max range as 
```{'min': 1, 'max': 100}```. 


### LOBCAST logic
The logic of the simulator can be summarized as follows:

1. Initialize LOBCAST.
2. Parse settings from the CLI.
3. Update settings.
4. Choose hyperparameter configurations.
5. Run the simulation, including data gathering, model selection, and training loop.
6. Generate a PDF with evaluation metrics.
7. Close the simulation.

The code below shows the simulation logic in `src.run`:
```
sim = LOBCAST()

setting_conf = sim.parse_cl_arguments()   # parse settings from CLI
sim.update_settings(setting_conf)         # updates simulation settings 

hparams_configs = grid_search_configurations(sim.HP_TUNABLE.__dict__)[0]   # a dict with params chosen by grid search at 0
sim.update_hyper_parameters(hparams_config)                                # update the simulation parameters 
sim.end_setup()

sim.run()         # run the simulation, data gathering, model selection and trainig loop
sim.evaluate()    # generate a pdf with the evaluation metrics
sim.close()
```


### Experimental Plans (_optional_)
Running multiple experiments sequentially is facilitated by instantiating an execution plan. Alternatively to `src.run`, 
one can run sequential tests from `src.run_batch`.

```
ep = ExecutionPlan(setup01.INDEPENDENT_VARIABLES,
                   setup01.INDEPENDENT_VARIABLES_CONSTRAINTS)
                   
setting_confs = ep.configurations()
```

An execution plan is defined in terms of `INDEPENDENT_VARIABLES` and `INDEPENDENT_VARIABLES_CONSTRAINTS`. 
These are two dictionaries. The first dictionary represents the variables to vary in a grid search. 
The `INDEPENDENT_VARIABLES_CONSTRAINTS` dictionary allows defining how the variable should be set when it does not vary, 
thus limiting the search concerning the grid search and eliminating certain configurations. The `setting_confs` contain 
the configurations to pass to `sim.update_settings(setting_conf)`, iteratively.

To run an execution plan with the dictionaries defined in `src.batch_experiments.setup01`, execute:
```
python -m src.run_batch
```

Procedures for gathering performances from different models and generating comprehensive plots for benchmarking will be added in a new update.

### Adding a New Model
To integrate a new model into LOBCAST, follow these steps:

1. Create model file: Add a `.py` file in the `src.models` directory. Define your new model class, inheriting from 
`src.models.lobcast_model.LOBCAST_model`:

```
class MyNewModel(LOBCAST_model):
    def __init__(self, input_dim, output_dim, param1, param2, param3):
        super().__init__(input_dim, output_dim)
        ...
```

2. Define hyperparameters: Optionally define the domains of your model parameters by creating a class that inherits from 
`src.hyper_parameters.HPTunable`:

```
class HP(HPTunable):
    def __init__(self):
        super().__init__()
        self.param1 = {"values": [16, 32, 64]}
        self.param2 = {"values": [.1, .5]}
``` 
3. Declare LOBCAST module: Instantiate a `src.models.lobcast_model.LOBCAST_module` to encapsulate the model and its hyperparameters:
```
mynewmodel_lm = LOBCAST_module(MLP, HP())
```

4. Declare model in Models enumerator: Add your model to the `src.models.models_classes.Models` enumerator:

```
class Models(Enum):
    NEW_MODEL = mynewmodel_lm
```

Now, you can execute the new model using the command:
```
python -m src.run --SEED 42 --PREDICTION_MODEL NEW_MODEL --IS_WANDB 0
```
Any undeclared settings will be assigned default values.


Optionally, enforce constraints on the model settings using `src.settings.Settings.check_parameters_validity`. For example:

```
constraint = (not self.PREDICTION_MODEL == cst.Models.NEW_MODEL or self.OBSERVATION_PERIOD == 10,
      f"At the moment, NEW_MODEL only allows OBSERVATION_PERIOD = 10, {self.OBSERVATION_PERIOD} given.")
```       
Ensure to add this constraint to `src.settings.Settings.check_parameters_validity.CONSTRAINTS` for enforcement.

### References
Prata, Matteo, et al. __"LOB-based deep learning models for stock price trend prediction: a benchmark study."__ Artificial Intelligence Review 57.5 (2024): 1-45.

> _The recent advancements in Deep Learning (DL) research have notably influenced the finance sector. We examine the 
> robustness and generalizability of fifteen state-of-the-art DL models focusing on Stock Price Trend Prediction (SPTP) 
> based on Limit Order Book (LOB) data. To carry out this study, we developed LOBCAST, an open-source framework that 
> incorporates data preprocessing, DL model training, evaluation and profit analysis. Our extensive experiments reveal 
> that all models exhibit a significant performance drop when exposed to new data, thereby raising questions about their 
> real-world market applicability. Our work serves as a benchmark, illuminating the potential and the limitations of current 
> approaches and providing insight for innovative solutions._
 
Link: https://link.springer.com/article/10.1007/s10462-024-10715-4 


### Acknowledgments
LOBCAST was developed by [Matteo Prata](https://github.com/matteoprata), [Giuseppe Masi](https://github.com/giuseppemasi99), [Leonardo Berti](https://github.com/LeonardoBerti00), [Andrea Coletta](https://github.com/Andrea94c), [Irene Cannistraci](https://github.com/icannistraci), Viviana Arrigoni.