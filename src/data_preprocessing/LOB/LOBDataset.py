

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import src.config as co

from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            dataset_type,
            stocks,
            start_end_trading_day,
            stockName2mu=dict(),
            stockName2sigma=dict(),
            num_classes=3,
            num_snapshots=100,
            one_hot_encoding=False
    ):
        self.dataset_type = dataset_type
        self.stocks = stocks
        self.start_end_trading_day = start_end_trading_day
        self.num_snapshots = num_snapshots
        self.num_classes = num_classes

        self.stockName2mu, self.stockName2sigma = stockName2mu, stockName2sigma

        stockName2databuilder = dict()

        # Choose the stock names to open to build the specific dataset.
        # No need to open all for test set, because mu/sig are pre-computed when prev opened train and dev
        stocksToOpen = None
        if dataset_type == co.DatasetType.TRAIN:
            # we open also the TEST stock(s) to determine mu and sigma for normalization, needed for all
            stocksToOpen = list(set(co.CHOSEN_STOCKS[co.STK_OPEN.TRAIN].value + co.CHOSEN_STOCKS[co.STK_OPEN.TEST].value))  # = [LYFT, NVDA]
        elif dataset_type == co.DatasetType.VALIDATION:
            stocksToOpen = co.CHOSEN_STOCKS[co.STK_OPEN.TRAIN].value  # = [LYFT]
        elif dataset_type == co.DatasetType.TEST:
            stocksToOpen = co.CHOSEN_STOCKS[co.STK_OPEN.TEST].value   # = [NVDA]

        for stock in stocksToOpen:
            path = co.DATASET_LOBSTER + f'_data_dwn_48_332__{stock}_{co.CHOSEN_PERIOD.value["train"][0]}_{co.CHOSEN_PERIOD.value["test"][1]}_10'

            normalization_mean = stockName2mu[stock] if stock in stockName2mu else None
            normalization_std = stockName2sigma[stock] if stock in stockName2sigma else None

            print(dataset_type, '\t', stocks, '\t', stock, '\t', start_end_trading_day, '\t', normalization_mean, '\t', normalization_std, '\t', path)

            databuilder = LOBSTERDataBuilder(
                stock,
                path,
                dataset_type=dataset_type,
                start_end_trading_day=start_end_trading_day,
                crop_trading_day_by=60*30,
                window_size_forward=co.FORWARD_WINDOW,
                window_size_backward=co.BACKWARD_WINDOW,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                num_snapshots=num_snapshots,
                label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
                is_data_preload=co.IS_DATA_PRELOAD
            )

            self.stockName2mu[stock], self.stockName2sigma[stock] = databuilder.normalization_means, databuilder.normalization_stds
            stockName2databuilder[stock] = databuilder

        print('stockName2mu:', self.stockName2mu)
        print('stockName2sigma:', self.stockName2sigma)

        self.stock2orderNlen = dict()
        self.x, self.y, self.stock_sym_name = list(), list(), list()
        for stock in self.stocks:
            print("Handling", stock, "for dataset", dataset_type)
            databuilder = stockName2databuilder[stock]
            samplesX, samplesY = databuilder.get_samples_x(), databuilder.get_samples_y()
            self.x.extend(samplesX)
            self.y.extend(samplesY)
            self.stock_sym_name.extend([stock]*len(samplesY))

        self.x = torch.from_numpy(np.array(self.x)).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(self.y)).type(torch.LongTensor)

        self.x_shape = tuple(self.x[0].shape)

        # print(len(self.x), len(self.y), len(self.stock_sym_name))
        print()
        print()

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index], self.y[index], self.stock_sym_name[index]
