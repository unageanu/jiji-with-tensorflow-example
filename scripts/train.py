# -*- coding: utf-8 -*-

from trade_results_loader import *
from trainer import *

loader = TradeResultsLoader()
data = loader.retrieve_trade_data("buy")


print data.train_data()

Trainer(data).train(10000)
