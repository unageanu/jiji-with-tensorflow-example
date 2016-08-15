# -*- coding: utf-8 -*-

from trade_results_loader import *
from model import *

loader = TradeResultsLoader()
data = TradeResults(loader.retrieve_trade_data())

with Trainer() as trainer:
    trainer.train(10000, data)
    trainer.save("./model.ckpt")
