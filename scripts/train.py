# -*- coding: utf-8 -*-

from trade_results_loader import *
from model import *

loader = TradeResultsLoader()

for sell_or_buy in ['sell', 'buy']:
    data = loader.retrieve_trade_data(sell_or_buy)
    with Trainer(sell_or_buy) as trainer:
        trainer.train(10000, data)
        trainer.save("../data/" + sell_or_buy +  ".ckpt")
