# -*- coding: utf-8 -*-

import pymongo
import pandas as pd

class TradeResults:

    def __init__(self, data):
        self.data = self.__normalize(self.__clean(data))

    def train_data(self):
        return self.__train_data().drop("profit_or_loss", axis=1)

    def test_data(self):
        return self.__test_data().drop("profit_or_loss", axis=1)

    def train_profit_or_loss(self):
        return self.__train_data()["profit_or_loss"]

    def test_profit_or_loss(self):
        return self.__test_data()["profit_or_loss"]

    def __train_data(self):
        return self.data.loc[lambda df: df.index % 3 == 0, :]

    def __test_data(self):
        return self.data.loc[lambda df: df.index % 3 != 0, :]

    def __clean(self, data):
        del data['_id']
        del data['entered_at']
        del data['exited_at']
        del data['ma25']
        del data['ma50']
        del data['ma75']
        data['sell_or_buy'] = data['sell_or_buy'].apply(
            lambda sell_or_buy: 0 if sell_or_buy == "sell" else 1)
        return data

    def __normalize(self, data):
        for col in data.columns:
            data[col] = (data[col] - data[col].mean())/data[col].std(ddof=0)
        data = data.fillna(0)
        return data

class TradeResultsLoader:

    DB_HOST='172.17.0.1'
    DB_PORT=27017

    DB='jiji_dev'
    COLLECTION='tensorflow_example_signals'

    def retrieve_trade_data(self, sell_or_buy):
        client = pymongo.MongoClient(
            TradeResultsLoader.DB_HOST, TradeResultsLoader.DB_PORT)
        collection = client[TradeResultsLoader.DB][TradeResultsLoader.COLLECTION]
        cursor = collection.find({"sell_or_buy": sell_or_buy}).sort("entered_at")
        return TradeResults(pd.DataFrame(list(cursor)))
