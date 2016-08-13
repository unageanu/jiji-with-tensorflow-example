# -*- coding: utf-8 -*-

import pymongo
import pandas as pd

class TradeResults:

    def __init__(self, data):
        self.raw  = data.copy()
        self.data = TradeResults.normalize(TradeResults.clean(data))

    def all_data(self):
        return self.__drop_profit_or_loss(self.data)

    def train_data(self):
        return self.__drop_profit_or_loss(self.__train_data())

    def test_data(self):
        return self.__drop_profit_or_loss(self.__test_data())

    def train_profit_or_loss(self):
        return self.__train_data()["normalized_profit_or_loss"]

    def test_profit_or_loss(self):
        return self.__test_data()["normalized_profit_or_loss"]

    def train_up_down(self):
        return self.__up_down(self.__train_data()["profit_or_loss"])

    def test_up_down(self):
        return self.__up_down(self.__test_data()["profit_or_loss"])

    def __train_data(self):
        # 全データの 2/3 を訓練データとして使う。
        # トレード時の地合いの影響を分散させるため、時系列でソートしたものから均等に抜き出す。
        return self.data.loc[lambda df: df.index % 3 != 0, :]

    def __test_data(self):
        # 全データの 1/3 をテストデータとして使う。
        return self.data.loc[lambda df: df.index % 3 == 0, :]

    def __drop_profit_or_loss(self, data):
        return data.drop("profit_or_loss", axis=1).drop("normalized_profit_or_loss", axis=1)

    def __up_down(self, profit_or_loss):
        return profit_or_loss.apply(
            lambda p: pd.Series([
                1 if p >  0  else 0,
                1 if p <= 0  else 0
            ], index=['up', 'down']))

    @staticmethod
    def clean(data):
        del data['_id']
        del data['entered_at']
        del data['exited_at']
        #del data['macd']
        #del data['macd_signal']
        #del data['macd_difference']
        del data['rsi_9']
        #del data['rsi_14']
        #del data['slope_10']
        #del data['slope_25']
        #del data['slope_50']
        #del data['ma_10_estrangement']
        # del data['ma_25_estrangement']
        #del data['ma_50_estrangement']
        del data['stochastics_k']
        del data['stochastics_d']
        del data['stochastics_sd']
        del data['fast_stochastics']
        del data['slow_stochastics']
        data['sell_or_buy'] = data['sell_or_buy'].apply(
            lambda sell_or_buy: 0 if sell_or_buy == "sell" else 1)
        return data

    @staticmethod
    def normalize(data):
        # すべてのデータをz-scoreで正規化する
        for col in data.columns:
            key = 'normalized_' + col if col == 'profit_or_loss' else col
            data[key] = (data[col] - data[col].mean())/data[col].std(ddof=0)
        data = data.fillna(0)
        return data



class TradeResultsLoader:

    DB_HOST='172.17.0.1'
    DB_PORT=27017

    DB='jiji_dev'
    COLLECTION='tensorflow_example_signals_ma'

    def retrieve_trade_data(self):
        client = pymongo.MongoClient(
            TradeResultsLoader.DB_HOST, TradeResultsLoader.DB_PORT)
        collection = client[TradeResultsLoader.DB][TradeResultsLoader.COLLECTION]
        cursor = collection.find().sort("entered_at")
        return pd.DataFrame(list(cursor))
