# -*- coding: utf-8 -*-

import pymongo
import pandas as pd

class TradeResults:

    def __init__(self, data):
        self.raw  = data.copy()
        self.data = TradeResults.normalize(self.__clean(data))

    def train_data(self):
        return self.__train_data().drop("profit_or_loss", axis=1)

    def test_data(self):
        return self.__test_data().drop("profit_or_loss", axis=1)

    def train_profit_or_loss(self):
        return self.__train_data()["profit_or_loss"]

    def test_profit_or_loss(self):
        return self.__test_data()["profit_or_loss"]

    def __train_data(self):
        # 全データの 2/3 を訓練データとして使う。
        # トレード時の地合いの影響を分散させるため、時系列でソートしたものから均等に抜き出す。
        return self.data.loc[lambda df: df.index % 3 != 0, :]

    def __test_data(self):
        # 全データの 1/3 をテストデータとして使う。
        return self.data.loc[lambda df: df.index % 3 == 0, :]

    def __clean(self, data):
        del data['_id']
        del data['entered_at']
        del data['exited_at']
        del data['macd']
        del data['macd_signal']
        del data['macd_difference']
        del data['rsi_9']
        del data['rsi_14']
        # del data['slope_10']
        # del data['slope_25']
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

# "macd" : -0.08847474632969465, "macd_signal" : 0.008035385434950403, "macd_difference" : -0.09651013176464505,
#  "rsi_9" : 11.522491349481289, "rsi_14" : 31.641972802922773,
#  "slope_10" : -0.0478454545454676, "slope_25" : -0.0491334153846157, "slope_50" : 0.020310183913563908,
#  "ma_10_estrangement" : -1.2587818267905968, "ma_25_estrangement" : -1.5053367409321698, "ma_50_estrangement" : -2.0569388344351762,
#  "stochastics_k" : 0, "stochastics_d" : 6.0998417493885775, "stochastics_sd" : 23.55665823570315, "fast_stochastics" : -6.0998417493885775, "slow_stochastics" : -17.456816486314573,
#   "profit_or_loss" : 310, "sell_or_buy" : "buy", "entered_at" : ISODate("2006-11-25T15:00:00Z"), "exited_at" : ISODate("2006-11-26T15:00:00Z") }

    @staticmethod
    def normalize(data):
        # すべてのデータをz-scoreで正規化する
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
        cursor = collection.find({"sell_or_buy":sell_or_buy}).sort("entered_at")
        return TradeResults(pd.DataFrame(list(cursor)))
