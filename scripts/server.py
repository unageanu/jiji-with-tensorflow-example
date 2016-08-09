# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, jsonify, request
from trade_results_loader import *
from model import *

loader = TradeResultsLoader()

sell_estimator = Estimator("sell")
sell_estimator.__enter__()
sell_estimator.restore("../data/sell.ckpt")

buy_estimator = Estimator("buy")
buy_estimator.__enter__()
buy_estimator.restore("../data/buy.ckpt")

sell_data = loader.retrieve_trade_data("sell").train_data()
buy_data  = loader.retrieve_trade_data("buy").train_data()


def read_data(body):
    sell_or_buy = body["sell_or_buy"]
    del body["sell_or_buy"]
    data = pd.DataFrame({k: [v] for k, v in body.items()}).append(
        sell_data if sell_or_buy == "sell" else buy_data)
    return (data, sell_or_buy)

# webapp
app = Flask(__name__)

@app.route('/api/estimator', methods=['POST'])
def estimate():
    data, sell_or_buy = read_data(request.json)
    estimator = sell_estimator if sell_or_buy == "sell" else buy_estimator
    results = estimator.estimate(TradeResults.normalize(data))
    return jsonify(result=float(results[0][0]))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
