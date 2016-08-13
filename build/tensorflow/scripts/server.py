# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, jsonify, request
from trade_results_loader import *
from model import *

loader = TradeResultsLoader()

estimator = Estimator()
estimator.__enter__()
estimator.restore("./model.ckpt")

trade_data  = loader.retrieve_trade_data()

# webapp
app = Flask(__name__)

@app.route('/api/estimator', methods=['POST'])
def estimate():
    data = pd.DataFrame({k: [v] for k, v in request.json.items()}).append(trade_data)
    results = estimator.estimate(TradeResults(data).all_data().iloc[[0]])
    return jsonify(result=("up" if results[0] == 0 else "down"))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
