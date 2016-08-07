# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, jsonify, request
from trade_results_loader import *
from model import *

# x = tf.placeholder("float", [None, 784])
# sess = tf.Session()
#
# with tf.variable_scope("simple"):
#     y1, variables = model.simple(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/simple.ckpt")
# def simple(input):
#     return sess.run(y1, feed_dict={x: input}).flatten().tolist()
#
# with tf.variable_scope("convolutional"):
#     keep_prob = tf.placeholder("float")
#     y2, variables = model.convolutional(x, keep_prob)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/convolutional.ckpt")
# def convolutional(input):
#     return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

loader = TradeResultsLoader()

sell_estimator = Estimator("sell")
sell_estimator.__enter__()
sell_estimator.restore("../data/sell.ckpt")

buy_estimator = Estimator("buy")
buy_estimator.__enter__()
buy_estimator.restore("../data/buy.ckpt")

def read_data(body):
    p body
    sell_or_buy = body["sell_or_buy"]
    del body["sell_or_buy"]
    base = loader.retrieve_trade_data(sell_or_buy)
    return pd.DataFrame(body).append(base)

# webapp
app = Flask(__name__)

@app.route('/api/estimator', methods=['POST'])
def estimate():
    data = read_data(request.json)
    results = estimator.estimate(TradeResults.normalize(data))
    return jsonify(results=results[0])

if __name__ == '__main__':
    app.run()
