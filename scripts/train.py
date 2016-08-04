# -*- coding: utf-8 -*-

from trade_results_loader import *
from trainer import *

loader = TradeResultsLoader()
data = loader.retrieve_trade_data("sell")


print data.train_data()
Trainer(data).train()

#
# DB_HOST='172.17.0.1'
# DB_PORT=27017
#
# DB='jiji_dev'
# COLLECTION='tensorflow_example_signals'
#
# def retrieve_trade_data(sell_or_buy):
#     client = pymongo.MongoClient(DB_HOST, DB_PORT)
#     collection = client[DB][COLLECTION]
#     cursor = collection.find({"sell_or_buy": sell_or_buy}).sort("entered_at")
#     return pd.DataFrame(list(cursor))
#
#
# def extract_train_data(data):
#     return data.loc[lambda df: df.index % 3 == 0, :]
#
#
# def extract_test_data(data):
#     return data.loc[lambda df: df.index % 3 != 0, :]
#
# def extract_profit_or_loss(data):
#     profit_or_loss = data.profit_or_loss
#     del data['profit_or_loss']
#     return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)
#
# def clean(data):
#     del data['_id']
#     del data['entered_at']
#     del data['exited_at']
#     data['sell_or_buy'] = data['sell_or_buy'].apply(
#         lambda sell_or_buy: 0 if sell_or_buy == "sell" else 1)
#     return data

#
# def create_network(dataset, trade_data):
#     column_size = len(dataset.columns)
#
#     weights = tf.Variable(tf.zeros([column_size, 1]))
#     biases = tf.Variable(tf.zeros([1]))
#
#     return tf.nn.relu(tf.matmul(trade_data, weights) + biases)
#
#
# def loss(model, profit_or_loss):
#     return tf.reduce_mean(tf.square(model - profit_or_loss))
#
# data = clean(retrieve_trade_data("sell"))
#
# train_data = extract_train_data(data)
# test_data = extract_test_data(data)
#
# train_profit_or_loss = extract_profit_or_loss(train_data)
# test_profit_or_loss = extract_profit_or_loss(test_data)
#
# trade_data = tf.placeholder("float", [None, len(train_data.columns)])
# profit_or_loss = tf.placeholder("float", [None, 1])
#
# model = create_network(train_data, trade_data)
#
# loss_op = loss(model, profit_or_loss)
# optimizer = tf.train.AdagradOptimizer(0.0001)
# train_step = optimizer.minimize(loss_op)
#
#
# best = float("inf")
# train_fd={ trade_data: train_data, profit_or_loss: train_profit_or_loss }
# test_fd ={ trade_data: test_data,  profit_or_loss: test_profit_or_loss }
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     for i in range(1001):
#         loss_train = sess.run(loss_op, feed_dict=train_fd)
#         sess.run(train_step, feed_dict=train_fd)
#         if loss_train < best:
#             best = loss_train
#             best_match = sess.run(model, feed_dict=test_fd)
#         if i %100 == 0:
#             print "step {}".format(i)
#             print loss_train
#             #print best_match
#             #pearson = np.corrcoef(best_match.flatten(), test_profit_or_loss.flatten())
#             #print 'train loss = {} ,test corrcoef={}'.format(best,pearson[0][1])
#
#
#
#
# #print(extract_test_data(data))
# #print(data['entered_at'])
# #df.ix[df.apply(sum, axis=1) >= 10, :]
