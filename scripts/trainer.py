# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Trainer:

    def __init__(self, data):
        self.data = data
        self.__setup_placeholder()
        self.__setup_model()
        self.__setup_ops()

    def train(self):
        best = float("inf")

        print self.__reshape(self.data.train_profit_or_loss())

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(1001):
                loss_train = sess.run(self.loss_op, feed_dict=self.train_feed_dict())
                sess.run(self.train_op, feed_dict=self.train_feed_dict())
                if loss_train < best:
                    best = loss_train
                    best_match = sess.run(self.model, feed_dict=self.test_feed_dict())
                if i %100 == 0:
                    print sess.run(self.weights)
                    print "step {}".format(i)
                    print loss_train
                    #print best_match
                    #pearson = np.corrcoef(best_match.flatten(), test_profit_or_loss.flatten())
                    #print 'train loss = {} ,test corrcoef={}'.format(best,pearson[0][1])

    def train_feed_dict(self):
        return {
            self.trade_data: self.data.train_data(),
            self.profit_or_loss: self.__reshape(self.data.train_profit_or_loss())
        }

    def test_feed_dict(self):
        return {
            self.trade_data: self.data.test_data(),
            self.profit_or_loss: self.__reshape(self.data.test_profit_or_loss())
        }

    def __setup_placeholder(self):
        column_size = self.__column_size()
        self.trade_data = tf.placeholder("float", [None, column_size])
        self.profit_or_loss = tf.placeholder("float", [None, 1])

    def __column_size(self):
        return len(self.data.train_data().columns)


    def __setup_model(self):
        column_size = self.__column_size()
        self.weights = tf.Variable(tf.zeros([column_size, 1]))
        self.biases = tf.Variable(tf.zeros([1]))
        self.model = tf.nn.relu(tf.matmul(self.trade_data, self.weights) + self.biases)

    def __setup_ops(self):
        self.loss_op = self.__loss()
        optimizer = tf.train.AdagradOptimizer(0.0001)
        self.train_op = optimizer.minimize(self.loss_op)

    def __loss(self):
        return tf.reduce_mean(tf.square(self.model - self.profit_or_loss))

    def __reshape(self, profit_or_loss):
        return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)
