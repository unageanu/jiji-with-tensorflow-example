# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Trainer:

    HIDDEN_UNIT_SIZE = 32

    def __init__(self, data):
        self.data = data
        self.__setup_placeholder()
        self.__setup_model()
        self.__setup_ops()

    def train(self):
        best = float("inf")
        merge_summaries = tf.merge_all_summaries()
        with tf.Session() as sess:
            summary_writer = tf.train.SummaryWriter('logs', graph_def=sess.graph_def)
            sess.run(tf.initialize_all_variables())
            for i in range(10001):
                loss_train = sess.run(self.loss_op, feed_dict=self.train_feed_dict())
                sess.run(self.train_op, feed_dict=self.train_feed_dict())
                if loss_train < best:
                    best = loss_train
                    best_match = sess.run(self.model, feed_dict=self.test_feed_dict())
                if i %100 == 0:
                    print "step {}".format(i)
                    print loss_train
                    summary_str = sess.run(merge_summaries, feed_dict=self.train_feed_dict())
                    summary_str += sess.run(merge_summaries, feed_dict=self.test_feed_dict())
                    summary_writer.add_summary(summary_str, i)

                    #pearson = np.corrcoef(best_match.flatten(), test_profit_or_loss.flatten())
                    #print 'train loss = {} ,test corrcoef={}'.format(best,pearson[0][1])

            print best_match

    def train_feed_dict(self):
        return {
            self.trade_data: self.data.train_data(),
            self.profit_or_loss: self.__reshape(self.data.train_profit_or_loss()),
            self.loss_label: "train"
        }

    def test_feed_dict(self):
        return {
            self.trade_data: self.data.test_data(),
            self.profit_or_loss: self.__reshape(self.data.test_profit_or_loss()),
            self.loss_label: "test"
        }

    def __setup_placeholder(self):
        column_size = self.__column_size()
        self.trade_data = tf.placeholder("float", [None, column_size])
        self.profit_or_loss = tf.placeholder("float", [None, 1])
        self.loss_label = tf.placeholder("string")

    def __column_size(self):
        return len(self.data.train_data().columns)


    def __setup_model(self):
        column_size = self.__column_size()
        with tf.name_scope('hidden1') as scope:
            hidden1_weight = tf.Variable(tf.truncated_normal([column_size, Trainer.HIDDEN_UNIT_SIZE], stddev=0.1), name='hidden1_weight')
            hidden1_bias = tf.Variable(tf.constant(0.1, shape=[Trainer.HIDDEN_UNIT_SIZE]), name='hidden1_bias')
            hidden1_output = tf.nn.relu(tf.matmul(self.trade_data, hidden1_weight) + hidden1_bias)
        with tf.name_scope('output') as scope:
            output_weight = tf.Variable(tf.truncated_normal([Trainer.HIDDEN_UNIT_SIZE, 1], stddev=0.1), name='output_weight')
            output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name='output_bias')
            output = tf.matmul(hidden1_output, output_weight) + output_bias
        self.model = output #tf.nn.l2_normalize(output, 0)

    def __setup_ops(self):
        self.loss_op = self.__loss()
        with tf.name_scope('training') as scope:
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss_op)

    def __loss(self):
         with tf.name_scope('loss') as scope:
            #loss = tf.nn.l2_loss(self.model - tf.nn.l2_normalize(self.profit_or_loss, 0))
            loss = tf.nn.l2_loss(self.model - self.profit_or_loss)
            tf.scalar_summary(self.loss_label, loss)
         return loss

    def __reshape(self, profit_or_loss):
        return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)
