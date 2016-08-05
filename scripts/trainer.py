# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Trainer:

    HIDDEN_UNIT_SIZE = 256

    def __init__(self, data):
        self.data = data
        self.__setup_placeholder()
        self.__setup_model()
        self.__setup_ops()

    def train(self, steps):
        best = float("inf")
        with tf.Session() as session:
            self.__prepare_train(session)

            for i in range(steps):
                loss_train = self.__do_train(session, i)

                if loss_train < best:
                    best = loss_train
                    self.__update_best_match(session)
                if i %100 == 0:
                    self.__print_status(session, i, loss_train)


    def __prepare_train(self, session):
        self.summary_writer = tf.train.SummaryWriter('logs', graph_def=session.graph_def)
        session.run(tf.initialize_all_variables())

    def __do_train(self, session, i):
        loss_train = session.run(self.loss_op, feed_dict=self.train_feed_dict())
        session.run(self.train_op, feed_dict=self.train_feed_dict())

        summary_str = session.run(self.merge_summaries, feed_dict=self.train_feed_dict())
        summary_str += session.run(self.merge_summaries, feed_dict=self.test_feed_dict())
        self.summary_writer.add_summary(summary_str, i)

        return loss_train

    def __update_best_match(self, session):
        # 現在のモデルを利用して、訓練データ、テストデータの利益を出力する
        self.best_match_train = session.run(self.model, feed_dict=self.train_feed_dict())
        self.best_match_test  = session.run(self.model, feed_dict=self.test_feed_dict())

    def __print_status(self, session, i, loss_train):
        # 現在のモデルを利用して推移した利益と実際の利益の相関を、訓練データ、テストデータそれぞれで計算し出力する
        pearson_train = np.corrcoef(self.best_match_train.flatten(), self.data.train_profit_or_loss().values.flatten())
        pearson_test  = np.corrcoef(self.best_match_test.flatten(), self.data.test_profit_or_loss().values.flatten())
        print 'step {} train loss = {} ,train corrcoef={} ,test corrcoef={} '.format(
            i, loss_train, pearson_train[0][1], pearson_test[0][1])

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
        self.model = tf.nn.l2_normalize(output, 0)

    def __setup_ops(self):
        self.loss_op = self.__loss()
        with tf.name_scope('training') as scope:
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss_op)
        self.merge_summaries = tf.merge_all_summaries()

    def __loss(self):
         with tf.name_scope('loss') as scope:
            loss = tf.nn.l2_loss(self.model - tf.nn.l2_normalize(self.profit_or_loss, 0))
            tf.scalar_summary(self.loss_label, loss)
         return loss

    def __reshape(self, profit_or_loss):
        return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)
