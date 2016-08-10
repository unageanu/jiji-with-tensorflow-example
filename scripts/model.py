# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Model:
    HIDDEN_UNIT_SIZE = 32
    COLUMN_SIZE = 7

    def __init__(self, context):
        self.context = context
        self.__setup_placeholder()
        self.__setup_model()
        self.__setup_ops()

    def __enter__(self):
        self.session = tf.Session()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
        return False

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

    def __setup_placeholder(self):
        column_size = Model.COLUMN_SIZE
        self.trade_data = tf.placeholder("float", [None, column_size])
        self.profit_or_loss = tf.placeholder("float", [None, 1])
        self.loss_label = tf.placeholder("string")

    def __setup_model(self):
        column_size = Model.COLUMN_SIZE
        with tf.name_scope('hidden1') as scope:
            hidden1_weight = tf.Variable(tf.truncated_normal([column_size, Estimator.HIDDEN_UNIT_SIZE], stddev=0.1), name='hidden1_weight')
            hidden1_bias = tf.Variable(tf.constant(0.1, shape=[Estimator.HIDDEN_UNIT_SIZE]), name='hidden1_bias')
            hidden1_output = tf.nn.relu(tf.matmul(self.trade_data, hidden1_weight) + hidden1_bias)
        with tf.name_scope('output') as scope:
            output_weight = tf.Variable(tf.truncated_normal([Estimator.HIDDEN_UNIT_SIZE, 1], stddev=0.1), name='output_weight')
            output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name='output_bias')
            output = tf.matmul(hidden1_output, output_weight) + output_bias
        self.model = output

    def __setup_ops(self):
        self.loss_op = self.__loss()
        with tf.name_scope('training') as scope:
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_op = optimizer.minimize(self.loss_op)
        self.merge_summaries = tf.merge_summary([self.loss_summary])

    def __loss(self):
         with tf.name_scope('loss') as scope:
            loss = tf.nn.l2_loss(tf.nn.l2_normalize(self.model,0) - tf.nn.l2_normalize(self.profit_or_loss,0))
            self.loss_summary = tf.scalar_summary(self.loss_label, loss)
         return loss



class Trainer(Model):

    def train(self, steps, data):
        best = float("inf")
        self.__prepare_train(self.session)
        for i in range(steps):
            loss_train = self.__do_train(self.session, i, data)
            if loss_train < best:
                best = loss_train
                self.__update_best_match(self.session, data)
            if i %100 == 0:
                self.__add_summary(self.session, i, data)
                self.__print_status(self.session, i, loss_train, data)

    def __prepare_train(self, session):
        self.summary_writer = tf.train.SummaryWriter('logs', graph_def=session.graph_def)
        session.run(tf.initialize_all_variables())

    def __do_train(self, session, i, data):
        loss_train = session.run(self.loss_op, feed_dict=self.train_feed_dict(data))
        session.run(self.train_op, feed_dict=self.train_feed_dict(data))
        return loss_train

    def __update_best_match(self, session, data):
        # 現在のモデルを利用して、訓練データ、テストデータの利益を出力する
        self.best_match_train = session.run(self.model, feed_dict=self.train_feed_dict(data))
        self.best_match_test  = session.run(self.model, feed_dict=self.test_feed_dict(data))

    def __add_summary(self, session, i, data):
        summary_str = session.run(self.merge_summaries, feed_dict=self.train_feed_dict(data))
        summary_str += session.run(self.merge_summaries, feed_dict=self.test_feed_dict(data))
        self.summary_writer.add_summary(summary_str, i)

    def __print_status(self, session, i, loss_train, data):
        # 現在のモデルを利用して推移した利益と実際の利益の相関を、訓練データ、テストデータそれぞれで計算し出力する
        pearson_train = np.corrcoef(self.best_match_train.flatten(), data.train_profit_or_loss().values.flatten())
        pearson_test  = np.corrcoef(self.best_match_test.flatten(), data.test_profit_or_loss().values.flatten())
        print 'step {} train loss = {} ,train corrcoef={} ,test corrcoef={} '.format(
            i, loss_train, pearson_train[0][1], pearson_test[0][1])

    def train_feed_dict(self, data):
        return {
            self.trade_data: data.train_data(),
            self.profit_or_loss: self.__reshape(data.train_profit_or_loss()),
            self.loss_label: self.context + "_train"
        }

    def test_feed_dict(self, data):
        return {
            self.trade_data: data.test_data(),
            self.profit_or_loss: self.__reshape(data.test_profit_or_loss()),
            self.loss_label: self.context + "_test"
        }

    def __reshape(self, profit_or_loss):
        return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)



class Estimator(Model):

    def estimate( self, data ):
        return self.session.run(self.model, feed_dict=self.estimate_feed_dict(data))

    def estimate_feed_dict(self, data):
        return {
            self.trade_data: data
        }
