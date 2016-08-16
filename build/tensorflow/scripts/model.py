# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Model:
    HIDDEN_UNIT_SIZE  = 32
    HIDDEN_UNIT_SIZE2 = 16
    COLUMN_SIZE = 9

    def __init__(self):
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
        self.trade_data   = tf.placeholder("float", [None, column_size])
        self.actual_class = tf.placeholder("float", [None, 2])
        self.keep_prob    = tf.placeholder("float")
        self.label        = tf.placeholder("string")

    def __setup_model(self):
        column_size = Model.COLUMN_SIZE
        w1 = tf.Variable(tf.truncated_normal([column_size, Estimator.HIDDEN_UNIT_SIZE], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[Estimator.HIDDEN_UNIT_SIZE]))
        h1 = tf.nn.relu(tf.matmul(self.trade_data, w1) + b1)

        w2 = tf.Variable(tf.truncated_normal([Estimator.HIDDEN_UNIT_SIZE, Estimator.HIDDEN_UNIT_SIZE2], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[Estimator.HIDDEN_UNIT_SIZE2]))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        h2_drop = tf.nn.dropout(h2, self.keep_prob)
        w2 = tf.Variable(tf.truncated_normal([Estimator.HIDDEN_UNIT_SIZE2, 2], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[2]))
        self.output = tf.nn.softmax(tf.matmul(h2_drop, w2) + b2)

    def __setup_ops(self):
        cross_entropy = -tf.reduce_sum(self.actual_class * tf.log(self.output))
        self.summary = tf.scalar_summary(self.label, cross_entropy)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        self.merge_summaries = tf.merge_summary([self.summary])
        correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.actual_class,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


class Trainer(Model):

    def train(self, steps, data):
        self.__prepare_train(self.session)
        for i in range(steps):
            self.__do_train(self.session, i, data)
            if i %100 == 0:
                self.__add_summary(self.session, i, data)
                self.__print_status(self.session, i, data)

    def __prepare_train(self, session):
        self.summary_writer = tf.train.SummaryWriter('logs', graph_def=session.graph_def)
        session.run(tf.initialize_all_variables())

    def __do_train(self, session, i, data):
        session.run(self.train_op, feed_dict=self.train_feed_dict(data))

    def __add_summary(self, session, i, data):
        summary_str = session.run(self.merge_summaries, feed_dict=self.train_feed_dict(data))
        summary_str += session.run(self.merge_summaries, feed_dict=self.test_feed_dict(data))
        self.summary_writer.add_summary(summary_str, i)

    def __print_status(self, session, i, data):
        # 現在のモデルを利用して推移した利益と実際の利益の相関を、訓練データ、テストデータそれぞれで計算し出力する
        train_accuracy = session.run(self.accuracy, feed_dict=self.train_feed_dict(data))
        test_accuracy  = session.run(self.accuracy, feed_dict=self.test_feed_dict(data))
        print 'step {} ,train_accuracy={} ,test_accuracy={} '.format(
            i, train_accuracy, test_accuracy)

    def train_feed_dict(self, data):
        return {
            self.trade_data: data.train_data(),
            self.actual_class: data.train_up_down(),
            self.keep_prob: 0.8,
            self.label: "train"
        }

    def test_feed_dict(self, data):
        return {
            self.trade_data: data.test_data(),
            self.actual_class: data.test_up_down(),
            self.keep_prob: 1,
            self.label: "test"
        }

    def __reshape(self, profit_or_loss):
        return profit_or_loss.values.reshape(len(profit_or_loss.values), 1)



class Estimator(Model):

    def estimate( self, data ):
        return self.session.run(tf.argmax(self.output,1), feed_dict=self.estimate_feed_dict(data))

    def estimate_feed_dict(self, data):
        return {
            self.trade_data: data,
            self.keep_prob: 1
        }
