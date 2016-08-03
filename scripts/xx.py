# -*- coding: utf-8 -*-

import sklearn
import tensorflow as tf
from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()

#データをロード
print "load diabetes data"
data = diabetes["data"].astype(np.float32)
target = diabetes['target'].astype(np.float32).reshape(len(diabetes['target']), 1)
#学習データとテストデータに分割
N=342
x_train, x_test = np.vsplit(data, [N])
y_train, y_test = np.vsplit(target, [N])
N_test = y_test.size

x= tf.placeholder("float",shape=[None,10])

# 1層目　入力10 出力256
with tf.name_scope('l1') as scope:
    weightl1 = tf.Variable(tf.truncated_normal([10, 256], stddev=0.1),name="weightl1")
    biasel1 = tf.Variable(tf.constant(1.0, shape=[256]), name="biasel1")
    outputl1=tf.nn.relu(tf.matmul(x,weightl1) + biasel1)

# 2層目 入力256 出力1
with tf.name_scope('l2') as scope:
    weightl2 = tf.Variable(tf.truncated_normal([256, 1], stddev=0.1),name="weightl2")
    biasel2 = tf.Variable(tf.constant(1.0, shape=[1]), name="biasel2")
    outputl2=tf.nn.relu(tf.matmul(outputl1,weightl2) + biasel2)



"""
誤差計算のための関数
MSEで誤差を算出
"""
def loss(output):
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(output - y_train))
    return loss


loss_op = loss(outputl2)
optimizer = tf.train.AdagradOptimizer(0.04)
train_step = optimizer.minimize(loss_op)

#誤差の記録
best = float("inf")

# 初期化
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    # initする
    sess.run(init_op)
    for i in range(20001):
        loss_train = sess.run(loss_op, feed_dict={x:x_train})
        sess.run(train_step, feed_dict={x:x_train})
        if loss_train < best:
            best = loss_train
            best_match = sess.run(outputl2, feed_dict={x:x_test})
        if i %1000 == 0:
            print "step {}".format(i)
            pearson = np.corrcoef(best_match.flatten(), y_test.flatten())
            print 'train loss = {} ,test corrcoef={}'.format(best,pearson[0][1])
