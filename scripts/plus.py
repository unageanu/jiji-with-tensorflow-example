import tensorflow as tf

def x2_plus_b(x, b):
    _x = tf.constant(x)
    _b = tf.constant(b)
    result = tf.square(_x)
    result = tf.add(result, _b)
    return result

with tf.Session() as sess:
    result = sess.run([x2_plus_b(2., 3.)])
    print result


p_x = tf.placeholder(tf.types.float32)
p_b = tf.placeholder(tf.types.float32)
p_x2_plus_b = tf.add(tf.square(p_x), p_b)

with tf.Session() as sess:
    result = sess.run([p_x2_plus_b], feed_dict={p_x: [2.], p_b: [3.]})
    print result
