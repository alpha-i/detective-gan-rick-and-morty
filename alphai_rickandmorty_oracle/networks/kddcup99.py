import tensorflow as tf


INIT_KERNEL = tf.random_normal_initializer(mean=0.0, stddev=0.02)
OUTPUT_DIM = 121  # Number of attributes in KDDCUP99


def leaky_relu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _impl_leaky_relu(x, alpha)
    else:
        return _impl_leaky_relu(x, alpha)


def _impl_leaky_relu(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def kddcup99_generator_network(noise, is_training):
    with tf.variable_scope('layer_1'):
        net = tf.layers.dense(noise,
                              units=64,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.nn.relu(net, name='relu')

    with tf.variable_scope('layer_2'):
        net = tf.layers.dense(net,
                              units=128,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.nn.relu(net, name='relu')

    with tf.variable_scope('layer_3'):
        net = tf.layers.dense(net,
                              units=121,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.nn.sigmoid(net, name='sigmoid')

    return net


def kddcup99_discriminator_network(inputs, keep_prob):
    is_training = False if keep_prob == 1 else True

    with tf.variable_scope('layer_1'):
        net = tf.layers.dense(inputs,
                              units=256,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = leaky_relu(net)
        net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                training=is_training)

    with tf.variable_scope('layer_2'):
        net = tf.layers.dense(net,
                              units=128,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = leaky_relu(net)
        net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                training=is_training)

    with tf.variable_scope('layer_3'):
        net = tf.layers.dense(net,
                              units=128,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = leaky_relu(net)
        net = tf.layers.dropout(net,
                                rate=0.2,
                                name='dropout',
                                training=is_training)

    intermediate_net = net

    with tf.variable_scope('layer_4'):
        net = tf.layers.dense(net,
                              units=1,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.nn.sigmoid(net, name='sigmoid')

    return net, intermediate_net
