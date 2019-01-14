import tensorflow as tf


INIT_KERNEL = tf.random_normal_initializer(mean=0.0, stddev=0.02)
OUTPUT_DIM = 784  # Number of pixels in MNIST (28*28)


def leaky_relu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _impl_leaky_relu(x, alpha)
    else:
        return _impl_leaky_relu(x, alpha)


def _impl_leaky_relu(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def mnist_generator_network(noise, is_training):
    with tf.variable_scope('layer_1'):
        net = tf.layers.dense(noise,
                              units=1024,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.layers.batch_normalization(net,
                                            training=is_training,
                                            name='batch_normalization')
        net = tf.nn.relu(net, name='relu')

    with tf.variable_scope('layer_2'):
        net = tf.layers.dense(net,
                              units=7 * 7 * 128,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.layers.batch_normalization(net,
                                            training=is_training,
                                            name='batch_normalization')
        net = tf.nn.relu(net, name='relu')

    net = tf.reshape(net, [-1, 7, 7, 128])

    with tf.variable_scope('layer_3'):
        net = tf.layers.conv2d_transpose(net,
                                         filters=64,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=INIT_KERNEL,
                                         name='conv')
        net = tf.layers.batch_normalization(net,
                                            training=is_training,
                                            name='batch_normalization')
        net = tf.nn.relu(net, name='relu')

    with tf.variable_scope('layer_4'):
        net = tf.layers.conv2d_transpose(net,
                                         filters=1,
                                         kernel_size=4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=INIT_KERNEL,
                                         name='conv')
        net = tf.nn.sigmoid(net, name='sigmoid')
        net = tf.reshape(net, [-1, OUTPUT_DIM])

    return net


def mnist_discriminator_network(inputs, is_training):
    with tf.variable_scope('layer_1'):
        net = tf.reshape(inputs, [-1, 28, 28, 1])
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=INIT_KERNEL,
                               name='conv')
        net = leaky_relu(net, 0.1, name='leaky_relu')

    with tf.variable_scope('layer_2'):
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=INIT_KERNEL,
                               name='conv')
        net = tf.layers.batch_normalization(net,
                                            training=is_training,
                                            name='batch_normalization')
        net = leaky_relu(net, 0.1, name='leaky_relu')

    net = tf.reshape(net, [-1, 7 * 7 * 64])

    with tf.variable_scope('layer_3'):
        net = tf.layers.dense(net,
                              units=1024,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        net = tf.layers.batch_normalization(net,
                                            training=is_training,
                                            name='batch_normalization')
        net = leaky_relu(net, 0.1, name='leaky_relu')

    with tf.variable_scope('layer_4'):
        net = tf.layers.dense(net,
                              units=1,
                              kernel_initializer=INIT_KERNEL,
                              name='fc')
        intermediate_layer = net
        net = tf.nn.sigmoid(net)

    net = tf.squeeze(net)

    return net, intermediate_layer
