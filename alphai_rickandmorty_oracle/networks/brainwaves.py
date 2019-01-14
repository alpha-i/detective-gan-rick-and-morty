import tensorflow as tf

from alphai_rickandmorty_oracle.networks.abstract import AbstractGanArchitecture

import alphai_rickandmorty_oracle.tflib as lib
import alphai_rickandmorty_oracle.tflib.ops.linear
import alphai_rickandmorty_oracle.tflib.ops.conv2d
import alphai_rickandmorty_oracle.tflib.ops.deconv2d

INIT_KERNEL = tf.random_normal_initializer(mean=0.0, stddev=0.02)
OUTPUT_DIM = 784  # Number of pixels in MNIST (28*28)
DIM = 64
Z_DIM = 128
DISC_FILTER_SIZE = 5


def leaky_relu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _impl_leaky_relu(x, alpha)
    else:
        return _impl_leaky_relu(x, alpha)


def _impl_leaky_relu(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


class BrainwavesGanArchitecture(AbstractGanArchitecture):
    def __init__(self, output_dimensions, plot_dimensions):
        super().__init__(output_dimensions, plot_dimensions)

    def generator_network(self, noise, is_training):
        net = lib.ops.linear.Linear('generator.Input', Z_DIM, 4 * 4 * 4 * DIM, noise)
        net = tf.nn.relu(net)
        net = tf.reshape(net, [-1, 4 * DIM, 4, 4])

        net = lib.ops.deconv2d.Deconv2D('generator.2', 4 * DIM, 2 * DIM, 5, net)
        net = tf.nn.relu(net)

        net = net[:, :, :7, :7]

        net = lib.ops.deconv2d.Deconv2D('generator.3', 2 * DIM, DIM, 5, net)
        net = tf.nn.relu(net)

        net = lib.ops.deconv2d.Deconv2D('generator.5', DIM, 1, 5, net)
        net = tf.nn.sigmoid(net)

        net = tf.reshape(net, [-1, OUTPUT_DIM])

        return net

    def discriminator_network(self, inputs, is_training):
        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.))

        net = tf.reshape(inputs, [-1, 1, 28, 28])  # 28x28 or 8x98
        net = lib.ops.conv2d.Conv2D('discriminator.1', 1, DIM, DISC_FILTER_SIZE, net,
                                    stride=2)  # name, input, output, filter
        net = leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob=keep_prob)  # adding dropout after activators
        net = lib.ops.conv2d.Conv2D('discriminator.2', DIM, 2 * DIM, DISC_FILTER_SIZE, net, stride=2)
        net = leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob=keep_prob)  # adding dropout after activators
        net = lib.ops.conv2d.Conv2D('discriminator.3', 2 * DIM, 4 * DIM, DISC_FILTER_SIZE, net, stride=2)
        net = leaky_relu(net)
        net = tf.nn.dropout(net, keep_prob=keep_prob)  # adding dropout after activators
        net = tf.reshape(net, [-1, 4 * 4 * 4 * DIM])  # D_

        intermediate_layer = net

        net = lib.ops.linear.Linear('discriminator.Output', 4 * 4 * 4 * DIM, 1, net)  # D
        net = tf.reshape(net, [-1])

        return net, intermediate_layer
