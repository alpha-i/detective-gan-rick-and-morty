import os
import time
import logging

import numpy as np
import tensorflow as tf

import alphai_rickandmorty_oracle.tflib as lib
import alphai_rickandmorty_oracle.tflib.ops.linear
import alphai_rickandmorty_oracle.tflib.ops.conv2d
import alphai_rickandmorty_oracle.tflib.ops.deconv2d
import alphai_rickandmorty_oracle.tflib.save_images
import alphai_rickandmorty_oracle.tflib.plot

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Forces GPU

DEFAULT_TRAIN_ITERS = 5000  # How many generator iterations to train for. Default 50k
DEFAULT_FIT_EPOCHS = 1000  # How many iterations to diagnose the anomaly
DEFAULT_Z_DIM = 200
Factor_M = 0.0  # factor M
LAMBDA_2 = 2.0  # Weight factor. Previously 0.4

DIM = 64  # Model dimensionality
BATCH_SIZE = 50  # Batch size
CRITIC_ITERS = 5  # Number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 784  # Number of pixels in MNIST (28*28)
DEFAULT_LEARN_RATE = 0.0001
DIAGNOSIS_LEARN_RATE = 0.01
DISC_FILTER_SIZE = 5

INIT_KERNEL = tf.random_normal_initializer(mean=0.0, stddev=0.02)

reuse = tf.AUTO_REUSE
getter = None

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    """ Discriminators tend to train better when using this activation function. """
    return tf.maximum(alpha * x, x)

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

class RickAndMorty(object):
    """
    Implementation of GAN neural network
    """
    def __init__(self, batch_size=BATCH_SIZE, output_dimensions=OUTPUT_DIM, learning_rate=DEFAULT_LEARN_RATE,
                 train_iters=DEFAULT_TRAIN_ITERS, z_dim=DEFAULT_Z_DIM, plot_save_path=None, load_path=None):
        """ Generative model which primarily consists of a generator and discriminator.

        :param int batch_size:
        :param output_dimensions:
        :param learning_rate: Learning rate
        :param train_iters: Number of training iterations
        :param z_dim: Size of random number entering generator
        """

        self.fixed_noise = tf.constant(np.random.normal(size=(128, z_dim)).astype('float32'))
        self.z_dim = z_dim
        self.saver = None
        self.is_initialised = False
        self.batch_size = batch_size
        self.output_dimensions = output_dimensions
        self.learning_rate = learning_rate
        self.train_iters = train_iters
        self.load_path = load_path

        self._plot_save_path = plot_save_path
        self.tf_session = tf.InteractiveSession()

        z_init = tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32)
        self.ano_z = tf.get_variable('ano_z', shape=[1, self.z_dim], dtype=tf.float32, initializer=z_init)
        self.chunk = tf.placeholder(tf.float32, shape=[1, self.output_dimensions])
        self.ano_z_optimiser, self.anomaly_score, self.fake_sample = self._build_diagnosis_tools()

    def __del__(self):
        self.tf_session.close()

    def save(self, file_path):
        """ Save a trained model. """

        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.save(self.tf_session, file_path)

    def _load_model(self):
        """ Load a trained model. Variables must already exist. """
        logging.info("Attempting to load model")
        self.saver = tf.train.Saver()
        self.saver.restore(self.tf_session, self.load_path)
        logging.info("Model restored.")
        self.is_initialised = True

    def generator(self, n_chunks, noise=None, is_training=True):
        """ Creates fake samples to mimic the normal data

        :param int n_chunks:
        :param noise:
        :return:
        """
        
        if noise is None:
            noise = tf.random_normal([n_chunks, self.z_dim])

        with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

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
                                      units=7*7*128,
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
                                         strides= 2,
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

    def discriminator(self, inputs, keep_prob):
        """ Decides whether the input is anomalous or not

        :param tensor inputs:
        :param tensor keep_prob: Probability of keeping node. Set to 0.5 for training; 1 for testing
        :return: tensor, tensor: output, feature_layer
        """
        if keep_prob == 1:
            is_training = False
        else:
            is_training = True
            
        with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

            with tf.variable_scope('layer_1'):
                net = tf.reshape(inputs, [-1, 28, 28, 1])
                net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=INIT_KERNEL,
                               name='conv')
                net = leakyReLu(net, 0.1, name='leaky_relu')

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
                net = leakyReLu(net, 0.1, name='leaky_relu')

            net = tf.reshape(net, [-1, 7 * 7 * 64])

            with tf.variable_scope('layer_3'):
                net = tf.layers.dense(net,
                          units=1024,
                          kernel_initializer=INIT_KERNEL,
                          name='fc')
                net = tf.layers.batch_normalization(net,
                                        training=is_training,
                                        name='batch_normalization')
                net = leakyReLu(net, 0.1, name='leaky_relu')

            # intermediate_layer = net

            with tf.variable_scope('layer_4'):
                net = tf.layers.dense(net,
                                      units=1,
                                      kernel_initializer=INIT_KERNEL,
                                      name='fc')
                intermediate_layer = net
                net = tf.nn.sigmoid(net)

            net = tf.squeeze(net)

            return net, intermediate_layer

    def generate_fake_chunks(self):
        """ Save random samples from the generator to help assess its performance. """

        if self._plot_save_path:
            fixed_fake_chunks = self.generator(128, noise=self.fixed_noise, is_training=False)
            samples = self.tf_session.run(fixed_fake_chunks)
            lib.save_images.save_images(
                samples.reshape((128, 28, 28)),
                os.path.join(self._plot_save_path, 'fake_chunks.png')
            )
            logging.info("Saving fake samples to png.")

    def get_cost_ops(self, real_data, keep_prob):
        """ Defines the cost functions which are used to train the discriminator and generator.

        :param real_data: To be fed into discriminator
        :param keep_prob: Tensor which should hold the value of 1.0 during testing
        :return: Cost associated with the generator and discriminator.
        """
        
        fake_data = self.generator(self.batch_size)
        real_d, inter_layer_real = self.discriminator(real_data, keep_prob)
        fake_d, inter_layer_fake = self.discriminator(fake_data, keep_prob)

        
        # Calculate seperate losses for discriminator with real and fake images
        real_discriminator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=inter_layer_real,
                                                    labels=tf.ones_like(inter_layer_real)))
        
        fake_discriminator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=inter_layer_fake,
                                                    labels=tf.zeros_like(inter_layer_fake)))

        # Add discriminator losses
        disc_cost = real_discriminator_loss + fake_discriminator_loss
        
        # Calculate loss for generator by flipping label on discriminator output        
        gen_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=inter_layer_fake,
                                                    labels=tf.ones_like(inter_layer_fake)))

        return gen_cost, disc_cost

    def get_training_ops(self, real_data, keep_prob):

        gen_cost, disc_cost = self.get_cost_ops(real_data, keep_prob)

        disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
        
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)

        return gen_train_op, disc_train_op

    def _build_diagnosis_tools(self):
        """ Runs at init in order to define operator and variables to be used later for running diagnostics.
        :return: Operator, tensor, tensor
        """

        fake_sample = self.generator(noise=self.ano_z, n_chunks=1, is_training=False)

        anomaly_score = tf.reduce_sum(tf.abs(tf.subtract(tf.reshape(self.chunk, [-1]), tf.reshape(fake_sample, [-1]))))
        ano_z_optimiser = tf.train.AdamOptimizer(learning_rate=DIAGNOSIS_LEARN_RATE, name='ano_z_optimizer').minimize(
            anomaly_score, var_list=self.ano_z)
        # self.learning_rate

        return ano_z_optimiser, anomaly_score, fake_sample

    def _initialise_model(self):
        """ Assigns initial values to parameters at the start of training. """
        logging.info("Initialising Model")
        self.tf_session.run(tf.global_variables_initializer())
        self.is_initialised = True

    def find_closest_synthetic_chunk(self, chunk, n_fit_epochs=DEFAULT_FIT_EPOCHS):
        """ Finds the closest generated chunk to the input data. Useful for highlighting the anomaly.

        :param chunk: The piece of data we wish to mimic
        :param n_fit_epochs: How many iterations we use to seek the best fit chunk
        :return: ndarray representing the generator's best impersonation of the chunk
        """

        logging.info("Searching for closest synthetic sample")

        if self.load_path:
            try:
                self._load_model()
            except Exception as e:
                logging.warning("Restore file not recovered: {}".format(e))

        if not self.is_initialised:
            self._initialise_model()

        logging.info("Training synthetic chunk.")
        for epoch in range(n_fit_epochs):
            ano_z, _, ano_score, best_fit_chunk = self.tf_session.run(
                [self.ano_z, self.ano_z_optimiser, self.anomaly_score, self.fake_sample],
                feed_dict={self.chunk: chunk.reshape([1, -1])})

        best_fit_chunk = self.tf_session.run(self.fake_sample)

        anomaly = chunk.flatten() - best_fit_chunk.flatten()
        anomaly_rms = np.std(anomaly)
        logging.info("Synthetic training procedure complete. Anomaly rms of {}".format(ano_z, anomaly_rms))

        return best_fit_chunk

    def _calculate_anomaly_score(self, input, ano_gen, ano_z, discriminator_fraction=0.1):
        """ TODO: Returns a detection based on a convex combination of discriminator and generator.

        :param input:
        :param ano_gen:
        :param ano_z:
        :param discriminator_fraction:
        :return: Total anomaly score
        """

        fake_sample = ano_gen(ano_z)
        res_loss = tf.reduce_mean(tf.abs(tf.subtract(input, fake_sample)))
        # anomaly_score = discriminator_fraction * disc_loss + (1 - discriminator_fraction) * res_loss

        return res_loss

    def run_discriminator(self, input):
        """ Executes a forward pass through the discriminator.

        :param input: Shape (any, self.x_dim)
        :return: nparray Probabilities p(sample is real) (if use_softmax=True; logits if False)
        """

        x = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dimensions])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        d_output, _ = self.discriminator(x, keep_prob)

        if self.load_path:  # Load model values if needed
            try:
                self._load_model()
            except Exception as e:
                logging.warning("Restore file not recovered: {}".format(e))

        if not self.is_initialised:
            self._initialise_model()

        # Cycle through batches
        n_samples = input.shape[0]
        n_batches = int(n_samples / self.batch_size)

        detection_list = []
        for i in range(n_batches):
            lo = i * self.batch_size
            hi = lo + self.batch_size
            input_batch = input[lo:hi]
            input_batch = input_batch.reshape((self.batch_size, -1))
            # d_output = tf.sigmoid(d_output)

            # keep prob is 1 for testing; 0.5 for training
            batch_scores = 1 * self.tf_session.run(d_output, feed_dict={x: input_batch, keep_prob: 1.})

            detection_list.append(batch_scores)

        # Now compute incomplete batch
        n_residuals = n_samples % self.batch_size
        if n_residuals > 0:
            if len(input.shape) == 3:  # is a multivariate case
                input_batch = np.zeros((self.batch_size, input.shape[1], input.shape[2]))
            elif len(input.shape) == 2:  # is a monovariate case
                input_batch = np.zeros((self.batch_size, self.output_dimensions))
            input_batch[0:n_residuals] = input[-n_residuals:]
            input_batch = input_batch.reshape((self.batch_size, -1))

            batch_scores = 1 * self.tf_session.run(d_output, feed_dict={x: input_batch, keep_prob: 1.})
            detection_list.append(batch_scores[0:n_residuals])

        detector_results = np.concatenate(detection_list).flatten()

        return detector_results

    def run_training_routine(self, train_sample):
        """ Train the generator and discriminator. Loading and saving of the model is performed if requested.

        :param train_sample: Data for training
        """

        real_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dimensions])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        gen_cost, disc_cost = self.get_cost_ops(real_data, keep_prob)
        gen_train_op, disc_train_op = self.get_training_ops(real_data, keep_prob)

        logging.debug("Start training loop...")
        random_batch_generator = train_sample.get_infinite_random_batch_generator(self.batch_size, strict=True)

        if self.load_path:
            self._load_model()

        if not self.is_initialised:
            self._initialise_model()

        for iteration in range(self.train_iters):
            start_time = time.time()

            if iteration % 1000 == 0:
                logging.info("Training iteration {} of {}".format(iteration, self.train_iters))

            if iteration > 0:
                _gen_cost, _ = self.tf_session.run(gen_train_op, feed_dict={keep_prob: 0.5})
                lib.plot.add_to_plot('train gen cost', _gen_cost)

            disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data = next(random_batch_generator)
                _data = _data.reshape((self.batch_size, -1))
                _disc_cost, _ = self.tf_session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data, keep_prob: 0.5}
                )

            lib.plot.add_to_plot('train disc cost', _disc_cost)
            lib.plot.add_to_plot('time', time.time() - start_time)

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                self.generate_fake_chunks()

            # Write logs every 100 iters
            if ((iteration < 5) or (iteration % 100 == 99)) and self._plot_save_path:
                lib.plot.flush(self._plot_save_path)

            lib.plot.tick()