import os
import logging
import shutil
from unittest import TestCase

import numpy as np
from sklearn.metrics import classification_report

from alphai_watson.performance import GANPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from alphai_rickandmorty_oracle.datasource.mnist import MNISTDataSource
from alphai_rickandmorty_oracle.detective import RickAndMortyDetective
from alphai_rickandmorty_oracle.model import RickAndMorty
from alphai_rickandmorty_oracle.networks.mnist import mnist_generator_network, mnist_discriminator_network

from tests.helpers import assert_train_files_exist

TEST_PATH = os.path.join(os.path.dirname(__file__), '..')
RESOURCES_PATH = os.path.join(TEST_PATH, 'resources')


class TestDetectiveMNIST(TestCase):
    def setUp(self):
        self.output_path = os.path.join(TEST_PATH, 'output')
        try:
            os.makedirs(self.output_path)
        except OSError:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(self.output_path)
        except Exception:
            pass

    def test_rm_train_detect_in_detective_wrap_mnist(self):
        abnormal_digit = 0
        roc_score_threshold = 0.6

        # Train and test data file
        train_data_file = os.path.join(RESOURCES_PATH, 'mnist_data_train_abnormalclass-{}.hd5'.format(abnormal_digit))
        test_data_file = os.path.join(RESOURCES_PATH, 'mnist_data_test_abnormalclass-{}.hd5'.format(abnormal_digit))

        n_sensors = 28
        n_timesteps = 784 // n_sensors

        train_data_source = MNISTDataSource(source_file=train_data_file,
                                            transformer=NullTransformer(number_of_timesteps=n_timesteps,
                                                                        number_of_sensors=n_sensors))
        test_data_source = MNISTDataSource(source_file=test_data_file,
                                           transformer=NullTransformer(number_of_timesteps=n_timesteps,
                                                                       number_of_sensors=n_sensors))

        train_data = train_data_source.get_train_data('NORMAL')

        # Get test data
        test_data_normal = test_data_source.get_train_data('NORMAL')
        test_data_abnormal = test_data_source.get_train_data('ABNORMAL')
        test_data = test_data_source.get_train_data('ALL')

        # Ground truth for ABNORMAL data is the abnormal_digit, ground truth for NORMAL data is 1
        n1 = np.zeros(len(test_data_abnormal.data))
        n2 = np.ones(len(test_data_normal.data))
        expected_truth = np.hstack((n1, n2))

        # Initialise model
        output_dimensions = 784
        plot_dimensions = (28, 28)
        batch_size = 64
        train_iters = 100

        model = RickAndMorty(generator_network=mnist_generator_network,
                             discriminator_network=mnist_discriminator_network,
                             output_dimensions=output_dimensions,
                             plot_dimensions=plot_dimensions,
                             batch_size=batch_size,
                             train_iters=train_iters,
                             plot_save_path=self.output_path)

        detective = RickAndMortyDetective(model_configuration={
            'model': model,
            'batch_size': batch_size,
            'output_dimensions': output_dimensions,
            'train_iters': train_iters,
            'plot_save_path': self.output_path
        })

        # Train
        detective.train(train_data)

        assert_train_files_exist(self.output_path)

        # Test
        detection_result = detective.detect(test_data)

        # Calculate ROC score
        roc_score = GANPerformanceAnalysis({}).analyse(
            detection_result=detection_result.data,
            expected_truth=expected_truth
        )
        logging.debug('ROC Score: {}'.format(roc_score))

        # Generate Classification Report
        train_results = detective.detect(train_data).data
        threshold = np.mean(train_results)
        prediction = [1 if x >= threshold else 0 for x in detection_result.data]

        target_names = ['ABNORMAL', 'NORMAL']
        logging.debug(classification_report(expected_truth, prediction, target_names=target_names))

        assert roc_score > roc_score_threshold, \
            'Post-train ROC score should be higher roc_score_threshold ({}). The GAN is not learning!'\
            .format(roc_score_threshold)
