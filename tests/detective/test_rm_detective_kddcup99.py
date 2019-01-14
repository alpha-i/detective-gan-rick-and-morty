import os
import logging
import shutil
from unittest import TestCase

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from alphai_watson.performance import GANPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from alphai_rickandmorty_oracle.datasource.kddcup99 import KDDCup99DataSource
from alphai_rickandmorty_oracle.detective import RickAndMortyDetective
from alphai_rickandmorty_oracle.model import RickAndMorty
from alphai_rickandmorty_oracle.networks.kddcup99 import kddcup99_generator_network, kddcup99_discriminator_network

from tests.helpers import assert_train_files_exist


TEST_PATH = os.path.join(os.path.dirname(__file__), '..')
RESOURCES_PATH = os.path.join(TEST_PATH, 'resources')


class TestDetectiveKDDCup99(TestCase):
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

    def test_rm_train_detect_in_detective_wrap_kddcup99(self):
        roc_score_threshold = 0.6

        # Train and test data file
        data_filename = os.path.join(RESOURCES_PATH, 'kddcup.data_10_percent_corrected')
        header_filename = os.path.join(RESOURCES_PATH, 'kddcup.names')

        kdd_datasource = KDDCup99DataSource(source_file=data_filename,
                                            header_file=header_filename,
                                            transformer=NullTransformer(8, 8))

        train_data = kdd_datasource.get_train_data('NORMAL')
        data_normal_test = kdd_datasource.get_train_data('NORMAL_TEST')
        data_abnormal_test = kdd_datasource.get_train_data('ABNORMAL_TEST')

        # Collate ground truth for test data
        n1 = np.ones(len(data_normal_test.data))
        n2 = np.zeros(len(data_abnormal_test.data))

        expected_truth = np.hstack((n1, n2))

        # Initialise model
        output_dimensions = 121
        plot_dimensions = (11, 11)
        batch_size = 64
        train_iters = 1000

        model = RickAndMorty(generator_network=kddcup99_generator_network,
                             discriminator_network=kddcup99_discriminator_network,
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
        test_results_normal = detective.detect(data_normal_test).data
        test_results_abnormal = detective.detect(data_abnormal_test).data
        detection_result = np.hstack((test_results_normal, test_results_abnormal))

        # Calculate ROC score
        roc_score = GANPerformanceAnalysis({}).analyse(
            detection_result=detection_result,
            expected_truth=expected_truth
        )
        logging.debug('ROC Score: {}'.format(roc_score))

        # Calculate model accuracy on train data to set a threshold to distinguish NORMAL and ABNORMAL data
        def model_accuracy(data, status, threshold=None):
            results = detective.detect(data).data
            if threshold is None:
                threshold = np.median(results)
            ground_truth = [status] * len(results)
            prediction = [1 if x >= threshold else 0 for x in results]
            logging.debug('Accuracy: {0:.2f}%'.format(100 * accuracy_score(ground_truth, prediction)))
            return threshold

        # Generate Classification Report
        accuracy_threshold = model_accuracy(train_data, 1)
        class_predictions = [1 if x >= accuracy_threshold else 0 for x in detection_result]

        target_names = ['ABNORMAL', 'NORMAL']
        logging.debug(classification_report(expected_truth, class_predictions, target_names=target_names))

        assert roc_score > roc_score_threshold, \
            'Post-train ROC score should be higher roc_score_threshold ({}). The GAN is not learning!' \
            .format(roc_score_threshold)
