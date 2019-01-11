import os
import logging

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from alphai_watson.performance import GANPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from alphai_rickandmorty_oracle.datasource.kddcup99 import KDDCup99DataSource
from alphai_rickandmorty_oracle.detective import RickAndMortyDetective
from alphai_rickandmorty_oracle.model_kddcup99 import RickAndMorty

TEST_PATH = os.path.join(os.path.dirname(__file__), '..')
RESOURCES_PATH = os.path.join(TEST_PATH, 'resources')


def test_rm_train_detect_in_detective_wrap_kddcup99():
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
    output_path = os.path.join(TEST_PATH, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch_size = 64
    output_dimensions = 121
    train_iters = 10000

    model = RickAndMorty(batch_size=batch_size,
                         output_dimensions=output_dimensions,
                         train_iters=train_iters,
                         plot_save_path=output_path)

    detective = RickAndMortyDetective(model_configuration={
        'model': model,
        'batch_size': batch_size,
        'output_dimensions': output_dimensions,
        'train_iters': train_iters,
        'plot_save_path': output_path
    })

    # Train
    detective.train(train_data)

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
