import os
import logging

import numpy as np
from sklearn.metrics import classification_report

from alphai_watson.performance import GANPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from alphai_rickandmorty_oracle.datasource.brainwaves import BrainwavesDataSource
from alphai_rickandmorty_oracle.detective import RickAndMortyDetective
from alphai_rickandmorty_oracle.model_brainwaves import RickAndMorty

TEST_PATH = os.path.join(os.path.dirname(__file__), '..')
RESOURCES_PATH = os.path.join(TEST_PATH, 'resources')


def test_rm_train_detect_in_detective_wrap_brainwaves():
    n_sensors = 16
    n_timesteps = 784 // n_sensors
    roc_score_threshold = 0.6

    train_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_sample_1.hd5')
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')

    train_data_source = BrainwavesDataSource(
        source_file=train_data_file,
        transformer=NullTransformer(number_of_timesteps=n_timesteps, number_of_sensors=n_sensors)
    )

    test_data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps=n_timesteps, number_of_sensors=n_sensors)
    )

    train_data = train_data_source.get_train_data('NORMAL')

    test_normal_samples = test_data_source.get_train_data('NORMAL')
    test_abnormal_samples = test_data_source.get_train_data('ABNORMAL')

    # Collate ground truth for test data
    n1 = np.ones(len(test_normal_samples.data))
    n2 = np.zeros(len(test_abnormal_samples.data))

    expected_truth = np.hstack((n1, n2))

    # Initialise model
    output_path = os.path.join(TEST_PATH, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch_size = 64
    output_dimensions = 784
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
    test_results_normal = detective.detect(test_normal_samples).data
    test_results_abnormal = detective.detect(test_abnormal_samples).data
    detection_result = np.hstack((test_results_normal, test_results_abnormal))

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
