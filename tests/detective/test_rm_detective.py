import itertools
import os

import numpy as np
from alphai_watson.datasource.brainwaves import BrainwavesDataSource
from alphai_watson.performance import GANPerformanceAnalysis
from alphai_watson.transformer import NullTransformer

from alphai_rickandmorty_oracle.detective import RickAndMortyDetective

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), '..', 'resources')


def test_rm_train_in_detective_wrap():
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_sample_1.hd5')
    n_sensors = 16
    n_timesteps = 784 // n_sensors

    train_data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps=n_timesteps, number_of_sensors=n_sensors))

    train_data = train_data_source.get_train_data('NORMAL')
    detective = RickAndMortyDetective(model_configuration={
        'batch_size': 64,
        'output_dimensions': 784,
        'train_iters': 1,
    })
    detective.train(train_data)


def test_rm_detect_in_detective_wrap():
    train_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_sample_1.hd5')
    test_data_file = os.path.join(RESOURCES_PATH, 'brainwaves_normal_and_abnormal.hd5')
    n_sensors = 16
    n_timesteps = 784 // n_sensors

    train_data_source = BrainwavesDataSource(
        source_file=train_data_file,
        transformer=NullTransformer(number_of_timesteps=n_timesteps, number_of_sensors=n_sensors)
    )

    test_data_source = BrainwavesDataSource(
        source_file=test_data_file,
        transformer=NullTransformer(number_of_timesteps=n_timesteps, number_of_sensors=n_sensors)
    )

    train_sample = train_data_source.get_train_data('NORMAL')

    detective = RickAndMortyDetective(model_configuration={})

    detective.train(train_sample)

    normal_samples = [sample for sample in test_data_source.get_test_data('NORMAL')]
    abnormal_samples = [sample for sample in test_data_source.get_test_data('ABNORMAL')]
    normal_and_abnormal_samples = list(itertools.chain(normal_samples, abnormal_samples))

    for test_sample in normal_and_abnormal_samples:
        detection_result = detective.detect(test_sample)

        expected_truth = np.zeros(detection_result.data.size)
        expected_truth[0] = 1

        roc_score = GANPerformanceAnalysis({}).analyse(
            detection_result=detection_result.data,
            expected_truth=expected_truth
        )

        assert -1 <= roc_score <= 1
