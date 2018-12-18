import logging

import contexttimer
import numpy as np
from alphai_watson.datasource import Sample
from alphai_watson.detective import AbstractDetective, DetectionResult

from alphai_rickandmorty_oracle.model_kddcup99 import RickAndMorty

DEFAULT_TRAIN_ITERS = 1  # 50k takes 3 hours
ITERATIONS_PER_TEST = 10000  # 1000
N_TIMESTEPS = 392  # Larger input to network, but perhaps helps training if more data per batch
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_DIMS = 784  # Number of sensors * features_per_sensor

logging.basicConfig(level=logging.DEBUG)


class RickAndMortyDetective(AbstractDetective):
    """
    Detective for Ricky and Morty GAN neural network.

    """
    def __init__(self, model_configuration: dict):
        """
        The model configuration is a dictionary containing the configuration parameters for the underlying ML model.

        The most important variables are:
            - batch_size: which determines the size of the batch during training
            - output_dimensions: set the dimension of the network. it must be set to be comform to the data shape
            - train_iters: how many iteration the model should do during training

        Optional values are:
            - plot_save_path: if it's valued it forces the model to dump an png image of the batch
            - load_path: defines where are the pre-trained model files located
            - save_path: defines where the training files should be saved.

        :param dict model_configuration:
        """
        batch_size = model_configuration.get('batch_size', DEFAULT_BATCH_SIZE)
        output_dimensions = model_configuration.get('output_dimensions', DEFAULT_MODEL_DIMS)
        train_iters = model_configuration.get('train_iters', DEFAULT_TRAIN_ITERS)
        plot_save_path = model_configuration.get('plot_save_path')
        load_path = model_configuration.get('load_path')
        save_path = model_configuration.get('save_path')

        self._config = dict(
            batch_size=batch_size,
            output_dimensions=output_dimensions,
            train_iters=train_iters,
            plot_save_path=plot_save_path
        )
        self.model = RickAndMorty(batch_size=batch_size, output_dimensions=output_dimensions, train_iters=train_iters,
                                  plot_save_path=plot_save_path, load_path=load_path)
        self.save_path = save_path
        self.load_path = load_path

    def train(self, train_sample: Sample) -> None:
        """
        Performs the training of the model

        :param alphai_watson.datasource.Sample train_sample:
        :return:
        """
        logging.debug("Starting session")
        self.model.run_training_routine(train_sample)
        logging.debug("Training complete.")

        if self.save_path:
            self.model.save(self.save_path)

    def detect(self, test_sample: Sample) -> DetectionResult:
        """
        Performs the detection of the model

        :param alphai_watson.datasource.Sample test_sample:
        :return:
        """

        logging.info("Running detector on {}".format(test_sample))

        test_data = test_sample.data.astype(np.float32)

        with contexttimer.Timer() as t:
            detection_array = self.model.run_discriminator(test_data)
        logging.info("Detection completed in {}".format(t.elapsed))

        return DetectionResult(
            data=detection_array,
            n_timesteps_in_chunk=test_sample.number_of_timesteps,
            original_sample_rate=test_sample.sample_rate
        )

    def diagnose(self, test_chunk):
        """

        Finds the closest synthetic chunk to the input data. Useful for highlighting the anomaly.

        :param ndarray test_chunk:
        :return: ndarray synthetic_chunk: The synthetic_chunk
        """

        test_chunk = test_chunk.astype(np.float32)
        synthetic_chunk = self.model.find_closest_synthetic_chunk(test_chunk)
        synthetic_chunk = synthetic_chunk.reshape(test_chunk.shape)

        return synthetic_chunk

    @property
    def configuration(self):
        """
        Return a dict with the used configuration, to be saved during testing phase
        :return:
        """
        return self._config
