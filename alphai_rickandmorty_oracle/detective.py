import logging

import contexttimer
import numpy as np
from alphai_watson.datasource import Sample
from alphai_watson.detective import AbstractDetective, DetectionResult

logging.basicConfig(level=logging.DEBUG)


class RickAndMortyDetective(AbstractDetective):
    """
    Detective for Ricky and Morty GAN neural network.

    """
    def __init__(self, model_configuration: dict):
        """
        The model configuration is a dictionary containing the configuration parameters for the underlying ML model.

        The most important variables are:
            - model: RickAndMorty model object defining GAN architecture and methods
            - output_dimensions: set the dimension of the network. it must be set to be conform to the data shape
            - batch_size: which determines the size of the batch during training
            - train_iters: how many iteration the model should do during training

        Optional values are:
            - plot_save_path: if it's valued it forces the model to dump an png image of the batch
            - load_path: defines where are the pre-trained model files located
            - save_path: defines where the training files should be saved.

        :param dict model_configuration:
        """
        model = model_configuration.get('model')
        output_dimensions = model_configuration.get('output_dimensions')
        batch_size = model_configuration.get('batch_size')
        train_iters = model_configuration.get('train_iters')
        plot_save_path = model_configuration.get('plot_save_path')
        load_path = model_configuration.get('load_path')
        save_path = model_configuration.get('save_path')

        self.model = model

        self._config = dict(
            output_dimensions=output_dimensions,
            batch_size=batch_size,
            train_iters=train_iters,
            plot_save_path=plot_save_path
        )

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

        :param alphai_watson.datasource.Sample test_sample: input sample to evaluate for anomaly
        :return alphai_watson.detective.DetectionResult: object containing detection verdict
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

    def diagnose(self, test_sample):
        """

        Finds the closest synthetic sample to the input data. Useful for highlighting the anomaly.

        :param ndarray test_sample: input sample on which to run root cause analysis
        :return: ndarray synthetic_sample: The closest synthetic sample to the test_sample that the
                                          generative model could produce
        """

        test_sample = test_sample.astype(np.float32)
        synthetic_sample = self.model.find_closest_synthetic_sample(test_sample)
        synthetic_sample = synthetic_sample.reshape(test_sample.shape)

        return synthetic_sample

    @property
    def configuration(self):
        """
        :return: dict with the used configuration, to be saved during testing phase
        """
        return self._config
