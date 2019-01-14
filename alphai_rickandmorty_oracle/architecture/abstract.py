from abc import ABCMeta, abstractmethod


class AbstractGanArchitecture(metaclass=ABCMeta):
    def __init__(self, output_dimensions, plot_dimensions):
        """

        :param output_dimensions:
        :param plot_dimensions:
        """
        self.output_dimensions = output_dimensions
        self.plot_dimensions = plot_dimensions

    @abstractmethod
    def generator_network(self, noise, is_training):
        """
        GAN generator architecture
        :param noise:
        :param is_training:
        :return:
        """

        raise NotImplementedError

    @abstractmethod
    def discriminator_network(self, inputs, is_training):
        """
        GAN discriminator architecture
        :param inputs:
        :param is_training:
        :return:
        """

        raise NotImplementedError
