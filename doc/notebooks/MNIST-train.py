import os

from alphai_watson.transformer import NullTransformer
from alphai_rickandmorty_oracle.datasource.mnist import MNISTDataSource
from alphai_rickandmorty_oracle.detective import RickAndMortyDetective

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), '../../tests', 'resources')

abnormal_digit = 0

# Train and test data file
train_data_file = os.path.join(RESOURCES_PATH, 'mnist_data_train_abnormalclass-{}.hd5'.format(abnormal_digit))
test_data_file = os.path.join(RESOURCES_PATH, 'mnist_data_test_abnormalclass-{}.hd5'.format(abnormal_digit))

# Model parameters
n_sensors = 28
n_timesteps = 784 // n_sensors

train_data_source = MNISTDataSource(source_file=train_data_file,
                                         transformer=NullTransformer(number_of_timesteps=n_timesteps,
                                                                     number_of_sensors=n_sensors))
test_data_source = MNISTDataSource(source_file=test_data_file,
                                         transformer=NullTransformer(number_of_timesteps=n_timesteps,
                                                                     number_of_sensors=n_sensors))

train_data = train_data_source.get_train_data('NORMAL')

detective = RickAndMortyDetective(model_configuration={
    'batch_size': 64,
    'output_dimensions': 784,
    'train_iters': 300,
    'save_path' : './mnist_models/MNIST-abnormalclass-{}'.format(abnormal_digit),
    # 'load_path' : 'trained_models/MNIST-abnormalclass-{}'.format(abnormal_digit),
    'plot_save_path' : './'
})

detective.train(train_data)
