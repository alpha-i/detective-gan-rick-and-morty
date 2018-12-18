import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from alphai_watson.performance import GANPerformanceAnalysis
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
    # 'save_path' : 'trained_models/MNIST-abnormalclass-{}'.format(abnormal_digit),
    'load_path' : './mnist_models/MNIST-abnormalclass-{}'.format(abnormal_digit),
    'plot_save_path' : './'
})

# Get test data
test_data_normal = test_data_source.get_train_data('NORMAL')
test_data_abnormal = test_data_source.get_train_data('ABNORMAL')
test_data = test_data_source.get_train_data('ALL')

# Ground truth for ABNORMAL data is 1 , ground truth for NORMAL data is 0
n1 = np.ones(len(test_data_abnormal.data))
n2 = np.zeros(len(test_data_normal.data))
expected_truth = np.hstack((n1, n2))

detection_result = detective.detect(test_data)

roc_score = GANPerformanceAnalysis({}).analyse(
  detection_result=detection_result.data,
  expected_truth=expected_truth
)

print(roc_score)

# Save ; Compared ground truth to np.rint(detection_result.data), which rounds probability <0.5 to 0 and >0.5 to 1
clf_rep = precision_recall_fscore_support(expected_truth, np.rint(detection_result.data))
out_dict = {
             "precision" :clf_rep[0].round(2)
            ,"recall" : clf_rep[1].round(2)
            ,"f1-score" : clf_rep[2].round(2)
            ,"support" : clf_rep[3]
            }
df_out = pd.DataFrame(out_dict, index = ['NORMAL', 'ABNORMAL'])
avg_tot = (df_out.apply(lambda x: round(x.mean(), 2) if x.name != "support" else  round(x.sum(), 2)).to_frame().T)
avg_tot.index = ["avg/total"]
df_out = df_out.append(avg_tot)

# Save Classification report to CSV
df_out.to_csv('classification_report_digit-{}.csv'.format(abnormal_digit), sep=';')
pass