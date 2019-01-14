import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from alphai_watson.datasource import AbstractDataSource


class KDDCup99DataSource(AbstractDataSource):
    """
    Implements a Datasource for the KDDCup99 10 Percent dataset
    """
    SAMPLE_TYPES = ['NORMAL', 'ABNORMAL']

    def __init__(self, source_file, header_file, transformer):
        self._header_file = header_file
        super().__init__(source_file, transformer)

    @property
    def sample_rate(self):
        return 10

    def _read_samples(self):
        """
        Parses the source file for a give sample_type.
        """

        logging.debug("Start file parsing.")
        data = pd.read_csv(self._source_file, header=None)
        
        data = pd.read_csv(self._source_file, header=None)
        header = pd.read_csv(self._header_file, delimiter=':', skiprows=1, header=None)
        header.columns = ['column', 'column_type']

        data.columns = header.column.tolist() + ['attack']
        data['attack'] = data['attack'].str.replace('.', '')
        data['label'] = 1
        data.loc[data['attack'] == 'normal', 'label'] = 0

        symbolic_columns = header.loc[header.column_type == ' symbolic.'].column.tolist()
        # print(symbolic_columns)

        for scol in symbolic_columns:
            data[scol] = pd.Categorical(data[scol])
            one_hot_cols = pd.get_dummies(data[scol], prefix=scol)
            data = pd.concat([data, one_hot_cols], axis=1)

        data = data.drop(columns=symbolic_columns)
        data = data.drop(columns=['attack'])

        # data.loc[data.attack != 'normal' , ['attack', 'label']].head(20)

        data_normal = data.loc[data['label'] == 0]
        data_abnormal = data.loc[data['label'] == 1]

        data_normal_train = data_normal.sample(frac=0.7)
        data_normal_test = data_normal.loc[~data_normal.index.isin(data_normal_train.index)]

        data_normal_train = data_normal_train.drop(columns=['label']).values
        data_normal_test = data_normal_test.drop(columns=['label']).values
        data_abnormal = data_abnormal.drop(columns=['label']).values
        
        scaler = MinMaxScaler()
        _ = scaler.fit(data_normal_train)
        data_normal_train = scaler.transform(data_normal_train)
        data_normal_test = scaler.transform(data_normal_test)
        data_abnormal = scaler.transform(data_abnormal)
        
        logging.debug('Normal {}; Train {}; Test{}'.format(data_normal.shape, data_normal_train.shape, data_normal_test.shape))
        logging.debug('Abnormal {}'.format(data_abnormal.shape))

        samples = {}
        samples['NORMAL'] = data_normal_train
        samples['NORMAL_TEST'] = data_normal_test
        samples['ABNORMAL_TEST'] = data_abnormal

        logging.debug("End file parsing.")

        return samples
    
    def _extract_and_process_samples(self, sample_list):
        """
        Returns samples
        """
        
        return sample_list
