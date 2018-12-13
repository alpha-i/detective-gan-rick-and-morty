import logging
import pandas as pd

from alphai_watson.datasource import AbstractDataSource


class KDDCup99DataSource(AbstractDataSource):
    """
    Implements a Datasource for the KDDCup99 10 Percent dataset
    """
    SAMPLE_TYPES = ['NORMAL', 'ABNORMAL']

    def __init__(self, source_file, transformer):
        super().__init__(source_file, transformer)

    @property
    def sample_rate(self):
        return 0

    def _read_samples(self):
        """
        Parses the source file for a give sample_type.
        """

        logging.debug("Start file parsing.")
        data = pd.read_csv(self._source_file, header=None)
        samples = {}
        samples['NORMAL'] = data.values

        logging.debug("End file parsing.")

        return samples
    
    def _extract_and_process_samples(self, sample_list):
        """
        Returns samples
        """
        return sample_list
