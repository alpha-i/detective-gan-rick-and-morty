import logging

import h5py

from alphai_watson.datasource import AbstractDataSource, Sample


class MNISTDataSource(AbstractDataSource):
    """
    Implements a Datasource for Kaggle Brainwaves data
    """
    SAMPLE_TYPES = ['NORMAL', 'ABNORMAL']

    def __init__(self, source_file, transformer):
        super().__init__(source_file, transformer)

    @property
    def sample_rate(self):
        return self._sample_rate

    def _read_samples(self):
        """
        Parses the source file for a give sample_type.
        Every sample should have the shape of (number_of_sensors, data_length)
        """

        logging.debug("Start file parsing")
        samples = {}
        with h5py.File(self._source_file, 'r') as _h5file:
            for sample_type in list(_h5file.keys()):
                _data = list(_h5file[sample_type])
                samples[sample_type] = _data

        self._sample_rate = 1.0  # FIXME
        logging.debug("end file parsing")

        return samples

    def get_train_data(self, sample_type):
        if sample_type == 'ALL':
            raw_samples = [self._raw_data[x] for x in list(self._raw_data.keys())]
            raw_samples = [item for sublist in raw_samples for item in sublist]
        else:
            raw_samples = self._raw_data[sample_type]

        assert len(raw_samples) > 0, "No training samples found."

        return Sample(
            data=self._extract_and_process_samples(raw_samples),
            sample_type=sample_type,
            sample_rate=self.sample_rate,
            number_of_timesteps=self._transformer.number_of_timesteps
        )
