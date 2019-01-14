import os


def assert_files_exist(path, filename_list):
    for filename in filename_list:
        assert os.path.isfile(os.path.join(path, filename))


def assert_train_files_exist(path):
    assert_files_exist(path, ['fake_samples.png', 'log.pkl', 'time.jpg', 'train_disc_cost.jpg', 'train_gen_cost.jpg'])
