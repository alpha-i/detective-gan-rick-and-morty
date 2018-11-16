import os

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import pickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def tick():
    _iter[0] += 1


def add_to_plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(destination_path):
    prints = []

    for name, vals in list(_since_last_flush.items()):
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        image_name = os.path.join(destination_path, name.replace(' ', '_') + '.jpg')
        plt.savefig(image_name)

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    log_file_path = os.path.join(destination_path, 'log.pkl')
    with open(log_file_path, 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
