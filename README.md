# Rick and Morty Detective

This repository contains an implementation of a Generative Adversarial Network (GAN) used for anomaly detection.

This repository doesn't contain standalone software but it is meant to work with Alpha-I's Watson and ADS for Aerospace project.

This is possible because the model is wrapped in a class which implements a Detective Interface


Repository structure
------------
The repository contains several folders, each one using a combination of datasource and models.

Directory structure:

```bash
alphai_rickandmorty_oracle/     # software package
    tflib/                      # folder containing support TensorFlow libraries
    detective.py                # implementation of the AbstractDetective interface
    model.py                    # implementation of the GAN neural network
doc/                            # folder containing documentation
tests/                          # folder containing the unit tests for the package
README.md                       # this file
requirements.txt                # dependencies for the package
dev-requirements.txt            # dependencies for the development environment
```

Code structure
------------
The main two python files of the package are `detective.py` and `model.py`.

`detective.py` contains the implementation of the AbstractDetective interface from `alphai_watson`, on top of which 
this repository in built. It overrides the abstract methods `train` and `detect` and the abstract property 
`configuration`. It also adds the method `diagnose` that returns the closest synthetic chunk to the input data that the 
model model was able to generate.

`model.py` contains the TensorFlow code implementation of the GAN. The `RickAndMorty` class defines the `generator` and 
the `discriminator` networks as methods, and the functions responsible for training the GAN, executing the Discriminator,
using the Generator to find the closest synthetic chunk to the input data, as well running performance evaluation, 
and saving and loading the model.


Installation
------------

```bash
$ conda create -n rickmorty python=3.5
$ source activate rickmorty
$ pip install -r dev-requirements.txt
```

To run test
```bash
$ pytest tests/
```
