from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_rickandmorty_oracle',
    version='0.0.8',
    description='Alpha-I Rick and Morty oracle',
    author='Fergus Simpson',
    author_email='fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'h5py==2.7.1',
        'tensorflow==1.4.0',
        'scipy',
        'sklearn',
        'contexttimer',
        'pandas==0.22',
        'tables==3.4.2',
        'alphai_watson>=0.1.2,<1.0.0',
        'matplotlib',
        'pillow',
    ],
    dependency_links=[]
)
