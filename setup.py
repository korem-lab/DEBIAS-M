import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join( os.path.dirname(__file__), fname) ).read()


setup(
    name='DEBIAS-M',
    version='0.0.1',
    author='George Austin', 
    author_email='gia2105@columbia.edu', 
    url='https://github.com/korem-lab/DEBIAS-M',
    packages=['debiasm'],
    include_package_data=True,
    python_requires='<3.11.0',
    install_requires=[
                      'numpy', 
                      'pandas',
                      'scikit-learn', 
                      'torch<=1.13.1',
                      'pytorch-lightning<=1.6',
                      'lightning-bolts==0.4.0'
                      ])
                      
