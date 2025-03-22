import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join( os.path.dirname(__file__), fname) ).read()


setup(
    name='DEBIAS-M',
    version='0.0.2',
    author='George Austin', 
    author_email='gia2105@columbia.edu', 
    url='https://github.com/korem-lab/DEBIAS-M',
    packages=['debiasm'],
    include_package_data=True,
    python_requires='>3.6.0',
    install_requires=[
                      'numpy<=1.26.4', 
                      'pandas',
                      'scikit-learn', 
                      'torch',
                      'pytorch-lightning>1.6',
                      'lightning-bolts>0.4.0'
                      ])
                      
