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
    install_requires=['numpy', 
                      'pandas', 
                      'torch==1.10.2',
                      'pytorch-lightning==1.5.10',
                      'lightning-bolts==0.4.0'
                     ]
)
                      