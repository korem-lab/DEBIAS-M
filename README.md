# DEBIAS-M - Domain adaptation with phenotype Estimation and Batch Integration Across Studies of the Microbiome

<!-- badges: start -->
[![main](https://github.com/korem-lab/DEBIAS-M/actions/workflows/main.yml/badge.svg)](https://github.com/korem-lab/DEBIAS-M/actions/workflows/main.yml)
<!-- badges: end -->

<img src='vignettes/DEBIAS-M-logo.png' align="right" height="250" />

Welcome to DEBIAS-M! This is a python package for processing bias correction in microbiome studies, that facilitates data harmonization, domain adaptation, predictive modeling, and batch correction. Uses samples from multiple studies in a count or relative abundance matrix format. Visit the [DEBIAS-M website](https://korem-lab.github.io/DEBIAS-M/) for the most detailed documentation.


## Installation
DEBIAS-M can be installed with pip through the following command:
`pip install git+https://github.com/korem-lab/DEBIAS-M.git`
As DEBIAS-M is a light package, the install time requires less than a minute on a standard machine.


## System requirements
Per our `setup.py` file: `python<3.11`,`numpy`, `pandas`, `torch==1.10.2`, `pytorch-lightning==1.5.10`, and `lightning-bolts==0.4.0`. No non-standard hardware is required for DEBIAS-M.

## Instructions for use
To begin, we recommend running the example demo we provide, which runs DEBIAS-M on a randomly generated dataset. We offer examples for all DEBIAS-M classes in the [DEBIAS-M website](https://korem-lab.github.io/DEBIAS-M/). 

## DEBIAS-M Demo
See the `Example.ipynb` notebook to see how to use the package. This notebook demonstrates the inputs and outputs of DEBIAS-M implementation (which mimicks the standard scikit-learn structure), and fits a DEBIAS-M model on the synthetic data. For the generated example of 480 samples and 100 features, the DEBIAS-M example completes in less than a minute. We provide a copy of the walkthrough code below. For further details on reproducing all results on our main analyses, please refer to our github repository containing the analysis code: https://github.com/korem-lab/v1-DEBIAS-M-Analysis/. Refer to the `Multitask-example.ipynb` notebook for an example running DEBIAS-M on multiple phenotypes at once.

```python
import numpy as np
from debiasm import DebiasMClassifier
```

### Build synthetic data for the example

The single-task DEBIAS-M classifier requires two inputs: 1) an X matrix, of dimension `(n_samples) x (1 + n_features)`, whose first column denotes the batch the sample originates from, and whose remaining columns describe the read counts observed for the corresponding sample-taxon pair; and 2) a binary y vector, of length `(n_samples)`, providing the classification label of each sample.

```python
np.random.seed(123)
n_samples = 96*5
n_batches = 5
n_features = 100

## the read count matrix
X = ( np.random.rand(n_samples, n_features) * 1000 ).astype(int)

## the labels
y = np.random.rand(n_samples)>0.5

## the batches
batches = ( np.random.rand(n_samples) * n_batches ).astype(int)
```


```python
## we assume the batches are numbered ints starting at '0',
## and they are in the first column of the input X matrices

X_with_batch = np.hstack((batches[:, np.newaxis], X))
X_with_batch[:5, :5]
```

    array([[  4, 696, 286, 226, 551],
           [  4, 513, 666, 105, 130],
           [  3, 542,  66, 653, 996],
           [  3,  16, 721,   7,  84],
           [  1, 456, 279, 932, 314]])


```python
y[:5]
```

    array([ True, False, False,  True,  True])


```python
## set the validation batch to '4'
val_inds = batches==4
X_train, X_val = X_with_batch[~val_inds], X_with_batch[val_inds]
y_train, y_val = y[~val_inds], y[val_inds]
```

### Run DEBIAS-M, using standard sklearn object formats


```python
y_train.shape
```




    (374,)




```python
X_train.shape
```




    (374, 101)




```python
dmc = DebiasMClassifier(x_val=X_val) ## give it the held-out inputs to account for
                                    ## those domains shifts while training

dmc.fit(X_train, y_train)
print('finished training!')
```
    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    /Users/george/opt/anaconda3/envs/pl2/lib/python3.6/site-packages/pytorch_lightning/trainer/data_loading.py:133: UserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      f"The dataloader, {name}, does not have many workers which may be a bottleneck."
    /Users/george/Desktop/sandbox/sandbox2/package_dev/DEBIAS-M/debiasm/torch_functions.py:75: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
      y_hat = softmax(x)
    /Users/george/opt/anaconda3/envs/pl2/lib/python3.6/site-packages/pytorch_lightning/trainer/data_loading.py:133: UserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      f"The dataloader, {name}, does not have many workers which may be a bottleneck."
    /Users/george/opt/anaconda3/envs/pl2/lib/python3.6/site-packages/pytorch_lightning/trainer/data_loading.py:433: UserWarning: The number of training samples (22) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      f"The number of training samples ({self.num_training_batches}) is smaller than the logging interval"
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...
    Trainer was signaled to stop but required minimum epochs (25) or minimum steps (None) has not been met. Training will continue...


    finished training!


### Assess results


```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, dmc.predict_proba(X_val)[:, 1]) 
## should be ~~0.5 in this example , since the data is all random
```



    0.4784313725490196




### Extract the the 'DEBIAS-ed' data

```python
X_debiassed = dmc.transform(X_with_batch)
X_debiassed[:5, :5]
```

    array([[0.01457723, 0.00534167, 0.00494237, 0.01204327, 0.01374663],
           [0.01068423, 0.01236932, 0.00228337, 0.00282551, 0.00610285],
           [0.01086351, 0.00140875, 0.01448351, 0.02095727, 0.01629219],
           [0.00031793, 0.01525674, 0.00015392, 0.00175223, 0.00472576],
           [0.00966502, 0.00629951, 0.01727931, 0.00599989, 0.01835283]])





