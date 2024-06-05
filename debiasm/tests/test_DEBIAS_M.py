#!/usr/bin/env python

import numpy as np
import pandas as pd
from debiasm import DebiasMClassifier
import unittest
from sklearn.metrics import roc_auc_score

class DMtest(unittest.TestCase):
    
    def test_DebiasMClassifier(self):
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

        ## we assume the batches are numbered ints starting at '0',
        ## and they are in the first column of the input X matrices
        ## for now, you can just set the first column to all zeros if we have only one batch

        X_with_batch = np.hstack((batches[:, np.newaxis], X))

        ## set the valdiation batch to '4'
        val_inds = batches==4
        X_train, X_val = X_with_batch[~val_inds], X_with_batch[val_inds]
        y_train, y_val = y[~val_inds], y[val_inds]


        dmc = DebiasMClassifier(x_val=X_val) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        roc= roc_auc_score(y_val, dmc.predict_proba(X_val)[:, 1])
        ## should be ~~0.5 in this notebook , since the data is all random
        
        self.assertTrue( roc > .65 and roc > .35 )
    
    
if __name__ == '__main__':
    unittest.main()