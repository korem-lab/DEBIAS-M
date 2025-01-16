#!/usr/bin/env python

import numpy as np
import pandas as pd
from debiasm import DebiasMClassifier, DebiasMRegressor, MultitaskDebiasMClassifier, MultitaskDebiasMRegressor, OnlineDebiasMClassifier, DebiasMClassifierLogAdd
import unittest
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.functional import mse_loss, l1_loss

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
        
        self.assertTrue( roc < .65 and roc > .35 )
        
        
        
        
        dmc = DebiasMClassifier(x_val=X_val, 
                                prediction_loss=mse_loss) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        roc= roc_auc_score(y_val, dmc.predict_proba(X_val)[:, 1])
        ## should be ~~0.5 in this notebook , since the data is all random
        
        self.assertTrue( roc < .65 and roc > .35 )
        
        
        ## make multiclass labels in {0,1,2}
        y_multiclass = ( np.random.rand(n_samples)>0.5 ) + ( np.random.rand(n_samples)>0.5 )
        y_train, y_val = y_multiclass[~val_inds], y_multiclass[val_inds]
        dmc = DebiasMClassifier(x_val=X_val) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        self.assertTrue( dmc.predict_proba(X_val).shape[1] == np.unique(y_train).shape[0]) 
        
        
    def test_DebiasMClassifierLogadd(self):
        np.random.seed(123)
        n_samples = 96*5
        n_batches = 5
        n_features = 100

        ## the read count matrix
        X = ( np.random.rand(n_samples, n_features) * 1000 )#.astype(int)
        
        X = np.log10( 1 + X ) ## just some mapping into logspace

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


        dmc = DebiasMClassifierLogAdd(x_val=X_val) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        roc= roc_auc_score(y_val, dmc.predict_proba(X_val)[:, 1])
        ## should be ~~0.5 in this notebook , since the data is all random
        
        self.assertTrue( roc < .65 and roc > .35 )
        
        
        dmc = DebiasMClassifierLogAdd(x_val=X_val, 
                                prediction_loss=mse_loss) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        roc= roc_auc_score(y_val, dmc.predict_proba(X_val)[:, 1])
        ## should be ~~0.5 in this notebook , since the data is all random
        
        self.assertTrue( roc < .65 and roc > .35 )
        
        
        ## make multiclass labels in {0,1,2}
        y_multiclass = ( np.random.rand(n_samples)>0.5 ) + ( np.random.rand(n_samples)>0.5 )
        y_train, y_val = y_multiclass[~val_inds], y_multiclass[val_inds]
        dmc = DebiasMClassifier(x_val=X_val) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmc.fit(X_train, y_train)
        
        self.assertTrue( dmc.predict_proba(X_val).shape[1] == np.unique(y_train).shape[0]) 
        
        
        
    def test_OnlineDebiasMClassifier(self):
        
        np.random.seed(123)
        ## import packages

        ## generate data for the example
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
        X_with_batch = np.hstack((batches[:, np.newaxis], X))
        ## set the valdiation batch to '4'
        val_inds = batches==4
        X_train, X_val = X_with_batch[~val_inds], X_with_batch[val_inds]
        y_train, y_val = y[~val_inds], y[val_inds]

        ### Run DEBIAS-M, using standard sklearn object methods
        odmc = OnlineDebiasMClassifier() ## give it the held-out inputs to account for
                                            ## those domains shifts while training
        odmc.fit(X_train, y_train)

        ## extract the 'DEBIAS-ed' data for other downstream analyses, if applicable 
        X_debiassed = odmc.transform(X_with_batch)
        roc= roc_auc_score(y_val, odmc.predict_proba(X_val)[:, 1]) ## drop the 'batch' columns
        ## should be ~~0.5 in this notebook , since the data is all random
        
        self.assertTrue( roc < .65 and roc > .35 )
    
    
    def test_DebiasMRegressor(self):
        np.random.seed(123)
        n_samples = 96*5
        n_batches = 5
        n_features = 100

        ## the read count matrix
        X = ( np.random.rand(n_samples, n_features) * 1000 ).astype(int)

        ## the labels
        y = np.random.rand(n_samples)

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


        dmr = DebiasMRegressor(x_val=X_val) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmr.fit(X_train, y_train)
        r2_= r2_score(y_val, dmr.predict(X_val))
        self.assertTrue( r2_ < .05 and r2_ > -.05 )
        
        dmr = DebiasMRegressor(x_val=X_val, 
                                prediction_loss=l1_loss) ## give it the held-out inputs to account for
                                            ## those domains shifts while training

        dmr.fit(X_train, y_train)
        r2_= r2_score(y_val, dmr.predict(X_val))
        self.assertTrue( r2_ < .05 and r2_ > -.05 )
        
    def test_MultitaskDebiasMClassifier(self):
        
        
        np.random.seed(123)
        n_samples = 96*5
        n_batches = 5
        n_features = 100
        n_tasks=10

        ## the read count matrix
        X = ( np.random.rand(n_samples, n_features) * 1000 ).astype(int)

        ## the labels
        y = np.random.rand(n_samples, n_tasks)>0.5 
        ## specify `-1` for entries where the label is unknown

        ## the batches
        batches = ( np.random.rand(n_samples) * n_batches ).astype(int)

        ## we assume the batches are numbered ints starting at '0',
        ## and they are in the first column of the input X matrices
        ## for now, you can just set the first column to all zeros if we have only one batch

        X_with_batch = np.hstack((batches[:, np.newaxis], X))
        X_with_batch[:5, :5]

        ## set the valdiation batch to '4'
        val_inds = batches==4
        X_train, X_val = X_with_batch[~val_inds], X_with_batch[val_inds]
        y_train, y_val = y[~val_inds], y[val_inds]

        multitask_model = MultitaskDebiasMClassifier(x_val=X_val)
        multitask_model.fit(X_train, y_train)

        predicted_scores = multitask_model.predict_proba(X_val)
        
        multitask_model = MultitaskDebiasMClassifier(x_val=X_val, 
                                                     prediction_loss=mse_loss)
        multitask_model.fit(X_train, y_train)

        predicted_scores = multitask_model.predict_proba(X_val)
        
        self.assertTrue( len(predicted_scores) == n_tasks )


    def test_MultitaskDebiasMRegressor(self):
        
        
        np.random.seed(123)
        n_samples = 96*5
        n_batches = 5
        n_features = 100
        n_tasks=10

        ## the read count matrix
        X = ( np.random.rand(n_samples, n_features) * 1000 ).astype(int)

        ## the labels
        y = np.random.rand(n_samples, n_tasks) 
        ## specify `-1` for entries where the label is unknown

        ## the batches
        batches = ( np.random.rand(n_samples) * n_batches ).astype(int)

        ## we assume the batches are numbered ints starting at '0',
        ## and they are in the first column of the input X matrices
        ## for now, you can just set the first column to all zeros if we have only one batch

        X_with_batch = np.hstack((batches[:, np.newaxis], X))
        X_with_batch[:5, :5]

        ## set the valdiation batch to '4'
        val_inds = batches==4
        X_train, X_val = X_with_batch[~val_inds], X_with_batch[val_inds]
        y_train, y_val = y[~val_inds], y[val_inds]

        multitask_model = MultitaskDebiasMRegressor(x_val=X_val)
        multitask_model.fit(X_train, y_train)
        predicted_scores = multitask_model.predict(X_val)
        
        
        multitask_model = MultitaskDebiasMRegressor(x_val=X_val, 
                                                     prediction_loss=l1_loss)
        multitask_model.fit(X_train, y_train)
        predicted_scores = multitask_model.predict(X_val)
        self.assertTrue( len(predicted_scores) == n_tasks )
        
    
if __name__ == '__main__':
    unittest.main()