import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from sklearn.base import BaseEstimator
from .torch_functions import DEBIASM_train_and_pred, DEBIASM_train_and_pred_log_additive

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

class DebiasMClassifier(BaseEstimator):
    """DebiasMClassifier: an sklean-style wrapper for the DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None,
                 x_val=0
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.x_val = x_val
        self.random_state=random_state
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, pd.DataFrame(X) )
            
        self.classes_ = np.unique(y)
        preds, mod = DEBIASM_train_and_pred(
                                            X, 
                                            self.x_val, 
                                            y, 
                                            0,
                                            batch_sim_strength = self.batch_str,
                                            learning_rate=self.learning_rate,
                                            min_epochs= self.min_epochs,
                                            l2_strength=self.l2_strength,
                                            w_l2 = self.w_l2
                                            )
        self.model = mod
        self.val_preds = preds
        
        return self

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """
        y = self.model.forward( torch.tensor( X ).float() ).detach().numpy()[:, 1]>0.5
        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        
        P = self.model.forward( torch.tensor( X ).float() ).detach().numpy()
        
        return P

class AdaptationDebiasMClassifier(BaseEstimator):
    """DebiasMClassifier: an sklean-style wrapper for the DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.random_state=random_state
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, pd.DataFrame(X) )
        
        self.classes_ = np.unique(y)
        preds, mod = DEBIASM_train_and_pred(
                                            X, 
                                            X, 
                                            y, 
                                            y,
                                            batch_sim_strength = self.batch_str/2,
                                            learning_rate=self.learning_rate,
                                            min_epochs= self.min_epochs,
                                            l2_strength=self.l2_strength,
                                            w_l2 = self.w_l2
                                            )
        self.model = mod
        self.train_X = pd.DataFrame(X)
        
        return self
    
    def transform(self, X):
        if self.model is None:
            raise(ValueError('You must run the `.fit()` method before executing this transformation'))
            
        x = torch.tensor(X.values)
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.model.batch_weights[batch_inds.long()] ) * x, p=1 )
        return( pd.DataFrame(x.detach().numpy(), 
                             index=X.index, 
                             columns=X.columns[1:]))


    
    def estimate_test_biases(self, x_test, iters=10000):
        
        x_base = pd.concat([ self.train_X.iloc[:, 0], 
                             self.transform(self.train_X)], 
                           axis=1 )

        x_base_means = F.normalize( torch.tensor( x_base.groupby(0)[x_base.columns].mean().iloc[:, 1:].values ),
                                   p=1 )

        input_dim=x_test.shape[1]
        test_bcfs = torch.nn.Parameter(data = torch.zeros(1,
                                                          input_dim)
                                      )

        optimizer=torch.optim.Adam([test_bcfs], 
                                   lr = 0.005
                                   )
        
        x_test_tensor = torch.tensor(x_test)
        
        for i in range(iters):                
            xt1 = F.normalize( x_test_tensor*torch.pow(2, test_bcfs), p=1)
            loss = sum( [pairwise_distance(xt1, a) 
                         for a in x_base_means] ).sum()*0.01 ## really just a lr scaling factor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        debiased_X_test = F.normalize( x_test_tensor * torch.pow(2, test_bcfs), p=1 )
        return(debiased_X_test)
        

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """
        debiased_X_test = self.estimate_test_biases(X)
        y = F.softmax( self.model.linear(debiased_X_test), dim=1 )[:, 1].detach().numpy() > 0.5
        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        debiased_X_test = self.estimate_test_biases(X)
        P = F.softmax( self.model.linear(debiased_X_test), dim=1 ).detach().numpy()
        return P
    
    
class DebiasMClassifierLogAdd(BaseEstimator):
    """DebiasMClassifier: an sklean-style wrapper for the DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None,
                 x_val=0
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.x_val = x_val
        self.random_state=random_state
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.classes_ = np.unique(y)
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e3, pd.DataFrame(X) )
        preds, mod = DEBIASM_train_and_pred_log_additive(
                                                X, 
                                                self.x_val, 
                                                y, 
                                                0,
                                                batch_sim_strength = self.batch_str,
                                                learning_rate=self.learning_rate,
                                                min_epochs= self.min_epochs,
                                                l2_strength=self.l2_strength,
                                                w_l2 = self.w_l2
                                                )
        self.model = mod
        self.val_preds = preds
        
        return self

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """
        y = self.model.forward( torch.tensor( X ).float() ).detach().numpy()[:, 1]>0.5
        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        
        P = self.model.forward( torch.tensor( X ).float() ).detach().numpy()
        
        return P
    
    
    
    
    