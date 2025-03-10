import warnings
warnings.filterwarnings("ignore",
                        ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", '.*In the future*')

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL)

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
import torch
def flatten(l):
    return [item for sublist in l for item in sublist]

def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import softmax, pairwise_distance
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy
from .pl_bolt_sklearn_module import SklearnDataModule
from sklearn.base import BaseEstimator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def rescale(xx):
    return(xx/xx.sum(axis=1)[:, np.newaxis] )

def to_categorical(y, num_classes=2):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class PL_DEBIAS_multitask(pl.LightningModule):
    """Mutitask DEBIAS-M model"""

    def __init__(
        self,
        X, 
        batch_sim_strength: float,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        w_l2 : float = 0.0,
        n_tasks = 2,
        prediction_loss=F.cross_entropy,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        
        if n_tasks <=1:
            raise(ValueError('only "{}" tasks specified, use base function if only a single task is needed'.format(n_tasks)))
        
        self.linear_weights = torch.nn.ModuleList([ 
                        nn.Linear(in_features=self.hparams.input_dim, 
                                  out_features=self.hparams.num_classes, 
                                  bias=bias) 
                                  for task in range(n_tasks) ])

        self.X=X[:, 1:]
        self.bs=X[:, 0].long()
        self.unique_bs=self.bs.unique().long()
        self.n_batches=self.unique_bs.max()+1
        self.batch_weights = torch.nn.Parameter(data = torch.zeros(self.n_batches,
                                                                   input_dim))

        self.batch_sim_str=batch_sim_strength
        self.prediction_loss=prediction_loss
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, 
                        p=1 )
        
        # a separate linear / softmax layer for each task
        y_hats = [softmax(self.linear_weights[i](x), dim=1)[:, -1] ## this implementation assumes all tasks are binary
                  for i in range(self.hparams.n_tasks)]
        return y_hats

    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hats = self.forward(x)
        
        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = sum( [ self.prediction_loss(y_hats[i][y[:, i]!=-1], 
                                           y[:, i][y[:, i]!=-1], 
                                           reduction="sum"
                                           )
                      for i in range(self.hparams.n_tasks) 
                    ] )
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum([ linear.weight.abs().sum()
                            for linear in self.linear_weights ])
            loss += self.hparams.l1_strength * l1_reg
        
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum([ linear.weight.pow(2).sum() \
                              for linear in self.linear_weights ])
            loss += self.hparams.l2_strength * l2_reg
            
        # DOMAIN similarity regularizer / bias correction
        if self.batch_sim_str > 0:
            x1 = torch.stack( [ ( torch.pow(2, self.batch_weights\
                                          )[torch.where(self.unique_bs==a)[0]] * \
                    (self.X[ torch.where(self.bs==a)[0] ] )  \
                   ).mean(axis=0) for a in self.unique_bs ] )

            x1=F.normalize(x1, p=1)

            loss += sum( [pairwise_distance(x1, a) for a in x1] ).sum() *\
                                    self.batch_sim_str


            
        # L2 regularizer for bias weight    
        if self.hparams.w_l2 > 0:
            # L2 regularizer for weighting parameter
            l2_reg = self.batch_weights.pow(2).sum()
            loss += self.hparams.w_l2 * l2_reg
            
        

        loss /= float( x.size(0) )
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hats = self.forward(x)
                               
         # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = sum( [ self.prediction_loss(y_hats[i][y[:, i]!=-1], 
                                      y[:, i][y[:, i]!=-1], 
                                      reduction="sum"
                                     )
                      for i in range(self.hparams.n_tasks) 
                    ] )
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum([ linear.weight.abs().sum()
                            for linear in self.linear_weights ])
            loss += self.hparams.l1_strength * l1_reg
        
        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum([ linear.weight.pow(2).sum() \
                              for linear in self.linear_weights ])
            loss += self.hparams.l2_strength * l2_reg
        
        self.log('val_loss', loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        return {"val_loss": val_loss}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        acc = 0#
        return {"test_loss": F.cross_entropy(y_hat, y),
                "acc": acc}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = 0
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_ce_loss": test_loss, "test_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"test_loss": test_loss,
                "log": tensorboard_logs,
                "progress_bar": progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=100)
        
        return parser
    
    
def DEBIASM_mutlitask_train_and_pred(X_train, 
                                     X_val, 
                                     y_train, 
                                     y_val,
                                     batch_sim_strength=1, 
                                     batch_size=None,
                                     w_l2 = 0,
                                     learning_rate=0.005,
                                     l2_strength=0,
                                     includes_batches=False,
                                     val_split=0.1, 
                                     test_split=0,
                                     min_epochs=15,
                                     prediction_loss=F.cross_entropy
                                     ):
    n_tasks = y_train.shape[1]
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mods=[]
    for i in range(n_tasks):

        inds_tmp = y_train[:, i] > -1
        baseline_mod = LogisticRegression(max_iter=2500)
        baseline_mod.fit(rescale( X_train[:, 1:][inds_tmp]), 
                         y_train[:, i][inds_tmp].astype(int))
        baseline_mods.append(baseline_mod)
        
        
    model = PL_DEBIAS_multitask(X = torch.tensor( np.vstack((X_train, X_val)) ),
                                batch_sim_strength = batch_sim_strength,
                                input_dim = X_train.shape[1]-1, 
                                num_classes = 2, 
                                batch_size = batch_size,
                                learning_rate = learning_rate,
                                l2_strength = l2_strength,
                                n_tasks=n_tasks,
                                w_l2 = w_l2, 
                                prediction_loss=prediction_loss
                                )
    
    # initialize parameters to lbe similar to standard logistic regression
    for i in range(n_tasks):
        try:
            model.linear_weights[i].weight.data[0]= \
                            -torch.tensor( baseline_mods[i].coef_[0] )
            model.linear_weights[i].weight.data[1]= \
                             torch.tensor( baseline_mods[i].coef_[0] )
        except:
            pass

    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train.astype(float),
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(
                         logger=False, 
                         enable_checkpointing=False,
                         callbacks=[EarlyStopping(monitor="val_loss", 
                                                  mode="min", 
                                                  patience=2)], 
                         check_val_every_n_epoch=2, 
                         enable_model_summary=False, 
                         enable_progress_bar=False,
                         min_epochs=min_epochs, 
                         max_epochs=1000
                         )
    trainer.fit(model, 
                train_dataloaders=dm.train_dataloader(), 
                val_dataloaders=dm.val_dataloader()
                )
    return(model)


class MultitaskDebiasMClassifier(BaseEstimator):
    """MultitaskDebiasMClassifier: an sklean-style wrapper for the Muiltitask DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None,
                 x_val=0,
                 prediction_loss=F.cross_entropy
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.x_val = x_val
        self.random_state=random_state
        self.prediction_loss=prediction_loss
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, pd.DataFrame(np.vstack((X, self.x_val)) ) )
            
        self.classes_ = np.unique(y)
        
        self.model = DEBIASM_mutlitask_train_and_pred(
                                                      X, 
                                                      self.x_val, 
                                                      y, 
                                                      0,
                                                      batch_sim_strength = self.batch_str,
                                                      learning_rate=self.learning_rate,
                                                      min_epochs= self.min_epochs,
                                                      l2_strength=self.l2_strength,
                                                      w_l2 = self.w_l2,
                                                      prediction_loss=self.prediction_loss
                                                      )
        
        return self

    def predict(self, X):
        """Perform multitask classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_all : n_task-length list of array-like of shape (n_samples,)
                    Predicted classisifcations for X.
        """
        y = [ (a[:, 1]>0.5).detach().numpy()
              for a in self.model.forward( torch.tensor( X ).float() )
                 ]
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
        P : list of ndarray of shape (n_samples, 2)
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        
        P = [ a.detach().numpy()
              for a in self.model.forward( torch.tensor( X ).float() )
                 ]
        return P

    
    def transform(self, X):
        """
        Return the esimtimated debiased X values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
         X_debiased : array-like of shape (n_samples, n_features)
            Test data.
        """
        
        if self.model is None:
            raise(ValueError('You must run the `.fit()` method before executing this transformation'))
            
        x = torch.tensor( np.array(X) )
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.model.batch_weights[batch_inds.long()] ) * x, p=1 )
        return( x.detach().numpy() )
