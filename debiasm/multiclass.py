import warnings
warnings.filterwarnings("ignore",
                        ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", '.*In the future*')

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

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

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .torch_functions import PL_DEBIASM

def rescale(xx):
    return(xx/xx.sum(axis=1)[:, np.newaxis] )

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
    
    
def DEBIASM_multiclass_train_and_pred(X_train, 
                                      X_val, 
                                      y_train, 
                                      y_val,
                                      batch_sim_strength=1,
                                      w_l2 = 0,
                                      batch_size=None,
                                      learning_rate=0.005,
                                      l2_strength=0,
                                      includes_batches=False,
                                      val_split=0.1, 
                                      test_split=0,
                                      min_epochs=15, 
                                      verbose=False, 
                                      prediction_loss=F.cross_entropy, 
                                      num_classes=3
                                      ):
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mod = LogisticRegression(max_iter=2500)
    baseline_mod.fit(rescale( X_train[:, 1:]), y_train)
        
        
    model = PL_DEBIASM(X = torch.tensor( np.vstack((X_train, X_val)) ),
                       batch_sim_strength = batch_sim_strength,
                       input_dim = X_train.shape[1]-1, 
                       num_classes = num_classes, 
                       batch_size = batch_size,
                       learning_rate = learning_rate,
                       l2_strength = l2_strength, 
                       w_l2 = w_l2, 
                       prediction_loss=prediction_loss
                       )
    
    ## initialize parameters to lbe similar to standard logistic regression
    model.linear.weight.data = torch.tensor( baseline_mod.coef_ ).float()
    
    
    y_train = torch.tensor( y_train ).long().detach().numpy() #due to windows numpy bug
    
    if prediction_loss!=F.cross_entropy:
        y_train=to_categorical(y_train, num_classes=y_train.max() + 1).astype(float)
        
    
    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train,
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(logger=False, 
                         checkpoint_callback=False,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)], 
                         check_val_every_n_epoch=2, 
                         weights_summary=None, 
                         progress_bar_refresh_rate=0,
                         min_epochs=min_epochs
                         )
    trainer.fit(model, 
                train_dataloaders=dm.train_dataloader(), 
                val_dataloaders=dm.val_dataloader()
               )
    
    ## get val predictions
    val_preds = model.forward( torch.tensor( X_val ).float() )[:, 1].detach().numpy()
    
    ## return predictions and the model
    return( val_preds, model )

    
def DEBIASM_multiclass_train_and_pred_log_additive(X_train, 
                         X_val, 
                         y_train, 
                         y_val,
                         batch_sim_strength=1,
                         w_l2 = 0,
                         batch_size=None,
                         learning_rate=0.005,
                         l2_strength=0,
                         includes_batches=False,
                         val_split=0.1, 
                         test_split=0,
                         min_epochs=15, 
                         prediction_loss=F.cross_entropy, 
                         num_classes=3
                         ):
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
    baseline_mod = LogisticRegression(max_iter=2500)
    baseline_mod.fit( X_train[:, 1:], y_train)
        
        
    model = PL_DEBIASM_log_additive(X = torch.tensor( np.vstack((X_train, X_val)) ),
                                    batch_sim_strength = batch_sim_strength,
                                    input_dim = X_train.shape[1]-1, 
                                    num_classes = num_classes, 
                                    batch_size = batch_size,
                                    learning_rate = learning_rate,
                                    l2_strength = l2_strength, 
                                    w_l2 = w_l2,
                                    prediction_loss=prediction_loss
                                    )
    
    ## initialize parameters to lbe similar to standard logistic regression
    model.linear.weight.data = torch.tensor( baseline_mod.coef_).float()
    
    y_train = torch.tensor( y_train ).long().detach().numpy() #due to windows numpy bug
    
    if prediction_loss != F.cross_entropy:
        y_train=to_categorical(y_train, num_classes=y_train.max() + 1).astype(float)

    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train,
                           val_split=val_split,
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(logger=False, 
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
    
    ## get val predictions
    val_preds = model.forward( torch.tensor( X_val ).float() )[:, 1].detach().numpy()
    
    ## return predictions and the model
    return( val_preds, model )

