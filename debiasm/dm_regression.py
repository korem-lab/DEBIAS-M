

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

from pl_bolts.datamodules import SklearnDataModule

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


    
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from sklearn.base import BaseEstimator

def rescale(xx):
    return(xx/xx.sum(axis=1)[:, np.newaxis] )


def batch_weight_feature_and_nbatchpairs_scaling(strength, df_with_batch):
    nbs = df_with_batch.iloc[:, 0].nunique()
    w = nbs * (nbs - 1) / 2
    return(strength /( w * ( df_with_batch.shape[1] - 1 ) ) )


class PL_SBCMP_regression(pl.LightningModule):
    """Logistic regression model."""

    def __init__(
        self,
        X, 
        batch_sim_strength: float,
        weighting_l2_strength: float,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        y_loss_scaling=1,
        prediction_loss=F.mse_loss,
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
        
        self.linear = nn.Linear(in_features=self.hparams.input_dim, 
                                out_features=self.hparams.num_classes, 
                                bias=bias)
        
        self.X=X[:, 1:]
        self.bs=X[:, 0].long()
        self.unique_bs=self.bs.unique().long()
        self.n_batches=self.unique_bs.max()+1
        self.batch_weights = torch.nn.Parameter(data = torch.zeros(self.n_batches,
                                                                   input_dim))

        self.batch_sim_str=batch_sim_strength
        self.y_loss_scaling=y_loss_scaling
        self.prediction_loss=prediction_loss
        
        
    def forward(self, x: Tensor) -> Tensor:
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.batch_weights[batch_inds.long()] ) * x, p=1 )
        y_hat = self.linear(x)
        return y_hat

    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)
        
        
        # PyTorch mse loss
        loss = self.prediction_loss(y_hat.squeeze(-1), y,) * self.y_loss_scaling
        
#         print('MSE loss: {:.3e}'.format(
#             ( F.mse_loss(y_hat.squeeze(-1), y, reduction="sum") * self.y_loss_scaling ).item())
#              )  
        
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
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
            
#             print('Domain sim: {:.3e}'.format( 
#                     ( sum( [pairwise_distance(x1, a) for a in x1] ).sum() *\
#                                     self.batch_sim_str ).item()
#                                 ) )

        if self.hparams.weighting_l2_strength > 0:
            # L2 regularizer for weighting parameter
            l2_reg = self.batch_weights.pow(2).sum()
                  
#             print('WL2 reg: {:.3e}'.format(
#                 ( self.hparams.weighting_l2_strength * l2_reg ).item() ) )
            loss += self.hparams.weighting_l2_strength * l2_reg


        loss /= float( x.size(0) )
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
         # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = self.prediction_loss(y_hat.squeeze(-1), y, reduction="sum") * self.y_loss_scaling
                
        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
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
        acc = 0
        return {"test_loss": self.prediction_loss(y_hat, y),
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
    
    
    
def DEBIASM_regression_train_and_pred(X_train, 
                                      X_val, 
                                      y_train, 
                                      y_val,
                                      batch_sim_strength=1, 
                                      batch_size=None,
                                      learning_rate=0.005,
                                      l2_strength=0,
                                      includes_batches=False, 
                                      val_split=0.1, 
                                      test_split=0.1,
                                      w_l2=0, 
                                      optimizer=Adam, 
                                      min_epochs=25,
                                      y_loss_scaling=1,
                                      prediction_loss=F.mse_loss
                                      ):
    
    if batch_size is None:
        batch_size=X_train.shape[0]
    
        
    model = PL_SBCMP_regression(X = torch.tensor( np.vstack((X_train, X_val)) ),
                                batch_sim_strength = batch_sim_strength,
                                input_dim = X_train.shape[1]-1, 
                                num_classes = 1, 
                                batch_size = batch_size,
                                learning_rate = learning_rate,
                                l2_strength = l2_strength,
                                weighting_l2_strength=w_l2,
                                y_loss_scaling=y_loss_scaling,
                                prediction_loss=prediction_loss
                                )
    
    ## build pl dataloader
    dm = SklearnDataModule(X_train, 
                           y_train, 
                           val_split=val_split, 
                           test_split=test_split
                           )

    ## run training
    trainer = pl.Trainer(logger=False, 
                         checkpoint_callback=False,
                         callbacks=[EarlyStopping(monitor="val_loss", 
                                                  mode="min", 
                                                  patience=2)], 
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
    val_preds = model.forward( torch.tensor( X_val ).float() ).detach().numpy()
    
    ## return predictions and the model
    return( val_preds, model )


class DebiasMRegressor(BaseEstimator):
    """DebiasMRegressorr: an sklean-style wrapper for the DEBIAS-M torch implementation."""
    def __init__(self,
                 *, 
                 batch_str = 'infer',
                 mse_scaling = 'infer',
                 learning_rate=0.005, 
                 min_epochs=25,
                 l2_strength=0,
                 w_l2=0,
                 random_state=None,
                 x_val=0, 
                 prediction_loss=F.mse_loss
                 ):
        
        self.learning_rate=learning_rate
        self.min_epochs=min_epochs
        self.l2_strength=l2_strength
        self.w_l2=w_l2
        self.batch_str=batch_str
        self.x_val = x_val
        self.mse_scaling=mse_scaling
        self.random_state=random_state
        self.prediction_loss=prediction_loss
        
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        if type(X)==pd.DataFrame:
            x = X.values
        else:
            x = X
            
        if type(self.x_val)==pd.DataFrame:
            xval=self.x_val.values
        else:
            xval=self.x_val
        
        if self.batch_str=='infer':
            self.batch_str = batch_weight_feature_and_nbatchpairs_scaling(1e4, pd.DataFrame( np.vstack((X, self.x_val ) )))
            
        if self.mse_scaling=='infer':
            self.mse_scaling = 1e3/np.var(y)
        
            
        self.classes_ = np.unique(y)
        preds, mod = DEBIASM_regression_train_and_pred(
                                                       x, 
                                                       xval, 
                                                       y, 
                                                       0,
                                                       batch_sim_strength = self.batch_str,
                                                       learning_rate=self.learning_rate,
                                                       min_epochs= self.min_epochs,
                                                       l2_strength=self.l2_strength,
                                                       w_l2 = self.w_l2, 
                                                       y_loss_scaling=self.mse_scaling, 
                                                        prediction_loss=self.prediction_loss
                                                       )
        self.model = mod
        self.val_preds = preds
        
        return self
    
    def transform(self, X):
        if self.model is None:
            raise(ValueError('You must run the `.fit()` method before executing this transformation'))
            
        if type(X)==pd.DataFrame:
            x = torch.tensor(X.values)
        else:
            x = torch.tensor(X)
            
            
        batch_inds, x = x[:, 0], x[:, 1:]
        x = F.normalize( torch.pow(2, self.model.batch_weights[batch_inds.long()] ) * x, p=1 )
        
        if type(X)==pd.DataFrame:
            return( pd.DataFrame(x.detach().numpy(), 
                                 index=X.index, 
                                 columns=X.columns[1:]))
        else:
            return( x.detach().numpy() )

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
        
        if type(X)==pd.DataFrame:
            x = torch.tensor(X.values)
        else:
            x = torch.tensor(X)
        
        y = self.model.forward( x.float() ).detach().numpy()
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