from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from Multi_VAE.metric import NDCG_binary_at_k_batch, Recall_at_k_batch
from lightning import LightningModule


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE


class MultiVaeModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = loss_function_vae

        # metric
        self.train_loss = MeanMetric()
        self.ndcg_50 = MeanMetric()
        self.recall_50 = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.train_loss.reset()
        self.ndcg_50.reset()
        self.recall_50.reset()

    def model_step(self, batch: Any):
        recon_batch, mu, logvar = self.forward(batch)
        if self.net.total_anneal_steps > 0:
            anneal = min(
                self.net.anneal_cap,
                1.0 * self.net.update_count / self.net.total_anneal_steps,
            )
        else:
            anneal = self.anneal_cap

        return recon_batch, mu, logvar, anneal

    def training_step(self, batch: Any, batch_idx: int):
        recon_batch, mu, logvar, anneal = self.model_step(batch)
        self.net.update_count += 1
        loss = self.criterion(recon_batch, batch, mu, logvar, anneal)
        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        self.log("train/loss", self.train_loss.compute(), sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        batch, heldout_data = batch
        recon_batch, mu, logvar, anneal = self.model_step(batch)
        loss = self.criterion(recon_batch, batch, mu, logvar, anneal)

        n50 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
        r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

        self.ndcg_50(n50.mean())
        self.recall_50(r50.mean())

    def on_validation_epoch_end(self):
        ndcg = self.ndcg_50.compute()
        recall = self.recall_50.compute()
        self.log_dict(
            {
                "val/ndcg@50": ndcg,
                "val/recall@50": recall,
            },
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/epoch_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FmModule(None, None, None)
