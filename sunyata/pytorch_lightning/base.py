import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sunyata.pytorch.arch.base import BaseCfg, RevSGD


class BaseModule(pl.LightningModule):
    def __init__(self, cfg:BaseCfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
    
    def _step(self, batch, mode):
        pass

    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "RevSGD":
            optimizer = RevSGD(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        # elif self.cfg.learning_rate_scheduler == "OneCycleLR":
        #     lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
        #         steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs, last_epoch=self.cfg.last_epoch)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr, last_epoch=self.cfg.last_epoch)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]   


class ClassifierModule(BaseModule):
    def __init__(self, cfg:BaseCfg):
        super().__init__(cfg)

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss    


class BYOL_EMA(pl.Callback):
    """
    Exponential Moving Average of BYOL 
    """
    def __init__(
        self, 
        student_name: str,
        teacher_name: str,
        initial_tau: float=0.999, 
        do_tau_update: bool=True):
        super().__init__()
        self.student_name = student_name
        self.teacher_name = teacher_name
        self.initial_tau = initial_tau
        self.current_tau = initial_tau
        self.do_tau_update = do_tau_update

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        student = getattr(pl_module, self.student_name) # pl_module.online_encoder
        teacher = getattr(pl_module, self.teacher_name) # pl_module.target_encoder

        self.update_weights(student, teacher)

        if self.do_tau_update:
            self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, student: nn.Module, teacher: nn.Module):
        for student_params, teacher_params in zip(student.parameters(), teacher.parameters()):
            teacher_params.data = self.current_tau * teacher_params.data + (1 - self.current_tau) * student_params.data