from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
from model.focal_loss import FocalLoss

# PyTorch Lightning module for model and trainer
class model_wrapper(LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        num_classes = self.config.n_output_values
        # Create separate metrics for training and validation
        self.metric = Accuracy(task='multiclass', num_classes=num_classes)
        self.get_confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.register_buffer('train_step', torch.tensor(0))
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.1]+[1 for _ in range(num_classes-1)], task_type='multi-class', num_classes=num_classes)

    # def _calculate_loss_acc(self, logits, y, metric):
    #     loss = F.cross_entropy(logits, y)
    #     acc = metric(logits, y)
    #     return loss, acc

    def _calculate_loss_acc(self, logits, y, metric):
        loss = self.focal_loss(logits, y)
        acc = metric(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        contig1, contig2, emb1, emb2, y = batch['contig1'], batch['contig2'], batch['emb1'], batch['emb2'], batch['kinship']
        # results = self.model(contig1, contig2, emb1, emb2)
        results = self.model(emb1, emb2)
        logits = results['logits'] # model outputs logits directly, no need for CLS token extraction
        loss, acc = self._calculate_loss_acc(logits, y, self.metric)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        self.train_step += 1
        self.log('train_step', self.train_step, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        contig1, contig2, emb1, emb2, y = batch['contig1'], batch['contig2'], batch['emb1'], batch['emb2'], batch['kinship']
        # results = self.model(contig1, contig2, emb1, emb2)
        results = self.model(emb1, emb2)
        logits = results['logits'] # model outputs logits directly, no need for CLS token extraction
        loss, acc = self._calculate_loss_acc(logits, y, self.metric)
        confusion_matrix = self.get_confusion_matrix(logits, y)
        outputs = {
            'confusion_matrix': confusion_matrix,
        }

        # for checkpointing
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        # for logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/acc',   acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_step', self.train_step, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer
    
    # logging graident norm
    def on_after_backward(self):
        if not self.config.log_grad_norm:
            return
        if self.global_step % self.config.log_every_n_steps != 0:
            return
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        # Calculate total norm
        total_norm = total_norm ** 0.5
        self.log('grad_norm', total_norm)
