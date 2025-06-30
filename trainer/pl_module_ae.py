import time
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
from model.focal_loss import FocalLoss
import time
from torch import nn

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
        # self.focal_loss = FocalLoss(gamma=2, alpha=[0.1]+[1 for _ in range(num_classes-1)], task_type='multi-class', num_classes=num_classes)
        # self.focal_loss = nn.CrossEntropyLoss()
        self.start_time = time.time()

    def _calculate_loss_acc(self, logits, y, metric):
        loss = F.cross_entropy(logits, y)
        acc = metric(logits, y)
        return loss, acc

    def _calculate_loss_mse(self, reconstructed, original):
        loss = F.mse_loss(reconstructed, original)
        
        # To calculate the pearson correlation coefficient, we first flatten the tensors
        # to treat them as one-dimensional vectors.
        reconstructed_flat = reconstructed.flatten()
        original_flat = original.flatten()
        
        # torch.corrcoef expects a single tensor where each row is a variable.
        # We stack the two flattened tensors to create a tensor of shape (2, N).
        input_tensor = torch.stack([reconstructed_flat, original_flat])
        
        # torch.corrcoef returns a 2x2 correlation matrix. The value at [0, 1]
        # is the Pearson correlation coefficient between the two original tensors.
        pearson_corr = torch.corrcoef(input_tensor)[0, 1]
        
        return loss, pearson_corr
    
    # def _calculate_loss_acc(self, logits, y, metric):
    #     loss = self.focal_loss(logits, y)
    #     acc = metric(logits, y)
    #     return loss, acc

    def training_step(self, batch, batch_idx):
        contig, emb, y = batch['contig'], batch['emb'], batch['population']
        results = self.model(contig, emb)
        # results = self.model(emb1, emb2)
        logits = results['logits'] # model outputs logits directly, no need for CLS token extraction
        loss0, acc0 = self._calculate_loss_acc(logits, y, self.metric)
        decompressed = results['decompressed']
        loss1, pearson_corr1 = self._calculate_loss_mse(decompressed, emb)
        loss = loss0*self.config.beta0 + loss1*self.config.beta1
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss0', loss0, prog_bar=True)
        self.log('train/loss1', loss1, prog_bar=True)
        self.log('train/acc0', acc0, prog_bar=True)
        self.log('train/pearson_corr1', pearson_corr1, prog_bar=True)
        self.train_step += 1
        self.log('train_step', self.train_step, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        self.log('train/seconds', time.time() - self.start_time, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        contig, emb, y = batch['contig'], batch['emb'], batch['population']
        results = self.model(contig, emb)
        # results = self.model(emb1, emb2)
        logits = results['logits'] # model outputs logits directly, no need for CLS token extraction
        loss0, acc0 = self._calculate_loss_acc(logits, y, self.metric)
        decompressed = results['decompressed']
        loss1, pearson_corr1 = self._calculate_loss_mse(decompressed, emb)
        loss = loss0 + loss1
        confusion_matrix = self.get_confusion_matrix(logits, y)
        outputs = {
            'confusion_matrix': confusion_matrix,
        }

        # for checkpointing
        self.log('val_acc', acc0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        # for logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/loss0', loss0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/loss1', loss1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/acc0',   acc0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/pearson_corr1', pearson_corr1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
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
