from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
import time
from trainer.plots import plot_predictions
import torch.distributed as dist

# PyTorch Lightning module for model and trainer
class model_wrapper(LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        # Create separate metrics for training and validation
        self.metric = Accuracy(task='multiclass', num_classes=self.config.n_output_values)
        self.get_confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.config.n_output_values)
        self.register_buffer('train_step', torch.tensor(0))
        self.correct_tokens = 0
        self.test_tokens = 0

    def _select_masked_positions(self, logits, y, mask):
        # select masked positions
        mask = mask.bool()        
        logits_masked = logits[mask, :]
        y_masked = y[mask]
        return logits_masked, y_masked

    def _calculate_loss_acc(self, logits, y, metric):
        # logits.shape: [batch, seqlen, n_output_values]
        # y.shape: [batch, seqlen]
        # reshape logits to [batch*seqlen, n_output_values]
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = F.cross_entropy(logits, y)
        acc = metric(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        x, id = batch['input'], batch['pos_id']
        result = self.model(x, id)
        logits = result['logits']
        logits0 = result['logits0']
        y = batch['target']
        # mask = batch['mask']
        # logits0, _ = self._select_masked_positions(logits0, y, mask)
        # logits, y = self._select_masked_positions(logits, y, mask)
        loss, acc = self._calculate_loss_acc(logits, y, self.metric)
        loss0, acc0 = self._calculate_loss_acc(logits0, y, self.metric)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss0', loss0, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        self.log('train/acc0', acc0, prog_bar=True)
        loss = self.config.beta1 * loss + self.config.beta0 * loss0
        self.train_step += 1
        self.log('train_step', self.train_step, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, id = batch['input'], batch['pos_id']
        result = self.model(x, id)
        logits = result['logits']
        logits0 = result['logits0']
        y = batch['target']
        # mask = batch['mask']
        # logits0, _ = self._select_masked_positions(logits0, y, mask)
        # logits, y = self._select_masked_positions(logits, y, mask)
        loss, acc = self._calculate_loss_acc(logits, y, self.metric)
        loss0, acc0 = self._calculate_loss_acc(logits0, y, self.metric)
        
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        confusion_matrix = self.get_confusion_matrix(logits, y)
        outputs = {
            'confusion_matrix': confusion_matrix,
        }

        if self.config.platform == 'v5000':
            reduce_fx="mean-v"
        else:
            reduce_fx="mean"

        # for checkpointing
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False, reduce_fx=reduce_fx)
        # for logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, reduce_fx=reduce_fx)
        self.log('val/loss0', loss0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, reduce_fx=reduce_fx)
        self.log('val/acc', acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, reduce_fx=reduce_fx)
        self.log('val/acc0', acc0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, reduce_fx=reduce_fx)
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

    def test_step(self, batch, batch_idx):
        start_time = time.time()
        x, id, y, seq_idx = batch['input'], batch['pos_id'], batch['target'], batch['seq_idx']
        result = self.model(x, id)
        logits = result['logits']
        emb = result['emb']
        end_time = time.time()
        duration = end_time - start_time
        print(f'duration: {duration:.4f}s')

        # only select the determined target values for accuracy calculation
        y_for_acc = y[y != self.config.mask_token]
        logits_for_acc = logits[y != self.config.mask_token, :]
        loss, acc = self._calculate_loss_acc(logits_for_acc, y_for_acc, self.metric)
        counts = y_for_acc.numel()
        self.correct_tokens += acc * counts
        self.test_tokens += counts
        # print(f'test_step: correct_tokens: {self.correct_tokens}, test_tokens: {self.test_tokens}')
        if hasattr(self.config, 'save_test_results') and self.config.save_test_results:
            # save the results
            pred = logits.argmax(dim=-1)
            # confidence_scores = self._get_confidence_scores(logits)
            outputs = {
                # 'yhat': pred,
                # 'batch': batch,
                # 'confidence_scores': confidence_scores,
                'person': batch['person'],
                'chr': self.config.chr,
                'contig': batch['contig'],
                'acc': acc,
                'emb': emb,
            }
            torch.save(outputs, f'{self.config.test_outdir}/result_gpu{self.global_rank}_batch{batch_idx}.pt')
        if hasattr(self.config, 'plot_test_results') and self.config.plot_test_results:
            # plot predictions
            plot_predictions(f'{self.config.test_outdir}', batch['input'].detach().cpu(), batch['target'].detach().cpu(), pred.detach().cpu(), png_name=f'predictions_gpu{self.global_rank}_batch{batch_idx}.png')

        return None
    
    def on_test_epoch_end(self):
        if self.trainer.world_size > 1:
            # wait for all processes to finish
            dist.barrier()            
            # sum up acc from all GPUs 
            correct_tokens = torch.tensor(self.correct_tokens, device=self.device)
            dist.all_reduce(correct_tokens, op=dist.ReduceOp.SUM)
            test_tokens = torch.tensor(self.test_tokens, device=self.device)
            dist.all_reduce(test_tokens, op=dist.ReduceOp.SUM)
            print(f'test_epoch_end: correct_tokens: {correct_tokens}, test_tokens: {test_tokens}')
            acc = correct_tokens / test_tokens
            dist.destroy_process_group()
        else:
            acc = self.correct_tokens / self.test_tokens
        print(f'\n\n ===== test acc: {acc:.5f} ===== \n\n')

