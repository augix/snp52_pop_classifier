from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Callback
import torch
from trainer.plots import plot_confusion_matrix, plot_predictions, plot_scatter, plot_confusion_matrix_with_bubbles
import torch.distributed as dist
import numpy as np

# Callbacks
def best_ckpt_callback(dirpath):
    return ModelCheckpoint(
        dirpath=dirpath,
        filename='{epoch:02d}-{val_acc:.5f}',
        save_top_k=3,
        monitor='val_acc',
        mode='max',
        save_last=True, # save the last checkpoint
        enable_version_counter=False, # enable saving multiple versions of the same result inside an epoch
    )

class confusion_plot_callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.confusion_matrices = []
        self.config = config

    def on_batch_end(self, outputs):
        self.confusion_matrices.append(outputs['confusion_matrix'])

    def on_epoch_end(self, trainer, pl_module):
        local_cm = torch.stack(self.confusion_matrices)
        local_cm = torch.sum(local_cm, dim=0)
        local_cm = local_cm.to(pl_module.device)
        if trainer.world_size > 1:
            dist.barrier()
            global_cm = local_cm.clone().detach()
            dist.all_reduce(global_cm, op=dist.ReduceOp.SUM)
        else:
            global_cm = local_cm.clone().detach()

        if trainer.is_global_zero:
            # convert to numpy
            global_cm = global_cm.detach().cpu().numpy()
            # calculate accuracy
            conf_sum = global_cm.sum()
            conf_correct = np.diag(global_cm).sum()
            acc = conf_correct / conf_sum * 100
            print(f'\n\n epoch{trainer.current_epoch}: acc: {acc:.4f}%, tested: {conf_sum}, correct: {conf_correct} \n\n')
            # plot confusion matrix
            outdir = self.config.outdir
            plot_confusion_matrix(global_cm, outdir, trainer.current_epoch)
            plot_confusion_matrix_with_bubbles(global_cm, outdir, trainer.current_epoch)
        # Clear the lists for the next epoch
        self.confusion_matrices.clear()    

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end(outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end(outputs)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module)
        
    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module)


class plot_predictions_callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Get a batch of data
            val_dataloaders = trainer.val_dataloaders
            if isinstance(val_dataloaders, list):
                dataloader = val_dataloaders[0]
            else:
                dataloader = val_dataloaders
            batch = next(iter(dataloader))

            # move to correct device
            batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}

            # model prediction
            # with torch.cuda.amp.autocast():
            x, id = batch['input'], batch['pos_id']
            result = pl_module.model(x, id)
            logits = result['logits']
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            input = batch['input'].detach().cpu().numpy()
            target = batch['target'].detach().cpu().numpy()
            outdir = self.config.outdir
            output_path = f'{outdir}/predictions_epoch{trainer.current_epoch}.png'
            plot_predictions(output_path, input, target, pred)

class plot_scatter_callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.yhats = []
        self.ys = []
        self.config = config

    def batch_end(self, trainer, outputs):
        if trainer.global_rank == 0:
            yhat = outputs['yhat'].detach().cpu()
            y = outputs['y'].detach().cpu()
            # Store as lists
            self.yhats.append(yhat)
            self.ys.append(y)

    def epoch_end(self, trainer):
        if trainer.global_rank == 0:
            # Concatenate predictions from all batches (handles different batch sizes)
            all_yhats = torch.cat(self.yhats, dim=0)
            all_ys = torch.cat(self.ys, dim=0)
            # convert to numpy
            all_yhats = all_yhats.to(torch.float32).numpy()
            all_ys = all_ys.to(torch.float32).numpy()
            outdir = self.config.outdir
            plot_scatter(outdir, all_ys, all_yhats, title=f'Scatter Plot Epoch {trainer.current_epoch}')

        # Clear the lists for the next epoch
        self.yhats.clear()
        self.ys.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batch_end(trainer, outputs)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.epoch_end(trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.batch_end(trainer, outputs)
        
    def on_test_epoch_end(self, trainer, pl_module):
        self.epoch_end(trainer)
