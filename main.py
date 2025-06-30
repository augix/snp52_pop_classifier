#! /usr/bin/env python

# IMPORT LIBRARIES
import os
import torch
import time

# Set CUDA memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Set float32 matmul precision for better performance on A800
torch.set_float32_matmul_precision('medium')

# Set deterministic behavior for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# IMPORT MY FUNCTIONS
# from trainer.pl_data import data_wrapper
# from trainer.pl_module import model_wrapper
from trainer.pl_trainer import setup_trainer
from trainer.pl_load_ckpt import load_ckpt

def load_module_from_path(file_path):
    module_namespace = {}
    with open(file_path, 'r') as file:
        exec(file.read(), module_namespace)
    return type('Module', (), module_namespace)

# Main execution function
def prepare(config):
    """Main function to run the training process."""

    # DATA
    data_module = load_module_from_path(config.data_py)
    theDataModule = data_module.theDataModule
    pl_data = theDataModule(config)
    pl_data.setup(config.train_or_test)    
    
    # MODEL
    model_module = load_module_from_path(config.model_py)
    theModel = model_module.theModel
    model = theModel(config)

    # model wrapper
    model_wrapper_module = load_module_from_path(config.model_wrapper_py)
    model_wrapper = model_wrapper_module.model_wrapper
    pl_model = model_wrapper(config, model)

    # trainer
    pl_trainer = setup_trainer(config)

    return config, pl_data, pl_model, pl_trainer

def test(config):
    test_start_time = time.time()

    config, pl_data, pl_model, pl_trainer = prepare(config)

    print('running test..., save to', f'{config.test_outdir}')
    os.makedirs(f'{config.test_outdir}', exist_ok=True)

    test_loader = pl_data.test_dataloader()    
    if os.path.exists(config.ckpt_path):
        pl_trainer.test(pl_model, test_loader, ckpt_path=config.ckpt_path)
    else:
        pl_trainer.test(pl_model, test_loader)

    print('test done')
    test_end_time = time.time()
    print(f'test duration: {test_end_time - test_start_time} seconds')

def train(config):
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    config, pl_data, pl_model, pl_trainer = prepare(config)
    train_loader = pl_data.train_dataloader()
    val_loader = pl_data.val_dataloader()
    print('print a batch of train_loader')
    for batch in train_loader:
        print(batch)
        break
    if os.path.exists(config.ckpt_path) and config.ckpt_resume:
        pl_model = load_ckpt(config, pl_model)
    else:
        print(f'Not resume from ckpt, start from scratch')
    pl_trainer.fit(pl_model, train_loader, val_loader)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.py")
    parser.add_argument("--devices", type=str, default="auto")
    args = parser.parse_args()

    # print the time
    print(f'running main.py at time: {time.time()}')
    print(f'with args: {args}')

    import runpy
    globals_dict = runpy.run_path(args.config)
    config = globals_dict['config']
    config.devices = args.devices
    print(config)

    if config.train_or_test == 'test':
        config.test_outdir = f'results/{config.name}/test'
        test(config)
    else:
        train(config)