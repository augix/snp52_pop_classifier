from config.default import config
config.name = 'compress_emb'
config.model_py = 'model/deepseek_with_be_compress_emb.py'
config.model_wrapper_py = 'trainer/pl_module_2loss.py'
config.strategy = 'ddp'  # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'

# logging 
config.use_wandb = True
config.wandb = {
    'project': 'snp_emb',                 # W&B project name
    'tags': [config.name],             # Tags for the run
    'group': 'mirror_seq',             # Group name for the run
}

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = False                          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)