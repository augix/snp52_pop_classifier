from config.default import config
config.name = 'contig1024_d512_cd8_cs16'
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

config.contig_len = 1024 
config.contig_step = 1024 # Step size between contigs 
config.contig_list = [30,40,50,60]
config.d_model = 512
config.d_ffn = 512*2
config.cd = 8
config.cs = 16
config.lr = 4e-3