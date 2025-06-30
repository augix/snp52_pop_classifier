from config.default import config
config.contig_len = 4096 
config.mask_frac = 0.85 
config.name = f'local_id_BE_compress_emb_contig{config.contig_len}_mask{config.mask_frac}'
config.data_py = 'data/data_contigs_local_id.py'
config.model_py = 'model/deepseek_with_be_compress_emb.py'
config.model_wrapper_py = 'trainer/pl_module_2loss.py'
config.strategy = 'ddp'  # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'

# logging 
config.use_wandb = True
config.wandb = {
    'project': 'snp_emb',                 # W&B project name
    'tags': [config.name],             # Tags for the run
    'group': 'local id',             # Group name for the run
}

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = True                          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)

config.contig_step = config.contig_len # Step size between contigs 
config.contig_list = [10]
config.n_layers = 8
config.n_heads = 8
config.d_model = 128
config.d_ffn = 128*2
config.cd = 128
config.cs = 1
config.lr = 2e-3
config.bs_train = 12
config.bs_val = 12
config.val_check_interval = 0.01
config.log_every_n_steps = 4
config.mask_low = config.mask_frac
config.mask_high = config.mask_frac
config.add_cls = False
config.n_cls = 1
config.dropout = 0.1
