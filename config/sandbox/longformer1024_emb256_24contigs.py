from config.default import config

# main
config.contig_len = 1024 
config.mask_frac = 0.85 
config.data_py = 'data/data_contigs.py'
config.model_py = 'model/longformer_with_be_compress_emb.py'
config.model_wrapper_py = 'trainer/pl_module_2loss.py'
config.strategy = 'ddp_find_unused_parameters_true'  # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'

# Name of the run
config.name = f'longformer_train_emb256_24xcontig{config.contig_len}'

# logging 
config.use_wandb = False
config.wandb = {
    'project': 'snp_emb',                 # W&B project name
    'tags': [config.name],             # Tags for the run
    'group': 'group',             # Group name for the run
}

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
# config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_path = '../task36_train_chr22/results/pretrain_contig82k_test/ckpt/epoch=01-val_acc=0.94975.ckpt'                          # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = False                          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)

# model
config.contig_step = config.contig_len # Step size between contigs 
config.contig_list = [i*3 for i in range(1, 24)]
# config.contig_list = None
config.n_layers = 16
config.n_heads = 16
config.d_model = 512
config.d_ffn = 512*2
config.cd = 256
config.cs = 1
config.lr = 4e-4
config.bs_train = 20
config.bs_val = 20
config.val_check_interval = 0.001
config.log_every_n_steps = 4
config.mask_low = config.mask_frac
config.mask_high = config.mask_frac
config.add_cls = False
config.n_cls = 1
config.dropout = 0.1
# config.freeze_layers = ['embed', 'correct_dim', 'layers']

# longformer
config.attention_window = 128
config.attention_dilation = 1
