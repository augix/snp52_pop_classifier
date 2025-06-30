from config.default import config
config.name = 'longformer_contig1024_d512_cd8_cs16'
config.model_py = 'model/longformer_with_be_compress_emb.py'
config.model_wrapper_py = 'trainer/pl_module_2loss.py'
config.strategy = 'ddp'  # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'

# logging 
config.use_wandb = True
config.wandb = {
    'project': 'snp_emb',                 # W&B project name
    'tags': [config.name, 'no_pretrain'],             # Tags for the run
    'group': 'longformer',             # Group name for the run
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

config.contig_len = 1024 
config.contig_step = 1024 # Step size between contigs 
config.contig_list = [30,40,50,60]
config.n_layers            = 16
config.n_heads             = 16
config.d_model             = 512
config.d_id                = 32 # 2^32 = 4,294,967,296 = 4 billion
config.d_value             = 4
config.d_ffn               = 512*2
config.attention_window    = 128    # seqlen % (w * 2) == 0 # this better be bigger than the dim of the model
config.cd = 8
config.cs = 16
config.lr = 4e-3
config.add_cls = False
config.beta1 = 1
config.beta0 = 1
config.freeze_layers = ['embed', 'id_embed', 'correct_dim', 'layers']