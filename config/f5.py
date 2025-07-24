fold = 'f5'

from config.ae import config
config.name = fold

# # == 1kG data ==
config.fn_labels_train = f'../task52_pop_classifier_data/res_s1_1kG_19pop/train_{fold}.tsv'
config.fn_labels_val  = f'../task52_pop_classifier_data/res_s1_1kG_19pop/val_{fold}.tsv'

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = False                          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)
