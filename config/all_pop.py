
from config.ae import config
config.name = 'all_pop'
config.n_output_values = 26               # Number of possible output values: 0-25 populations

# # == 1kG data ==
config.fn_labels_train = f'../task52_pop_classifier_data/res_s2/train0.tsv'
config.fn_labels_val  = f'../task52_pop_classifier_data/res_s2/val0.tsv'

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = False                          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)
