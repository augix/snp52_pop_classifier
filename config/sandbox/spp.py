from argparse import Namespace
import torch
import os

config = Namespace(
    name = 'spp',
    platform = 'sanya',

    # data 
    data_py = 'data/data_emb.py',  # Path to data module
    # == pop CHS ==
    # fn_emb_train   = '../task48c_prepare_kinship_data_ped-sim/res_s3_ped-sim_40_trees_CHS/emb.pt',
    # fn_pairs_train = '../task48c_prepare_kinship_data_ped-sim/res_s4_ped-sim_40_trees_CHS/pairs.tsv',
    # # ped-sim val 4 trees pop CHS
    # fn_emb_val     = '../task48c_prepare_kinship_data_ped-sim/res_s3_ped-sim_4_trees_CHS/emb.pt',
    # fn_pairs_val   = '../task48c_prepare_kinship_data_ped-sim/res_s4_ped-sim_4_trees_CHS/pairs.tsv',

    # == all populations ==
    fn_emb_train   = '../task48c_prepare_kinship_data_ped-sim/res_s3_train_40_trees/emb.pt',
    fn_pairs_train = '../task48c_prepare_kinship_data_ped-sim/res_s4_train_40_trees/pairs.tsv',
    fn_emb_val     = '../task48c_prepare_kinship_data_ped-sim/res_s3_val_4_trees/emb.pt',
    fn_pairs_val   = '../task48c_prepare_kinship_data_ped-sim/res_s4_val_4_trees/pairs.tsv',

    # == small ped-sim data ==
    # fn_emb_train   = '../task48d_prepare_ped-sim_trees_4train_1val/res_s3_small_ped-sim_train_4_trees/emb.pt',
    # fn_pairs_train = '../task48d_prepare_ped-sim_trees_4train_1val/res_s4_small_ped-sim_train_4_trees/pairs.tsv',
    # fn_emb_val     = '../task48d_prepare_ped-sim_trees_4train_1val/res_s3_small_ped-sim_val_1_tree/emb.pt',
    # fn_pairs_val   = '../task48d_prepare_ped-sim_trees_4train_1val/res_s4_small_ped-sim_val_1_trees/pairs.tsv',

    # subset data, only train and validate certain populations and certain kinship values
    # populations = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN','GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI'],
    populations = ['BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'FIN','GBR', 'GIH', 'IBS', 'ITU', 'JPT', 'KHV', 'MXL', 'PEL', 'PJL', 'PUR', 'STU', 'TSI'],
    # populations = ['AMR', 'EAS', 'EUR', 'SAS'],
    # kinship_values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    # n_output_values = 20,               # Number of possible output values: 0-19 kinship
    kinship_values = [0,1,2,3,4,5,6,7,8,9,10,11,12],
    n_output_values = 13,               # Number of possible output values: 0-19 kinship

    # training
    add_cls = False,
    n_cls = 0,                         # Number of cls tokens reserved
    cpu_per_worker = 2,                # Number of CPU cores per worker for data loading
    bs_train = 16,                      # Reduced batch size for training to save memory
    bs_val = 16,                        # Reduced batch size for validation to save memory
    n_id = 3000,                        # Number of contig IDs, 2935 for all contigs in a genome
    mask_fraction = 0.0,               # Fraction of masking

    # model 
    model_py = 'model/spp.py',  # Path to model module
    model_wrapper_py = 'trainer/pl_module.py',  # Path to model wrapper module
    dropout = 0.1,                     # Dropout rate
    seqlen = 2934,
    # attention
    n_layers = 4,                      # Number of transformer layers
    n_heads = 4,                       # Number of attention heads
    d_model = 128,                      # Model dimension
    d_ffn = 128*2,                      # Feed-forward network dimension
    d_emb = 128,                    # Dimension of the embeddings
    d_id = 14,                         # Dimension of ID embeddings, 4 for 2^4=16, 6 for 2^6=64, 8 for 2^8=256, 10 for 2^10=1024, 12 for 2^12=4096, 14 for 2^14=16384, 16 for 2^16=65536, 2^17 for 2^17=131072, 32 for 2^32=4294967296
    # compression
    bin_sizes = [1024, 512, 256, 128, 64], # correspond to 3, 6, 12, 24, 48 contigs per bin
    kernel_size = 8,
    stride = 4,
    cs = 16,
    cd = 6,

    # training 
    train_or_test = 'train',            # Whether to train or test
    max_epochs = 200,                   # Maximum number of training epochs
    strategy = 'ddp',  # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'
    precision = '32',           # Precision for training: 'bf16-true', '16-mixed', '16-true', '32-true'
    # dtype = torch.bfloat16,            # Data type for model parameters
    dtype = torch.float32,            # Data type for model parameters
    lr = 1e-2,                         # Learning rate
    weight_decay = 1e-4,               # Weight decay for optimizer
    accumulate_grad_batches = 1,       # Number of gradient accumulation steps
    max_grad_norm = 2,                 # Maximum gradient norm for clipping. 1.0 is conservative, 0.5 is aggressive, 0.0 is no clipping, 5.0 allows large gradients
    log_grad_norm = True,              # Whether to log gradient norm
    val_check_interval = 0.5,          # Fraction of training epoch after which to run validation
    log_every_n_steps = 10,            # Log metrics every N steps
    nnodes = 1,                        # Number of nodes for distributed training
    devices = 'auto',                  # GPU ids to use ('auto' for all available)
    use_swa = False,                    # Whether to use SWA
    confusion_plot = True,

)

# logging 
config.use_wandb = False
config.use_swanlab = True
config.wandb = {
    'project': 'kinship',                 # W&B project name
    'tags': [config.name],             # Tags for the run
    'group': 'kinship',             # Group name for the run
}

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt'  # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = True                          # Whether to resume from checkpoint

# Create directories if they don't exist
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)