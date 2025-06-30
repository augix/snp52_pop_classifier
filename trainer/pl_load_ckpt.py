import torch
import os

def load_ckpt(config, pl_model):
    # 1. torch.load
    if os.path.isdir(config.ckpt_path):  # deepspeed checkpoint
        config.ckpt_path = os.path.join(config.ckpt_path, 'checkpoint/mp_rank_00_model_states.pt')
    checkpoint = torch.load(config.ckpt_path, map_location='cpu')
    
    # 2. load state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'module' in checkpoint: # Common key when saving DataParallel/DDP models
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint

    # 3. remove prefix
    prefix_to_remove = "model."
    cleaned_state_dict = {k.replace(prefix_to_remove, "", 1): v
                            for k, v in state_dict.items()}
    # 4. load weights, reporting issues
    print(f'loading checkpoint from {config.ckpt_path}')
    missing_keys, unexpected_keys = pl_model.model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # update train_step
    if 'train_step' in state_dict:
        pl_model.train_step.copy_(state_dict['train_step'].to(pl_model.device))
        print(f'current train_step: {pl_model.train_step}')

    # 5. freeze layers
    # check if config has attribute freeze_layers
    if hasattr(config, 'freeze_layers'):
        print(f'freezing layers in config.freeze_layers: {config.freeze_layers}')
        frozen = []
        un_frozen = []
        for name, param in pl_model.model.named_parameters():
            name_root = name.split('.')[0]
            if name_root in config.freeze_layers:
                param.requires_grad = False
                frozen.append(name_root)
            else:
                un_frozen.append(name_root)
        print(f'frozen layers: {set(frozen)}')
        print(f'un-frozen layers: {set(un_frozen)}')

    return pl_model
