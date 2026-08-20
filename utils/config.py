
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp, fname):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        env_cfg = yaml.safe_load(stream) or {}
        # Provide a sensible default if 'root_dir' is missing in env config
        root_dir = env_cfg.get('root_dir', os.getcwd())
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set sensible defaults for missing keys to improve robustness
    cfg.setdefault('backbone', 'resnet_ts')
    cfg.setdefault('setup', 'pretext')
    cfg.setdefault('criterion', 'pretext')
    cfg.setdefault('res_kwargs', {})
    # Ensure resnet kwargs have required fields
    if not cfg['res_kwargs']:
        cfg['res_kwargs'] = {'in_channels': 1, 'mid_channels': 4}
    else:
        cfg['res_kwargs'].setdefault('in_channels', 1)
        cfg['res_kwargs'].setdefault('mid_channels', 4)
    cfg.setdefault('num_heads', 1)
    cfg.setdefault('optimizer', 'adam')
    cfg.setdefault('optimizer_kwargs', {'lr': 0.001, 'weight_decay': 0.0})
    cfg.setdefault('scheduler', 'constant')
    cfg.setdefault('scheduler_kwargs', {'lr_decay_rate': 0.1, 'lr_decay_epochs': [60, 80]})
    cfg.setdefault('anomaly_kwargs', {'portion': 0.0})
    cfg.setdefault('augmentation_strategy', 'ts')
    cfg.setdefault('augmentation_kwargs', {'random_resized_crop': {'size': 224}, 'normalize': {'mean': [0.5], 'std': [0.5]}})
    cfg.setdefault('transformation_kwargs', {'noise_sigma': 0.01, 'crop_size': 224})
    cfg.setdefault('batch_size', cfg.get('batch_size', 128))
    cfg.setdefault('num_workers', cfg.get('num_workers', 4))
    cfg.setdefault('epochs', cfg.get('epochs', 100))
    cfg.setdefault('val_db_name', cfg.get('train_db_name'))

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, fname+'/pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['fname'] = fname
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['bottomk_neighbors_train_path'] = os.path.join(pretext_dir, 'bottomk-train-neighbors.npy')
    cfg['aug_train_dataset'] = os.path.join(pretext_dir, 'aug_train_dataset.pth')
    cfg['pretext_features_train_path'] = os.path.join(pretext_dir, 'pretext_features_train.npy')
    cfg['pretext_features_test_path'] = os.path.join(pretext_dir, 'pretext_features_test.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-test-neighbors.npy')
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['bottomk_neighbors_val_path'] = os.path.join(pretext_dir, 'bottomk-test-neighbors.npy')
    cfg['contrastive_dataset'] = os.path.join(pretext_dir, 'con_train_dataset.pth')


    if cfg.get('setup') == 'classification':
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        classification_dir = os.path.join(base_dir, fname+ '/classification')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(classification_dir)
        cfg['classification_dir'] = classification_dir
        cfg['classification_checkpoint'] = os.path.join(classification_dir, 'checkpoint.pth.tar')
        cfg['classification_model'] = os.path.join(classification_dir, 'model.pth.tar')
        cfg['classification_trainfeatures'] = os.path.join(classification_dir, 'classification_traintfeatures.csv')
        cfg['classification_trainprobs'] = os.path.join(classification_dir, 'classification_trainprobs.csv')
        cfg['classification_testfeatures'] = os.path.join(classification_dir, 'classification_testtfeatures.csv')
        cfg['classification_testprobs'] = os.path.join(classification_dir, 'classification_testprobs.csv')

    return cfg 
