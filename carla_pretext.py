import argparse
import os
import torch
import numpy as np
import pandas
from utils.mypath import MyPath

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_val_transformations1, get_optimizer,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import contrastive_evaluate
from utils.repository import TSRepository
from utils.train_utils import pretext_train
from utils.utils import fill_ts_repository
from termcolor import colored
from statsmodels.tsa.stattools import adfuller
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parser
parser = argparse.ArgumentParser(description='pretext')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--fname',
                    help='Config the file name of Dataset')
args = parser.parse_args()

def main():
    # # Set PyTorch-specific threading options
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1) 

    print(colored('CARLA Pretext stage --> ', 'yellow'))
    p = create_config(args.config_env, args.config_exp, args.fname)

    model = get_model(p)
    best_model = None
    model = model.to(device)
   
    # CUDNN
    # torch.backends.cudnn.benchmark = True

    train_transforms = get_train_transformations(p)

    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations1(p)

    # Data loading
    print(colored('\n- Get dataset and dataloaders for ' + p['train_db_name'] + ' dataset - timeseries ' + p['fname'], 'green'))
    sanomaly = inject_sub_anomaly(p)

    # Load AMR-specific data when requested
    if p.get('train_db_name', None) == 'amr':
        from utils.amr_dataset import load_amr_data
        amr_file = os.path.join(MyPath.db_root_dir('amr'), p['fname'])
        data_dict = load_amr_data(amr_file, window_size=p.get('window_size', 200))

        train_dataset = get_train_dataset(
            p, train_transforms, sanomaly,
            to_augmented_dataset=True,
            data=data_dict['train_data'],
            label=data_dict['train_labels']
        )

        val_dataset = get_val_dataset(
            p, val_transforms, sanomaly, False,
            train_dataset.mean, train_dataset.std,
            data_dict['val_data'], data_dict['val_labels']
        )

    else:
        # Generic dataset loading for other datasets
        train_dataset = get_train_dataset(p, train_transforms, sanomaly, to_augmented_dataset=True)
        val_dataset = get_val_dataset(p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std)

    train_dataloader = get_train_dataloader(p, train_dataset)

    print("\n=== DATALOADER DEBUG ===")
    print("train_dataset type:", type(train_dataset))
    print("train_dataloader type:", type(train_dataloader))

    sample = train_dataset[0]
    print("dataset[0] type:", type(sample))

    if isinstance(sample, dict):
        print("dataset[0] keys:", sample.keys())
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: type={type(v)}")
    else:
        print("dataset[0] =", sample)

    batch = next(iter(train_dataloader))

    print("batch type:", type(batch))

    if isinstance(batch, dict):
        print("batch keys:", batch.keys())
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"  batch[{k}] shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  batch[{k}] type={type(v)}")
    else:
        print("batch =", batch)

    val_dataloader = get_val_dataloader(p, val_dataset)
    base_dataloader = get_val_dataloader(p, train_dataset)

    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    # TS Repository
   # base_dataset = get_train_dataset(p, train_transforms, panomaly, sanomaly, to_augmented_dataset=True, split='train')

    ts_repository_base = TSRepository(
        len(train_dataset),
        p['model_kwargs']['features_dim'],
        p['num_classes'],
        p['criterion_kwargs']['temperature']
    )
    ts_repository_base.to(device)
    ts_repository_val = TSRepository(len(val_dataset),
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])
    ts_repository_val.to(device)

    criterion = get_criterion(p)
    criterion = criterion.to(device)

    # optimizer = get_optimizer(p, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['optimizer_kwargs']['lr'])
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.to(device)
    
    # Training
    pretext_best_loss = np.inf
    prev_loss = None
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # print('EPOCH ----> ', epoch)
        tmp_loss = pretext_train(train_dataloader, model, criterion, optimizer, epoch, prev_loss, device=device)
        
        # Checkpoint
        if tmp_loss <= pretext_best_loss:
            pretext_best_loss = tmp_loss
            best_model = model

    # Save final model
    torch.save(best_model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the classification loss.
    print(colored('Fill TS Repository for mining the nearest/furthest neighbors (train) ...', 'blue'))
    ts_repository_aug = TSRepository(len(train_dataset) * 2,
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature']) #need size of repository == 1+num_of_anomalies
    fill_ts_repository(p, base_dataloader, model, ts_repository_base, real_aug = True, ts_repository_aug = ts_repository_aug)
    # out_pre = np.column_stack((ts_repository_base.features, ts_repository_base.targets))
    out_pre = np.column_stack((ts_repository_base.features.cpu().numpy(), ts_repository_base.targets.cpu().numpy()))

    np.save(p['pretext_features_train_path'], out_pre)
    topk = 10
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_train_path'], knearest)
    np.save(p['bottomk_neighbors_train_path'], kfurtherst)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill TS Repository for mining the nearest/furthest neighbors (val) ...', 'blue'))

    fill_ts_repository(p, val_dataloader, model, ts_repository_val, real_aug=False, ts_repository_aug=None)
    # out_pre = np.column_stack((ts_repository_val.features, ts_repository_val.targets))
    out_pre = np.column_stack((ts_repository_val.features.cpu().numpy(), ts_repository_val.targets.cpu().numpy()))

    np.save(p['pretext_features_test_path'], out_pre)
    topk = 10
    print('Mine the nearest and furthest neighbors (Top-%d)' %(topk))
    kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)
    np.save(p['topk_neighbors_val_path'], knearest)
    np.save(p['bottomk_neighbors_val_path'], kfurtherst)

 
if __name__ == '__main__':
    main()
