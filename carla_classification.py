
import argparse
import os
import torch
import pandas
import numpy as np
from utils.mypath import MyPath
from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_val_transformations1, \
                                get_train_dataset, get_train_dataloader, get_aug_train_dataset,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate, inject_sub_anomaly
from utils.evaluate_utils import get_predictions, classification_evaluate, pr_evaluate
from utils.train_utils import self_sup_classification_train
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

FLAGS = argparse.ArgumentParser(description='classification Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--fname', help='Config the file name of Dataset')

def main():
    global best_f1
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp, args.fname)
    print(colored('CARLA Self-supervised Classification stage --> ', 'yellow'))

    # CUDNN
   # torch.backends.cudnn.benchmark = True

    # Data
    print(colored('\n- Get dataset and dataloaders for ' + p['train_db_name'] + ' dataset - timeseries ' + p['fname'], 'green'))
    train_transformations = get_train_transformations(p)
    sanomaly = inject_sub_anomaly(p)
    val_transformations = get_val_transformations1(p)
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)

    if p['train_db_name'] == 'amr':
        from utils.amr_dataset import load_amr_data
    
    amr_file = os.path.join(MyPath.db_root_dir('amr'), p['fname'])
    data_dict = load_amr_data(amr_file, window_size=p.get('window_size', 200))
    
    base_dataset = get_train_dataset(
        p, train_transformations, sanomaly,
        to_augmented_dataset=True,
        data=data_dict['train_data'],
        label=data_dict['train_labels']
    )
    
    val_dataset = get_val_dataset(
        p, val_transformations, sanomaly, False,
        base_dataset.mean, base_dataset.std,
        data_dict['val_data'], data_dict['val_labels']
    )

    val_dataloader = get_val_dataloader(p, val_dataset)

    print(colored('-- Train samples size: %d - Test samples size: %d' %(len(train_dataset), len(val_dataset)), 'green'))

    # Model
    model = get_model(p, p['pretext_model'])
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: classification will only update the cluster head', 'red'))

    # Loss function
    criterion = get_criterion(p)
    criterion.to(device)

    print(colored('\n- Model initialisation', 'green'))
    # Checkpoint
    if os.path.exists(p['classification_checkpoint']):
        print(colored('-- Model initialised from last checkpoint: {}'.format(p['classification_checkpoint']), 'green'))
        checkpoint = torch.load(p['classification_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        normal_label = checkpoint['normal_label']

    else:
        print(colored('-- No checkpoint file at {} -- new model initialised'.format(p['classification_checkpoint']), 'green'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        normal_label = 0


    best_f1 = -1 * np.inf
    # best_loss = np.inf
    print(colored('\n- Training:', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('-- Epoch %d/%d' %(epoch+1, p['epochs']), 'blue'))

        lr = adjust_learning_rate(p, optimizer, epoch)
        self_sup_classification_train(train_dataloader, model, criterion, optimizer, epoch,
                                      p['update_cluster_head_only'])

        if (epoch == p['epochs']-1):
            tst_dl = get_val_dataloader(p, train_dataset)
            predictions, _ = get_predictions(p, tst_dl, model, True, True)
        else:
            tst_dl = get_val_dataloader(p, train_dataset)
            predictions = get_predictions(p, tst_dl, model, False, False)

        label_counts = torch.bincount(predictions[0]['predictions'])
        majority_label = label_counts.argmax()

        classification_stats = classification_evaluate(predictions)
        lowest_loss_head = classification_stats['lowest_loss_head']
        lowest_loss = classification_stats['lowest_loss']
        predictions = get_predictions(p, val_dataloader, model, False, False)

        rep_f1 = pr_evaluate(predictions, compute_confusion_matrix=False, majority_label=majority_label)

        if rep_f1 > best_f1:
            best_f1 = rep_f1
            nomral_label = majority_label
            # print('New Checkpoint ...')
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head, 'normal_label': normal_label}, p['classification_model'])
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': normal_label},
                       p['classification_checkpoint'])


    model_checkpoint = torch.load(p['classification_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': p['epochs'], 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'normal_label': normal_label},
               p['classification_checkpoint'])
    normal_label = model_checkpoint['normal_label']
    tst_dl = get_val_dataloader(p, val_dataset)
    predictions, _ = get_predictions(p, tst_dl, model, True)

if __name__ == "__main__":
    main()
