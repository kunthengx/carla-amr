"""Evaluate AMR test data using a cluster mapping fitted on validation data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix

from utils.amr_dataset import AMR3PhaseDataset, load_amr_data
from utils.common_config import get_model
from utils.config import create_config
from utils.mypath import MyPath


def fit_cluster_mapping(labels, clusters, num_classes):
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (clusters, labels), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.empty(num_classes, dtype=np.int64)
    mapping[rows] = cols
    return mapping, counts


@torch.no_grad()
def predict(model, dataset, head, batch_size):
    if not len(dataset):
        raise ValueError('No windows in evaluation split. Check window_size.')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([model(batch['ts_org'].float())[head].softmax(dim=1)
                      for batch in loader]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_env', default='configs/env_config.yaml')
    parser.add_argument('--config_exp', default='configs/amr_config.yaml')
    parser.add_argument('--fname')
    parser.add_argument('--model', help='Optional classification model path')
    parser.add_argument('--output_dir', help='Defaults to classification/evaluation_test')
    args = parser.parse_args()
    torch.set_num_threads(2)
    p = create_config(args.config_env, args.config_exp, args.fname, setup='classification')
    data = load_amr_data(MyPath.resolve_dataset_file('amr', p['fname']),
                         window_size=p['window_size'])
    mean = data['train_data'].mean(axis=0)
    std = data['train_data'].std(axis=0) + 1e-8

    def dataset(split):
        return AMR3PhaseDataset(
            None, data[split + '_data'], data[split + '_labels'],
            data[split + '_locations'], window_size=p['window_size'],
            is_train=False, mean=mean, std=std,
        )

    val, test = dataset('val'), dataset('test')
    checkpoint = torch.load(args.model or p['classification_model'],
                            map_location='cpu', weights_only=True)
    model = get_model(p)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    head = checkpoint.get('head')
    head = 0 if head is None else int(head)
    val_probs = predict(model, val, head, p['batch_size'])
    mapping, counts = fit_cluster_mapping(val.window_labels, val_probs.argmax(axis=1),
                                          p['num_classes'])
    test_probs = predict(model, test, head, p['batch_size'])
    clusters = test_probs.argmax(axis=1)
    predicted = mapping[clusters]
    labels = list(range(p['num_classes']))
    names = ['Normal', 'Defect', 'Theft']
    report = classification_report(test.window_labels, predicted, labels=labels,
                                   target_names=names, zero_division=0, output_dict=True)
    matrix = confusion_matrix(test.window_labels, predicted, labels=labels)
    output = Path(args.output_dir or Path(p['classification_dir']) / 'evaluation_test')
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'split': 'test', 'mapping_fitted_on': 'val', 'head': head,
        'window_size': p['window_size'], 'num_test_windows': len(test),
        'cluster_to_class': {str(i): int(v) for i, v in enumerate(mapping)},
        'validation_cluster_class_counts': counts.tolist(),
        'report': report,
        'anomalies_predicted_normal': int(matrix[1:, 0].sum()),
    }
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    pd.DataFrame(matrix, index=names, columns=names).rename_axis('actual').to_csv(
        output / 'confusion_matrix.csv')
    rows = pd.DataFrame({'location': test.window_locations, 'label': test.window_labels,
                         'cluster': clusters, 'prediction': predicted})
    for cluster, label in enumerate(mapping):
        rows['prob_' + names[label].lower()] = test_probs[:, cluster]
    rows.to_csv(output / 'predictions.csv', index=False)
    print(classification_report(test.window_labels, predicted, labels=labels,
                                target_names=names, zero_division=0))
    print('Confusion matrix (rows=actual, columns=predicted):\n', matrix)
    print('Cluster mapping fitted on validation:', summary['cluster_to_class'])
    print('Anomalies predicted as Normal:', summary['anomalies_predicted_normal'])
    print('Results:', output)


if __name__ == '__main__':
    main()
