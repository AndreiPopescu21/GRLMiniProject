from model import GNN
from datasets import get_dataset_by_name
from train import train_graph_classification, train_node_classification
import numpy as np
from collections import defaultdict
import json 
import torch

# Experiment parameters
cfg = {
    'cora': {
        'in_dim': 1433,
        'hidden_dim': 32,
        'out_dim': 7,
        'lr': 0.01,
        'weight_decay': 1e-2,
        'epochs': 250
    },
    'citeseer': {
        'in_dim': 3703,
        'hidden_dim': 64,
        'out_dim': 6,
        'lr': 0.01,
        'weight_decay': 1e-2,
        'epochs': 250
        },
    'PROTEINS': {
        'in_dim': 3,
        'hidden_dim': 128,
        'out_dim': 2,
        'lr': 0.001,
        'weight_decay': 1e-3,
        'epochs': 250
    },
    'MUTAG': {
        'in_dim': 7,
        'hidden_dim': 128,
        'out_dim': 2,
        'lr': 0.001,
        'weight_decay': 1e-3,
        'epochs': 250
    }
}

# Dictionaries to store experiment results
result = defaultdict(list)
benchmarks = {}
# Loop through the experiments
for dataset_name in cfg:
    # Get the dataset by the name from the config
    dataset = get_dataset_by_name(dataset_name)

    # Get parameters
    in_dim = cfg[dataset_name]['in_dim']
    hidden_dim = cfg[dataset_name]['hidden_dim']
    out_dim = cfg[dataset_name]['out_dim']
    lr = cfg[dataset_name]['lr']
    weight_decay = cfg[dataset_name]['weight_decay']
    epochs = cfg[dataset_name]['epochs']

    # Loop through the possible positional encodings
    for encoding in ['laplacian', 'adjacency', 'none']:
        #  Repeat the experiment for 5 times
        for i in range(5):
            # Initialise the model and train it
            if dataset_name in ['cora', 'citeseer']:
                model = GNN(in_dim, hidden_dim, out_dim, False)
                g = dataset[1]
                acc = train_node_classification(g, model, lr, weight_decay, epochs)
            else:
                model = GNN(in_dim, hidden_dim, out_dim, True)
                acc = train_graph_classification(dataset, model, lr, weight_decay, epochs)
            if type(acc) is torch.Tensor:
                acc = acc.item()
            result[dataset_name + '_' + encoding].append(acc)
        #  Store the results
        benchmarks[dataset_name + '_' + encoding + '_mean'] = np.array(result[dataset_name + '_' + encoding]).mean()
        benchmarks[dataset_name + '_' + encoding + '_std'] = np.array(result[dataset_name + '_' + encoding]).std()

# Save the results in a file
with open('result.json', 'w') as file:
    json.dump(result, file)
with open('benchmarks.json', 'w') as file:
    json.dump(benchmarks, file)
