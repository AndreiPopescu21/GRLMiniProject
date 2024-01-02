import dgl
import torch
import dgl.data
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

# Calculate the laplacian of the matrix
# and return its eigenvectors
def laplacian_eigenvectors(graph, k):
    adj_matrix = graph.adjacency_matrix().to_dense()
    degrees = adj_matrix.sum(axis=0)
    deg_matrix = torch.diag(degrees)
    laplacian = deg_matrix - adj_matrix
    laplacian_sparse = sp.csr_matrix(laplacian.detach().numpy())
    eigenvalues, eigenvectors = eigs(laplacian_sparse, k=k, which='SM')
    eigenvectors = torch.abs(torch.from_numpy(eigenvectors.real))

    return eigenvalues, eigenvectors

# Make a pairwise difference of the adjacent nodes and put them as edge features
def edge_features_from_eigenvectors(graph, eigenvectors):
    edge_features = torch.zeros((graph.number_of_edges(), eigenvectors.shape[1]))

    for i, (src, dst) in enumerate(zip(*graph.edges())):
        src_eigenvector = eigenvectors[src]
        dst_eigenvector = eigenvectors[dst]
        edge_feature = src_eigenvector - dst_eigenvector
        edge_features[i] = edge_feature

    return torch.abs(edge_features) * 4 + 5

# Get the node classification datasets (Cora/Citeseer)
def get_dataset_citations(name='cora', encoding='laplacian'):
    # Get the dataset
    if name == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    else:
        dataset = dgl.data.CiteseerGraphDataset()
    g = dataset[0]

    # Add positional encodings to the graphs in the dataset
    if encoding == 'laplacian':
        _, eigenvectors = laplacian_eigenvectors(g, 1)
        edge_feats = edge_features_from_eigenvectors(g, eigenvectors)
    elif encoding == 'adjacency':
        adj_matrix = g.adjacency_matrix().to_dense()
        _, eigenvectors = scipy.linalg.eigh(adj_matrix)
        adj_eigenvectors = torch.abs(torch.tensor(eigenvectors[:1])).T
        edge_feats = edge_features_from_eigenvectors(g, adj_eigenvectors)
    elif encoding == 'random':
        edge_feats = torch.rand((g.num_edges(), 1)) * 2 + 1
    else:
        # Add ones as features, so that the model acts as it does
        # not have positional encodings in the message passing process
        edge_feats = torch.ones((g.num_edges(), 1))

    # Add the positional encodings as edge features
    g.edata['e'] = edge_feats
    return dataset, g

# Get the graph classification datasets (PROTEIN/MUTAG)
def get_dataset_graph_classification(name, encoding='laplacian'):
    # Get the dataset
    dataset = dgl.data.GINDataset(name, False)
    for data in dataset:
        g = data[0]
        # Add positional encodings to the graphs in the dataset
        if encoding == 'laplacian':
            _, eigenvectors = laplacian_eigenvectors(g, 1)
            edge_feats = edge_features_from_eigenvectors(g, eigenvectors)
        elif encoding == 'adjacency':
            adj_matrix = g.adjacency_matrix().to_dense()
            _, eigenvectors = scipy.linalg.eigh(adj_matrix)
            adj_eigenvectors = torch.abs(torch.tensor(eigenvectors[:1])).T
            edge_feats = edge_features_from_eigenvectors(g, adj_eigenvectors)
        elif encoding == 'random':
            edge_feats = torch.rand((g.num_edges(), 1)) * 2 + 1
        else:
            # Add ones as features, so that the model acts as it does
            # not have positional encodings in the message passing process
            edge_feats = torch.ones((g.num_edges(), 1))

        # Add the positional encodings as edge features
        g.edata['e'] = edge_feats
        
    return dataset

# Return the dataset, given its name
def get_dataset_by_name(name, encoding='laplacian'):
    if name in ['cora', 'citeseer']:
        return get_dataset_citations(name, encoding)
    return get_dataset_graph_classification(name, encoding)
