import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

# The train method for the node classification task
def train_node_classification(g, model, lr, wd, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0
    best_test_acc = 0

    # Prepare the dataset
    feats = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    # Train loop
    for epoch in range(epochs):
        logits = model(g, feats)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        best_val_acc = max(best_val_acc, val_acc)
        best_test_acc = max(best_test_acc, test_acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, loss: {loss:.4f}, train acc: {train_acc:.4f} val acc: {val_acc:.4f}, test acc: {test_acc:.4f}"
            )
    return best_test_acc

# The evaluate model function for the graph classification task
def evaluate_graph_classification(dataloader, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph
        labels = labels
        feat = batched_graph.ndata.pop("attr")
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc

# The train method for the graph classification task
def train_graph_classification(dataset, model, lr, wd, epochs):
    # Prepare the train and test dataloader
    train_len = int(0.9 * len(dataset))
    train_idx = [i for i in range(train_len)]
    test_idx = [i for i in range(train_len, len(dataset))]
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetRandomSampler(train_idx),
        drop_last=False,
        shuffle=False)
    test_dataloader = GraphDataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetRandomSampler(test_idx),
        drop_last=False,
        shuffle=False)

    best_test_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_dataloader):
            batched_graph = batched_graph
            labels = labels
            feat = batched_graph.ndata.pop("attr")
            logits = model(batched_graph, feat)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate_graph_classification(train_dataloader, model)
        test_acc = evaluate_graph_classification(test_dataloader, model)
        best_test_acc = max(best_test_acc, test_acc)
        print(
            f"Epoch {epoch} Loss {total_loss/(batch + 1):.4f} Train Acc. {train_acc:.4f} Validation Acc. {best_test_acc:.4f} "
            )
    return best_test_acc