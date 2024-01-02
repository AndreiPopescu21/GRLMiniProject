import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl

# The layer of the GNN
class Layer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Layer, self).__init__()
        self.Ws = nn.Linear(in_feat, out_feat)
        self.Wn = nn.Linear(in_feat, out_feat)
        self.Wu = nn.Linear(in_feat, out_feat)
        self.linear = nn.Linear(out_feat, out_feat)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Perform message passing using the edge information
            g.update_all(
                message_func=fn.u_mul_e("h", "e", "m"),
                reduce_func=fn.mean("m", "h_p"),
            )
            # Perform message passing without edge information
            g.update_all(
                message_func=fn.copy_u("h", "m"),
                reduce_func=fn.mean("m", "h_u"),
            )
            h_p = g.ndata["h_p"]
            h_u = g.ndata["h_u"]
            # Combine the current features with the ones resulted from
            # the aggregation
            h_total = self.Ws(h) + self.Wn(h_p) + self.Wu(h_u)
            return self.linear(h_total)
        

# GNN class
class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph_classification):
        super(GNN, self).__init__()
        self.conv1 = Layer(in_feats, h_feats)
        self.conv2 = Layer(h_feats, num_classes)
        self.graph_classification = graph_classification

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        # Use sum aggregation to combine features
        if self.graph_classification:
            h = dgl.sum_nodes(g, 'h')
        return h
    
    