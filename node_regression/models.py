import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='GCN', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean'))

        return gg

class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN', p=2., drop_in=0, drop=0, use_gg_conv=True):
        super(G2_GNN, self).__init__()
        self.conv_type = conv_type
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers
        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
            if use_gg_conv == True:
                self.conv_gg = GCNConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid,nhid,heads=4,concat=True)
            if use_gg_conv == True:
                self.conv_gg = GATConv(nhid,nhid,heads=4,concat=True)
        else:
            print('specified graph conv not implemented')

        if use_gg_conv == True:
            self.G2 = G2(self.conv_gg,p,conv_type,activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv,p,conv_type,activation=nn.ReLU())

    def forward(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_
        X = F.dropout(X, self.drop, training=self.training)

        return torch.relu(self.dec(X))


class plain_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN', drop_in=0, drop=0):
        super(plain_GNN, self).__init__()
        self.conv_type = conv_type
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers
        if conv_type == 'plain_GCN':
            self.conv = GCNConv(nhid, nhid)
            self.enc = GCNConv(nfeat, nhid)
            self.dec = GCNConv(nhid, nclass)
        elif conv_type == 'plain_GAT':
            self.conv = GATConv(nhid,nhid,heads=4,concat=True)
            self.enc = GATConv(nfeat, nhid,heads=4,concat=True)
            self.dec = GATConv(nhid, nclass,heads=4,concat=True)
        else:
            print('specified graph conv not implemented')

    def forward(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)

        if self.conv_type == 'plain_GAT':
            X = F.elu(self.enc(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = torch.relu(self.enc(X,edge_index))

        for i in range(self.nlayers):
            if self.conv_type == 'plain_GAT':
                X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X = torch.relu(self.conv(X, edge_index))
        X = F.dropout(X, self.drop, training=self.training)

        if self.conv_type == 'plain_GAT':
            X = torch.relu(self.dec(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = torch.relu(self.dec(X,edge_index))

        return X