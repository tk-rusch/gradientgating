from torch_geometric.datasets import Planetoid
import torch
import numpy as np
from torch_geometric.utils import convert
import scipy.sparse as sp
from torch_geometric.data import Data


def get_data(hom_level=0, graph=1):
  seed = 123456
  dataset = Planetoid('../data', 'Cora')

  if(hom_level==0):
    loader = np.load('data/h0.0' + str(round(hom_level, 3)) + '-r' + str(graph) + '.npz')
  else:
    loader = np.load('data/h0.'+str(round(hom_level,3))+'-r'+str(graph)+'.npz')

  adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                       loader['adj_indptr']), shape=loader['adj_shape'])

  features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                            loader['attr_indptr']), shape=loader['attr_shape'])

  y = torch.tensor(loader.get('labels')).long()
  x = torch.tensor(sp.csr_matrix.todense(features)).float()
  edge_index, edge_features = convert.from_scipy_sparse_matrix(adj)

  data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=torch.zeros(y.size()[0], dtype=torch.bool),
    test_mask=torch.zeros(y.size()[0], dtype=torch.bool),
    val_mask=torch.zeros(y.size()[0], dtype=torch.bool)
  )
  dataset.data = data

  num_nodes = data.y.shape[0]

  rnd_state = np.random.RandomState(seed)

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  idx = rnd_state.choice(num_nodes, size=num_nodes, replace=False)
  idx_train = idx[:int(0.5*num_nodes)]
  idx_val = idx[int(0.5*num_nodes):int(0.75*num_nodes)]
  idx_test = idx[int(0.75*num_nodes):]

  dataset.data.train_mask = get_mask(idx_train)
  dataset.data.val_mask = get_mask(idx_val)
  dataset.data.test_mask = get_mask(idx_test)

  return dataset.data
