import torch
import numpy as np
from numpy import genfromtxt
import networkx
from torch_geometric import utils
import json

def get_data(name='chameleon',split=0):
  edges = genfromtxt('data/'+name+'_edges.csv', delimiter=',')[1:].astype(int)
  G = networkx.Graph()
  for edge in edges:
    G.add_edge(edge[0],edge[1])

  data = utils.from_networkx(G)
  y = genfromtxt('data/'+name+'_target.csv', delimiter=',')[1:,-1].astype(int)
  y = y/np.max(y)
  data.y = torch.tensor(y).float()

  with open('data/'+name+'_features.json', 'r') as myfile:
      file=myfile.read()
  obj = json.loads(file)

  if name == 'chameleon':
    x = np.zeros((2277,3132))
    for i in range(2277):
      feats = np.array(obj[str(i)])
      x[i,feats] = 1

  elif name == 'squirrel':
    x = np.zeros((5201, 3148))
    for i in range(5201):
      feats = np.array(obj[str(i)])
      x[i, feats] = 1

  data.x = torch.tensor(x).float()

  path = '../data/' + name
  splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')

  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data