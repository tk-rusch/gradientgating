from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
import torch
import numpy as np

def get_data(name, split=0):
  path = '../data/' +name
  if name in ['chameleon','squirrel']:
    dataset = WikipediaNetwork(root=path, name=name)
  if name in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(path ,name=name)
  if name == 'film':
    dataset = Actor(root=path)

  data = dataset[0]
  if name in ['chameleon', 'squirrel']:
    splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
  if name in ['cornell', 'texas', 'wisconsin']:
    splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  if name == 'film':
    splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')
  if name in ['Cora', 'Citeseer', 'Pubmed']:
      splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data
