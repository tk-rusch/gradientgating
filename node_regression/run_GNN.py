from models import *
import torch
import torch.optim as optim
import numpy as np
from data_handling import get_data
import argparse

def train(args, split):
    data = get_data(args.dataset,split)
    best_eval_loss = 1e5
    bad_counter = 0
    best_test_loss = 1e5
    patience = 200

    nout = 1

    if args.dataset == 'chameleon':
        ninp = 3132
    elif args.dataset == 'squirrel':
        ninp = 3148

    if 'plain' in args.GNN:
        model = plain_GNN(ninp, args.nhid, nout, args.nlayers, args.GNN, args.drop_in, args.drop,).to(args.device)
    else:
        model = G2_GNN(ninp, args.nhid, nout, args.nlayers, args.GNN, args.G2_exp, args.drop_in, args.drop,
                       args.use_G2_conv).to(args.device)

    lf = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    @torch.no_grad()
    def test(model, data):
        model.eval()
        out, losses = model(data).squeeze(-1), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(out[mask], data.y.squeeze()[mask])/torch.mean(data.y)
            losses.append(loss.item())
        return losses

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(args.device)).squeeze(-1)
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

        [train_loss, val_loss, test_loss] = test(model, data)

        if (val_loss < best_eval_loss):
            best_eval_loss = val_loss
            best_test_loss = test_loss
        else:
            bad_counter += 1

        if ((epoch+1) == patience):
            break

        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(split, epoch, train_loss, val_loss, test_loss))

    return best_test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='squirrel',
                        help='dataset name: squirrel, chameleon')
    parser.add_argument('--GNN', type=str, default='GCN',
                        help='base GNN model used with G^2: GCN, GAT -- '
                             'plain GNN versions: plain_GCN, plain_GAT')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=500,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--drop_in', type=float, default=0.2,
                        help='input dropout rate')
    parser.add_argument('--drop', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight_decay')
    parser.add_argument('--G2_exp', type=float, default=5.,
                        help='exponent p in G^2')
    parser.add_argument('--use_G2_conv', type=bool, default=False,
                        help='use a different GNN model for the gradient gating method')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')

    args = parser.parse_args()

    n_splits = 10
    best_results = []
    for split in range(n_splits):
        best_results.append(train(args, split))

    best_results = np.array(best_results)
    mean_acc = np.mean(best_results)
    std = np.std(best_results)

    log = 'Final test results -- mean: {:.4f}, std: {:.4f}'
    print(log.format(mean_acc, std))
