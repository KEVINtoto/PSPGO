""" train PSPGO model. """


import argparse
from logzero import logger
import scipy.sparse as ssp
import numpy as np
import time
import torch as th
import torch.nn as nn
import dgl
import dgl.data.utils
import os
from utils import *
from metrics import compute_metrics
from pspgo import PSPGO
import yaml


def train(args, dataset):
    device = th.device('cuda:' + str(args.gpu))
    # dataset
    g, feature_matrix, label_matrix, train_idx = dataset['g'], dataset['feature'], dataset['label'], dataset['train_idx']
    valid_idx, valid_y = dataset['valid_idx'], dataset['valid_y'] 
    go_ic, label_classes = dataset['goic'], dataset['label_classes']

    model = PSPGO(feature_matrix.shape[1], args.n_hidden, label_matrix.shape[1],
                args.n_mlp_layers, args.n_prop_steps, mlp_drop=args.mlp_dropout,
                attn_heads=args.attn_heads, feat_drop=args.feat_dropout, attn_drop=args.attn_dropout,
                residual=args.residual, share_weight=args.share_weight).to(device)
    # loss function
    ce_loss_fn = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)
    # training dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_prop_steps)
    train_dataloader = dgl.dataloading.NodeDataLoader(g, train_idx, sampler, device='cpu', batch_size=args.batch_size, 
                                                      shuffle=True, num_workers=2,  drop_last=False)
    # train
    best_fmax = 0.0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        for i, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            batch_tic = time.time()
            blocks = [blk.to(device) for blk in blocks]
            batch_x = ( th.from_numpy(feature_matrix[input_nodes.numpy()].indices).long().to(device), 
                        th.from_numpy(feature_matrix[input_nodes.numpy()].indptr).long().to(device), 
                        th.from_numpy(feature_matrix[input_nodes.numpy()].data).float().to(device) )  
            batch_y = th.from_numpy(label_matrix[seeds.numpy()]).float().to(device)
            logits, _ = model(blocks, batch_x)
            loss = ce_loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info("Epoch {:04d} | Batch {:04d} | Train Loss: {:.4f} | Time: {:.4f}".
                        format(epoch, i, loss.item(), time.time() - batch_tic))
        # eval
        model.eval()
        unique_idx = np.unique(valid_idx)
        index_mapping = {idx: i for i, idx in enumerate(unique_idx)}
        res_idx = np.asarray([ index_mapping[idx] for idx in valid_idx ])
        valid_dataloader = dgl.dataloading.NodeDataLoader(g, unique_idx, sampler, device='cpu',
                                     batch_size=args.batch_size, shuffle=False, num_workers=0,  drop_last=False)
        pred_list = []
        for input_nodes, _, blocks in valid_dataloader:
            blocks = [blk.to(device) for blk in blocks]
            batch_x = ( th.from_numpy(feature_matrix[input_nodes.numpy()].indices).long().to(device), 
                        th.from_numpy(feature_matrix[input_nodes.numpy()].indptr).long().to(device), 
                        th.from_numpy(feature_matrix[input_nodes.numpy()].data).float().to(device) )  
            batch_pred, _ = model(blocks, batch_x)
            pred_list.append(th.sigmoid(batch_pred).cpu().detach().numpy())
        valid_pred = np.vstack(pred_list)[res_idx]
        (fmax_, smin_, threshold), aupr_ = compute_metrics(valid_y, valid_pred, go_ic, label_classes)
        logger.info("Epoch {:04d} | Valid X Fmax: {:.4f} | Smin: {:.4f} | threshold: {:.2f} | AUPR: {:.4f}  Time: {:.4f} ".
                    format(epoch, fmax_, smin_, threshold, aupr_, time.time() - t0))
        if fmax_ > best_fmax:
            logger.info(F'improved from {best_fmax} to {fmax_}, save model.')
            best_fmax = fmax_
            th.save(model.state_dict(), os.path.join(args.model_dir, F"{args.ontology}_{args.model}_{args.model_id}.ckp"))


def main(args):
    # load config 
    with open('./config/data.yaml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, yaml.FullLoader)
    data_dir = config['data_dir']
    ont = args.ontology

    logger.info("Start loading data.")

    uniprot2string = get_uniprot2string(os.path.join(data_dir, config['uniprot']['id_map']))
    pid2index = get_pid2index(os.path.join(data_dir, config['network']['index_map']))

    dgl_path = os.path.join(data_dir, config['network']['dgl_path'])
    logger.info(F"Loading dgl graph: {dgl_path}......")
    g = dgl.load_graphs(dgl_path)[0][0]
    logger.info(F"The info of the heterogeneous network: {g}")

    logger.info("Loading annotation data......")
    train_pid_list, train_label_list = get_pid_and_label_list(os.path.join(data_dir, config[ont]['dir'], config[ont]['train']['goa']))
    valid_pid_list, valid_label_list = get_pid_and_label_list(os.path.join(data_dir, config[ont]['dir'], config[ont]['valid']['goa']))
    logger.info(F"Number of train pid: {len(train_pid_list)}, valid pid: {len(valid_pid_list)}.")

    logger.info("Get Diamond result of valid set.")
    make_diamond_db(os.path.join(data_dir, config['network']['fasta']), os.path.join(data_dir, config['network']['diamond_db']))
    valid_diamond_result = diamond_homo(os.path.join(data_dir, config['network']['diamond_db']),
                                        os.path.join(data_dir, config[ont]['dir'], config[ont]['valid']['fasta']),
                                        os.path.join(data_dir, config[ont]['dir'], config[ont]['valid']['diamond']))

    logger.info("Mapping pid to network index.")
    train_idx, train_pid_list, train_label_list = get_network_index(uniprot2string, pid2index, train_pid_list, train_label_list)
    valid_idx, valid_pid_list, valid_label_list = get_network_index(uniprot2string, pid2index, valid_pid_list, valid_label_list,
                                                                    valid_diamond_result, category='eval')
    logger.info(F"Number of train index: {len(train_idx)}, valid index: {len(valid_idx)}.")
    
    logger.info('Get feature matrix.')
    feature_matrix = ssp.load_npz(os.path.join(data_dir, config['network']['interpro']))
    logger.info(F'Shape of feature matrix: {feature_matrix.shape}')

    logger.info('Get label matrix.')
    go_ic, go_list = get_go_ic(os.path.join(data_dir, config[ont]['dir'], config[ont]['goic']))
    go_mlb = get_mlb(os.path.join(data_dir, config[ont]['dir'], config[ont]['gomlb']), go_list)
    label_classes = go_mlb.classes_
    train_y = go_mlb.transform(train_label_list).astype(np.float32)
    valid_y = go_mlb.transform(valid_label_list).astype(np.float32)
    label_matrix = np.zeros((feature_matrix.shape[0], label_classes.shape[0]))
    label_matrix[train_idx] = train_y.toarray()
    flag = np.full(feature_matrix.shape[0], False, dtype=bool)
    flag[train_idx] = np.full(train_idx.shape[0], True, dtype=bool)
    g.ndata['flag'] = th.from_numpy(flag)

    dataset = dict()
    dataset['g'], dataset['feature'], dataset['label'], dataset['train_idx'] = g, feature_matrix, label_matrix, train_idx
    dataset['valid_idx'], dataset['valid_y'] = valid_idx, valid_y
    dataset['goic'], dataset['label_classes'] = go_ic, label_classes

    logger.info("Data loading is complete.")
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    logger.info("Start training...")
    train(args, dataset)
    logger.info('Finished train.\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train PSPGO model.')
    parser.add_argument("--model", type=str, default="PSPGO",
                        help="model name.")
    parser.add_argument("--ontology", type=str, default="mf")

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--use_ddp", default=False, action='store_true')
    parser.add_argument('--n_gpus', type=int, default=2,
                        help='number of gpu for distributed training, only available if use_ddp is true')

    parser.add_argument("--n_mlp_layers", type=int, default=1)
    parser.add_argument("--n_prop_steps", type=int, default=2)
    parser.add_argument("-e", "--n_epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=40,
                        help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--mlp_dropout", type=float, default=0.5,
                        help="mlp dropout probability")
    parser.add_argument("--feat_dropout", type=float, default=0.0,
                        help="feature dropout probability")
    parser.add_argument("--attn_dropout", type=float, default=0.0,
                        help="attention dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n_hidden", type=int, default=512,
                        help="hidden size")
    parser.add_argument("--residual", type=bool, default=True,
                        help="whether to make residual connection")
    parser.add_argument("--attn_heads", type=int, default=1,
                        help="the number of attention heads")
    parser.add_argument("--share_weight", type=bool, default=True,
                        help="whether parameters are shared between different types of networks")
    
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default='models',
                        help='path for save the model parameters')
    
    args = parser.parse_args()

    logger.info("Running the training script for PSPGO model.")
    logger.info(F"Ontology: {args.ontology}")
    logger.info(F"Hyperparameters: ")
    logger.info(F"\t* Training epoch: {args.n_epochs}")
    logger.info(F"\t* Batch size: {args.batch_size}")
    logger.info(F"\t* Learning rate: {args.lr}")
    logger.info(F"\t* MLP layers: {args.n_mlp_layers}")
    logger.info(F"\t* Propagation layers: {args.n_prop_steps}")
    logger.info(F"\t* Attention heads: {args.attn_heads}")
    logger.info(F"\t* MLP dropout: {args.mlp_dropout}")
    logger.info(F"\t* Feature dropout: {args.feat_dropout}")
    logger.info(F"\t* Attention dropout: {args.attn_dropout}")
    main(args)


############################################################################### end ################################################################################