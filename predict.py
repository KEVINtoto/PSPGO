

import argparse
from logzero import logger
import scipy.sparse as ssp
import numpy as np
import torch as th
import dgl
import dgl.data.utils
import os
from utils import *
from pspgo import PSPGO
import yaml


# alphas = {'mf': 0.6, 'cc': 0.9, 'bp': 0.2}


def main(args):
    # load data config 
    with open('./config/data.yaml', 'r', encoding='UTF-8') as f:
        data_config = yaml.load(f, yaml.FullLoader)
    data_dir = data_config['data_dir']
    result_dir = data_config['result_dir']
    ont = args.ontology

    uniprot2string = get_uniprot2string(os.path.join(data_dir, data_config['uniprot']['id_map']))
    pid2index = get_pid2index(os.path.join(data_dir, data_config['network']['index_map']))

    dgl_path = os.path.join(data_dir, data_config['network']['dgl_path'])
    g = dgl.load_graphs(dgl_path)[0][0]

    train_pid_list, train_label_list = get_pid_and_label_list(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['train']['goa']))

    train_idx, train_pid_list, train_label_list = get_network_index(uniprot2string, pid2index, train_pid_list, train_label_list)

    feature_matrix = ssp.load_npz(os.path.join(data_dir, data_config['network']['interpro']))
    
    go_ic, go_list = get_go_ic(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['goic']))
    go_mlb = get_mlb(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['gomlb']), go_list)
    label_classes = go_mlb.classes_
    train_y = go_mlb.transform(train_label_list).astype(np.float32)
    label_matrix = np.zeros((feature_matrix.shape[0], label_classes.shape[0]))
    label_matrix[train_idx] = train_y.toarray()
    flag = np.full(feature_matrix.shape[0], False, dtype=bool)
    flag[train_idx] = np.full(train_idx.shape[0], True, dtype=bool)
    g.ndata['flag'] = th.from_numpy(flag)


    # load model config 
    with open('./config/pspgo.yaml', 'r', encoding='UTF-8') as f:
        model_config = yaml.load(f, yaml.FullLoader)
    model_dir = model_config['model_dir']

    make_diamond_db(os.path.join(data_dir, data_config['network']['fasta']), os.path.join(data_dir, data_config['network']['diamond_db']))
    if args.input_file == None:
        pred_pid_list = get_pid_list(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['test']['fasta']))
        pred_diamond_result  = diamond_homo(os.path.join(data_dir, data_config['network']['diamond_db']),
                                            os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['test']['fasta']),
                                            os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['test']['diamond']))
    else:
        pred_pid_list = get_pid_list(args.input_file)
        pred_diamond_result  = diamond_homo(os.path.join(data_dir, data_config['network']['diamond_db']), args.input_file,
                                            os.path.join(result_dir, args.input_file+'.diamond'))

    pred_index = list()
    tmp_pid_list = list()
    ct = 0
    for pid in pred_pid_list:
        if pid2index.get(pid) != None:
            pred_index.append(pid2index[pid])
            tmp_pid_list.append(pid)
        elif uniprot2string.get(pid) != None and pid2index.get(uniprot2string[pid]) != None:
            pred_index.append(pid2index[uniprot2string[pid]])
            tmp_pid_list.append(pid)
        elif pred_diamond_result.get(pid) != None:
            pred_index.append( pid2index[ max( pred_diamond_result[pid].items(), key=lambda x: x[1] )[0] ] )
            tmp_pid_list.append(pid)
            ct += 1
    logger.info(F"There are {ct} proteins that don't have network index.")

    device = 'cpu'
    if args.gpu >= 0 and th.cuda.is_available():
        device = 'cuda:%d' % args.gpu

    pred_x_score_list, pred_y_score_list = list(), list()
    n_models = model_config['n_models']
    logger.info('Start predict......')
    logger.info('Load trained model.')
    for model_id in np.arange(n_models):
        model = PSPGO(feature_matrix.shape[1], model_config['n_hidden'], label_matrix.shape[1],
                model_config['n_mlp_layers'], model_config['n_prop_steps'], mlp_drop=model_config['mlp_dropout'],
                attn_heads=model_config['attn_heads'], feat_drop=model_config['feat_dropout'], attn_drop=model_config['attn_dropout'],
                residual=model_config['residual'], share_weight=model_config['share_weight']).to(device)
        model.load_state_dict(th.load(os.path.join(model_dir, F'{args.ontology}_{args.model}_{model_id}.ckp'))) 
        test_x_score, test_y_score = model.inference(g, pred_index, feature_matrix, label_matrix, model_config['batch_size'], device)
        pred_x_score_list.append(test_x_score)
        pred_y_score_list.append(test_y_score)

    alpha = model_config['alphas'][ont]
    pred_combine_score = ( alpha * pred_x_score_list[0] + (1 - alpha) * pred_y_score_list[0] ) / n_models
    for i in np.arange(1, n_models):
        pred_combine_score +=  (alpha * pred_x_score_list[i] + (1 - alpha) * pred_y_score_list[i]) / n_models

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if args.input_file == None:
        with open(os.path.join(result_dir, F'{ont}_test_{args.model}_prediction.txt'), 'w') as f:
            for i in range(pred_combine_score.shape[0]):
                for j in range(pred_combine_score.shape[1]):
                    f.write(F'{tmp_pid_list[i]}\t{label_classes[j]}\t{pred_combine_score[i, j]}\n')
    else:
        with open(os.path.join(result_dir, args.input_file+'.prediction'), 'w') as f:
            for i in range(pred_combine_score.shape[0]):
                for j in range(pred_combine_score.shape[1]):
                    f.write(F'{tmp_pid_list[i]}\t{label_classes[j]}\t{pred_combine_score[i, j]}\n')
    logger.info(F'Finished save predicted result.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument("--model", type=str, default="PSPGO")
    parser.add_argument("--ontology", type=str, default="mf")

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    
    parser.add_argument("-f", "--input_file", type=str, default=None,
                        help="The fasta file path of the protein that needs to be predicted.")

    args = parser.parse_args()
    logger.info("Running the PSPGO model for prediction.")
    logger.info(F"Ontology: {args.ontology}")
    logger.info(F"The input FASTA file path: {args.input_file}")
    main(args)

######################################################################### end ################################################################################