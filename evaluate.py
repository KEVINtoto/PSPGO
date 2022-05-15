

import argparse
from utils import get_mlb, get_pred_matrix, get_pid_and_label_list, get_go_ic
from logzero import logger
from metrics import compute_metrics
import yaml
import os


def main(args):
    # load data config 
    with open('./config/data.yaml', 'r', encoding='UTF-8') as f:
        data_config = yaml.load(f, yaml.FullLoader)
    data_dir = data_config['data_dir']
    result_dir = data_config['result_dir']
    ont = args.ontology

    logger.info(F'Evaluate {args.model} model.')
    go_ic, go_list = get_go_ic(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['goic']))
    mlb = get_mlb(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['gomlb']), go_list)
    pid_list, label_list = get_pid_and_label_list(os.path.join(data_dir, data_config[ont]['dir'], data_config[ont]['test']['goa']))
    pid2index = {pid: i for i, pid in enumerate(pid_list)}
    label_matrix = mlb.transform(label_list)
    label_classes = mlb.classes_
    go2index = {go_id: i for i, go_id in enumerate(label_classes)}
    pred_matrix = get_pred_matrix(os.path.join(result_dir, F'{ont}_test_{args.model}_prediction.txt'), pid2index, go2index)

    (fmax_, smin_, threshold), aupr_ = compute_metrics(label_matrix, pred_matrix, go_ic, label_classes)
    logger.info("Aspect: {} | Fmax: {:.4f} | Smin: {:.4f} | AUPR: {:.4f} | threshold: {:.2f}\n".format(args.ontology, fmax_, smin_, aupr_, threshold))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='PSPGO')
    parser.add_argument("--ontology", type=str, default='mf')

    args = parser.parse_args()
    logger.info(args)
    main(args)