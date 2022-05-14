
from scipy.sparse import csr_matrix
import warnings
import math
import numpy as np
from sklearn.metrics import auc


def compute_metrics(targets: csr_matrix, scores: np.ndarray, go_ic, label_classes):
    """
    return fmax, smin, threshold and aupr
    """
    metrics = 0.0, 0.0, 0.0
    precisions = list()
    recalls = list()
    for t in (threshold / 100 for threshold in range(101)):
        t_scores = csr_matrix((scores >= t).astype(np.int32))
        fmax_, p, r = fmax(targets, t_scores)
        smin_ = smin(targets, t_scores, go_ic, label_classes)
        precisions.append(p)
        recalls.append(r)
        metrics = max(metrics, (fmax_, smin_, t))
    return metrics, aupr(np.asarray(precisions), np.asarray(recalls))

def fmax(targets: csr_matrix, scores: csr_matrix):
    fmax_ = 0.0
    tp = scores.multiply(targets).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p, r = tp / scores.sum(axis=1), tp / targets.sum(axis=1)
        p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
    if np.isnan(p):
        return fmax_, 0.0, r
    try:
        fmax_ = 2 * p * r / (p + r) if p + r > 0.0 else 0.0
    except ZeroDivisionError:
        pass
    return fmax_, p, r

def smin(targets: csr_matrix, scores: csr_matrix, go_ic, annot_classes):
    ru = 0.0
    mi = 0.0
    n = targets.shape[0]
    tp = scores.multiply(targets)
    fn = targets - tp
    fp = scores - tp
    fn_ic_class = fn.sum(axis=0).tolist()[0]
    fp_ic_class = fp.sum(axis=0).tolist()[0]
    for i, j in enumerate(fn_ic_class):
        if j > 0:
            ru += j * go_ic[annot_classes[i]]
    for i, j in enumerate(fp_ic_class):
        if j > 0:
            mi += j * go_ic[annot_classes[i]]
    ru /= n
    mi /= n
    smin_ = math.sqrt(ru * ru + mi * mi)
    return smin_

def aupr(precisions: np.ndarray, recalls: np.ndarray):
    desc_score_indices = np.argsort(recalls, kind="mergesort")[::-1]
    recalls = recalls[desc_score_indices]
    precisions = precisions[desc_score_indices]
    aupr_ = auc(recalls, precisions)
    return aupr_