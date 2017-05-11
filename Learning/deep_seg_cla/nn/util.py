import sys
import numpy as np
import tensorflow as tf

from generic import print_scores

MAX_INT = np.iinfo(np.int32).max


def _normalize(x, m=None, s=None):
    m = np.mean(x, axis=(1, 2), keepdims=True) if m is None else m
    s = np.std(x, axis=(1, 2), keepdims=True) if s is None else s
    x = x - m
    return x / s


def _op_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/op', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def exp_average(curr, prev, ratio):
    if prev is None:
        return curr
    else:
        return np.average([curr, prev], weights=[ratio, 1 - ratio])


def randint(state):
    return state.randint(MAX_INT)


def progress_bar(curr, total, length=20):
    completion = curr / total
    n_complete = int(completion * length)
    n_incomplete = length - n_complete
    return "|{}{}| {:3.2f}%".format(n_complete * "#", n_incomplete * " ", completion * 100)


def print_tf_scores(y_true, probas, labels=None, out=sys.stdout):
    """
    y_true:  [n_samples, n_classes] (if [n_samples] call print_scores directly)
    probas:  [n_samples, n_classes]
    labels:  [n_classes]
    """
    probas_true = y_true
    y_pred = np.argmax(probas, axis=1)
    y_true = np.argmax(probas_true, axis=1)
    print_scores(y_true, y_pred, probas[:, 1], labels=labels, file=out)
