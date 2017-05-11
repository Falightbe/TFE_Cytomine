import datetime
import os
import re
import sys

import numpy as np
from scipy.ndimage import imread
from sklearn.metrics import f1_score

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def custom_iso(clean=''):
    return re.sub('[:-]', clean, datetime.datetime.now().isoformat())


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, out=sys.stdout):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    out.write("    " + empty_cell)
    for label in labels:
        out.write(" %{0}s".format(columnwidth) % label)
    out.write(os.linesep)
    # Print rows
    for i, label1 in enumerate(labels):
        out.write("    %{0}s ".format(columnwidth) % label1)
        for j in range(len(labels)):
            cell = "%{0}.1f ".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            out.write(cell)
        out.write(os.linesep)


def load_images(files, shape):
    samples, (rows, cols, chans) = files.shape[0], shape
    X = np.zeros((samples, rows, cols, chans), dtype=np.uint8)
    for i, file in enumerate(files):
        sys.stdout.write("Fetching the images ({}%)...\r".format(round(100 * i / float(samples), ndigits=2)))
        X[i] = imread(file)
    print("\nImages: {}".format(X.shape))
    return X


def print_scores(y_test, y_pred, probas, labels, file=None):
    binary = len(np.unique(y_test)) == 2
    cc = accuracy_score(y_test, y_pred)
    ra = recall_score(y_test, y_pred) if binary else None
    pr = precision_score(y_test, y_pred) if binary else None
    f1 = f1_score(y_test, y_pred) if binary else None
    cm = confusion_matrix(y_test, y_pred)

    if probas is not None:
        roc = roc_auc_score(y_test, probas) if binary else None

    if file is not None:
        file.write("Accuracy : {}".format(cc) + os.linesep)
        if ra is not None: file.write("Recall   : {}".format(ra) + os.linesep)
        if pr is not None: file.write("Precision: {}".format(pr) + os.linesep)
        if f1 is not None: file.write("f1-score : {}".format(f1) + os.linesep)
        if probas is not None and roc is not None:
            file.write("Roc auc  : {}".format(roc) + os.linesep)
        print_cm(cm, labels, out=file)
    else:
        print("Accuracy : {}".format(cc))
        print("Recall   : {}".format(ra))
        print("Precision: {}".format(pr))
        print("f1-score : {}".format(f1))
        if probas is not None and roc is not None:
            print("Roc auc  : {}".format(roc))
        print_cm(cm, labels)


def compute_weights(n0, n1, cls_weight=0.5):
    a = [[n0 / float(n1), -1], [n0, 0]]
    b = [0, cls_weight]
    return tuple(np.linalg.solve(a, b))


def parse_bool(s):
    return s.lower() in ['true', '1', 't', 'y']


def parse_list(l, conv=int):
    return [conv(v) for v in l.split(",")] if len(l) > 0 else []


class MultiFileLogger(object):
    def __init__(self, paths=None, files=None, force_create=False):
        self._paths = paths if paths is not None else []
        self._opened_files = [None] * len(self._paths)
        self._passed_files = files if files is not None else []

        # Create the non existing folders path
        if force_create:
            for path in self._paths:
                dirname = os.path.dirname(path)
                if len(dirname) > 0 and not os.path.exists(dirname):
                    os.makedirs(dirname)

    @property
    def _files(self):
        return self._opened_files + self._passed_files

    def __enter__(self):
        try:
            for i, path in enumerate(self._paths):
                self._opened_files[i] = open(path, "w+")
        except IOError as e:
            self._clean()
            raise e

    def log(self, s, end="\n"):
        self._map(lambda f: f.write("{}".format(s) + end))

    def write(self, s):
        self.log(s, end="")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._clean()

    def _map(self, func, opened=None):
        if opened is None:
            files = self._files
        elif opened:
            files = self._opened_files
        else:
            files = self._passed_files

        for f in files:
            if f is not None:
                func(f)

    def _clean(self):
        self._map(lambda f: f.close(), opened=True)
        self._opened_files = [None] * len(self._paths)

    def log_stdout(self):
        return StdOutCapturer(logger=self)


class StdOutCapturer(object):
    def __init__(self, logger=None):
        self._buffer = StringIO()
        self._old_stdout = None
        self._logger = logger

    def __enter__(self):
        self._buffer = StringIO()
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        self._old_stdout = sys.stdout
        sys.stdout = self._buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._old_stdout
        self._old_stdout = None
        if self._logger is not None:
            self._logger.write(self._buffer.getvalue())

    def read(self):
        return self._buffer.getvalue()

