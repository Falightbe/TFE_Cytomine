from abc import abstractmethod, ABCMeta, abstractproperty

import numpy as np
import tensorflow as tf
from nn.subwindows import random_subwindows, COLORSPACE_RGB
from sklearn.utils import check_random_state

from generic.dataset import ImageProvider, ImageDataset
from nn.util import progress_bar, _normalize


def predict_proba(sess, X, y, images, classes, probas, batch_size=64, more_feed_dict=None):
    """Predict class probabilities for all input images (in batch, to avoid memory error)
    Parameters
    ----------
    sess: tf.Session
    X: tf.Tensor
        Input tensor
    y: tf.Tensor
        Output tensor
    images: np.array
        Array containing the images to be classified
    classes: np.array
        Actual classes of those images
    probas: tf.Operator
        Op for computing the probabilities
    batch_size: int
        Batch size
    more_feed_dict: dict
        Some additionnal information to pass as feed dict
    """
    all_probas = None
    nb_batches = (len(images) // batch_size) + (1 if len(images) % batch_size > 0 else 0)
    for batch_index in range(nb_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end]
        feed_dict = {
            X: batch,
            y: classes[start:end]
        }
        if more_feed_dict is not None:
            feed_dict.update(more_feed_dict)
        predicted = sess.run([probas], feed_dict)
        all_probas = predicted[0] if all_probas is None else np.vstack((all_probas, predicted[0]))
    return all_probas


def predict_proba_with_windows(sess, X, images, n_classes, probas, n_subwindows, batch_size=16, random_state=None,
                               feed_dict=None, **window_args):
    """
    Predict probabilities from aggregated subwindows

    Parameters
    ----------
    sess: tf.Session
    X: tf.Tensor
        Placeholder of network's inputs
    images: iterable
        List of image files of which the probabilities must be predicted
    probas:
    n_subwindows:
    batch_size:
    random_state:
    feed_dict:
    window_args:
    :return:
    """
    random_state = check_random_state(random_state)
    all_probas = np.zeros((len(images), n_classes))
    nb_batches = (len(images) // batch_size) + (1 if len(images) % batch_size > 0 else 0)
    target_height, target_width = X.get_shape().as_list()[1:-1]
    _X, _ = random_subwindows(
        images,
        np.zeros((len(images),), dtype=np.int8),  # don't care about output classes for predicting probabilities
        n_subwindows,
        random_state=random_state,
        target_height=target_height,
        target_width=target_width,
        **window_args
    )
    for batch_index in range(nb_batches):
        # start and end of current batch
        start = batch_index * (batch_size * n_subwindows)
        end = min(start + batch_size * n_subwindows, _X.shape[0])

        # compute for batch
        _feed_dict = {
            X: _X[start:end]
        }

        if feed_dict is not None:
            _feed_dict.update(feed_dict)

        predicted, = sess.run([probas], feed_dict=_feed_dict)

        # aggregate
        aggr_start = batch_size * batch_index
        aggr_end = min(aggr_start + batch_size, len(images))

        for i in range(aggr_end - aggr_start):
            all_probas[batch_index * batch_size + i] = np.mean(predicted[i * n_subwindows:(i+1) * n_subwindows], axis=0)

        print("\r{}".format(progress_bar(batch_index, nb_batches)), end="")
    print("\r{}".format(progress_bar(nb_batches, nb_batches)))
    return all_probas


class Trainer(metaclass=ABCMeta):
    def __init__(self, image_provider: ImageProvider, session, input, output, loss, optimizer, batch_size,
                 feed_dict=None, summary_period=10, print_period=1):
        self._image_provider = image_provider
        self._batch_size = batch_size
        self._session = session
        self._feed_dict = feed_dict if feed_dict is not None else dict()
        self._input = input
        self._output = output
        self._loss = loss
        self._optimizer = optimizer
        self._summary_period = summary_period
        self._print_period = print_period
        # Setters
        self._disp_metric = None
        self._disp_metric_op = None
        self._train_update_ops = None
        self._summaries = None
        self._summary_writer = None
        self._summaries_feed_dict = None

    def set_displayed_metric(self, name, op):
        self._disp_metric = name
        self._disp_metric_op = op

    def set_train_update_ops(self, update_ops):
        self._train_update_ops = update_ops

    def set_summaries(self, writer, summaries, feed_dict=None):
        self._summaries = summaries
        self._summary_writer = writer
        self._summaries_feed_dict = feed_dict if feed_dict is not None else dict()

    def train_epoch(self, epoch_nb, random_state=None):
        random_state = check_random_state(random_state)
        self._set_epoch_state(random_state)
        # start iterations
        losses, metric_evals = list(), list()
        n_iterations = int(self.n_iterations)
        for i in range(n_iterations):
            x, y, loss, metric = self._iteration(random_state)

            # eval metric to display if there is one
            if metric is not None:
                metric_evals.append(metric)

            losses.append(loss)

            if i % self._print_period == 0:
                metric = None if self._disp_metric is None else np.mean(metric_evals[-10:])
                print("\r" + self.summary_string(i, np.mean(losses[-10:]), metric=metric), end="")

            if i % self._summary_period == 0 and self._has_summaries():
                summaries, = self._session.run([self._summaries], feed_dict=self._fill_summaries_feed_dict(x, y))
                self._summary_writer.add_summary(summaries, n_iterations * epoch_nb + i)

        self._clear_epoch_state()
        metric = None if self._disp_metric is None else np.mean(metric_evals[-10:])
        print("\r" + self.summary_string(n_iterations, np.mean(losses[-10:]), metric=metric))

    def summary_string(self, iter, loss, metric=None, padding=10):
        summary = "{} - loss {:3.5f}".format(progress_bar(iter, self.n_iterations), loss)
        if metric is not None:
            summary += " - {} {:3.5f}".format(self._disp_metric, metric)
        summary += padding * ""
        return summary

    def _has_summaries(self):
        return self._summaries is not None and self._summary_writer is not None

    def _fill_summaries_feed_dict(self, input, output):
        feed = {self._input: input, self._output: output}
        feed.update(self._summaries_feed_dict)
        return feed

    def _iteration(self, random_state):
        """Execute an optimization iteration 'iter' of the network on a random batch of data
        Returns
        -------
        x: ndarray
            Inputs
        y: ndarray
            Outputs
        loss: float
            Current batch loss
        """
        x, y = self._input_data(random_state)
        feed_dict = self._fill_feed_dict(x, y)
        ops = self._train_ops_list()
        results = self._session.run(ops, feed_dict=feed_dict)
        return x, y, results[0], results[-1]

    def _fill_feed_dict(self, input, output):
        feed = {self._input: input, self._output: output}
        feed.update(self._feed_dict)
        return feed

    def _train_ops_list(self):
        """Order:
            1) Loss op
            2) Optimizer op
            3 ... K-1) train update ops
            K) If defined, metric op or None op
        """
        ops = [self._loss, self._optimizer]
        if self._train_update_ops is not None:
            ops.extend(self._train_update_ops)
        if self._disp_metric_op is not None:
            ops.append(self._disp_metric_op)
        else:
            ops.append(tf.constant(None))
        return ops

    def _set_epoch_state(self, random_state=None):
        """Set the trainer in a valid state. Override for defining a state"""
        pass

    def _clear_epoch_state(self):
        """Reset the state of the trainer. Override if you have defined set_state"""
        pass

    @abstractmethod
    def _input_data(self, random_state):
        """Return a random batch of data to be processed by the network for training
        Returns
        -------
        X: ndarray
            Training input data (batch_size, height, width, channels)
        y: ndarray
            Output data (batch_size, ?, [...])
        """
        pass

    @abstractproperty
    def n_iterations(self):
        """

        Returns
        -------
        iter: int
        """
        pass

    @property
    def image_provider(self):
        return self._image_provider

    @property
    def batch_size(self):
        return self._batch_size


class BatchTrainer(Trainer):
    """Assume fixed size images and performs batch training on them"""
    def __init__(self, **kwargs):
        super(BatchTrainer, self).__init__(**kwargs)

    def _input_data(self, random_state):
        X_batch, y_batch = self.image_provider.batch(self.batch_size, random_state=random_state)
        _X = np.array(ImageDataset.load_images(X_batch))
        return _X, y_batch

    @property
    def n_iterations(self):
        return len(self.image_provider) // self.batch_size


class RandomSubwindowsTrainer(Trainer):
    def __init__(self, n_subwindows, min_size, max_size, normalize=True, target_width=16, target_height=16, n_jobs=-1,
                 colorspace=COLORSPACE_RGB, **kwargs):
        super(RandomSubwindowsTrainer, self).__init__(**kwargs)
        self._normalize = normalize
        self._n_subwindows = n_subwindows
        self._min_size, self._max_size = min_size, max_size
        self._target_width, self._target_height = target_width, target_height
        self._n_jobs = n_jobs
        self._colorspace = colorspace
        self._windows = None
        self._outputs = None

    def _set_epoch_state(self, random_state=None):
        x, y = self.image_provider.all()
        self._windows, self._outputs = random_subwindows(
            x, y, self._n_subwindows, random_state=random_state,
            min_size=self._min_size, max_size=self._max_size,
            target_height=self._target_height, target_width=self._target_width,
            n_jobs=-1, colorspace=self._colorspace, backend="multiprocessing"
        )
        if self._normalize:
            self._windows = _normalize(self._windows)

    def _clear_epoch_state(self):
        self._windows = None
        self._outputs = None

    def _input_data(self, random_state):
        idx = random_state.choice(self._windows.shape[0], size=(self.batch_size,), replace=True)
        return self._windows[idx], self._outputs[idx]

    @property
    def n_iterations(self):
        # All images could be seen during the epoch
        return (len(self.image_provider) * self._n_subwindows) // self.batch_size
