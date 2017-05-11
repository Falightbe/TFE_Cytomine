import operator

import tensorflow as tf
from functools import reduce

from tensorflow.contrib import metrics

INIT_NORMAL = "truncated_normal"
INIT_ZERO = "zeros"
INIT_CONSTANT = "constant"


def _variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar(name + '/sttdev', stddev)
    tf.summary.scalar(name + '/max', tf.reduce_max(var))
    tf.summary.scalar(name + '/min', tf.reduce_min(var))
    tf.summary.histogram(var.op.name, var)


def _minimal_variable_summary(var, name):
    """Attach a little less summaries to a Tensor than variable_summaries (no histogram, no stddfev."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    tf.summary.scalar(name + '/max', tf.reduce_max(var))
    tf.summary.scalar(name + '/min', tf.reduce_min(var))


# https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
def _activation_summary(x, name):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tf.summary.histogram(name + '/activation', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def _variable(dim, dtype, name=None, init=None, **init_params):
    """Create a tensorflow variable and optionally initializes it

    Parameters
    ----------
    dim: tuple/list
        The dimension of the tensor
    name: basestring
        Name of the variable
    init: object
        The variable initialization method
    summarize: bool
        True for creating summaries on the various interesting components
    init_params: dict
        Dictionnary of parameters to pass to the variable (seed, dtype and shape excluded)
        -> INIT_ZERO: none
        -> INIT_CONST: value=__CONST_VALUE__
        -> INIT_NORMAL: mean=__MEAN_VALUE__, stddev=__STDDEV_VALUE__, seed=__SEED_VALUE__
    Returns
    -------

    """
    if init is None or init == INIT_ZERO:
        var_init = tf.zeros(dim, dtype=dtype)
    elif init == INIT_CONSTANT:
        var_init = tf.constant(**init_params, dtype=dtype, shape=dim)
    elif init == INIT_NORMAL:
        var_init = tf.truncated_normal(dim, dtype=dtype, **init_params)
    else:
        raise ValueError("Unknown initialization method '{}'.".format(init))
    return tf.Variable(initial_value=var_init, name=name, dtype=dtype)


def conv_layer(input, depth, name="conv", filter_size=(3, 3), strides=(1, 1, 1, 1), dtype=tf.float32,
               seed=0, pool=None, pooling_stride=(2, 2), pooling_kernel=(2, 2), dropout=None, summarize=False,
               init=INIT_NORMAL, batch_norm=False, eps=1e-8, **init_params):
    """Build a 2D convolutional layer

    Parameters
    ----------
    input: tf.Tensor
        Input of the conv. layer
    depth: int
        Number of filters in the conv. layer
    name: basestring (default=None)
        Name of the layer
    filter_size: tuple (subtype: (int, int), default: (3, 3))
        Height and width of the filter
    strides: tuple (subtype: (int, int, int, int), default: (1, 1, 1, 1))
        Stride for the conv. filter
    dtype: tf.DType (default: tf.float32)
        Type of the weight and bias variables
    seed: int (default: 0)
        Random seed for the layer initialization
    pool: basestring (default: None)
        The type of pooling to apply (among "avg" and "max). None for no pooling.
    pooling_stride: tuple (basetype: (int, int), default: (2, 2))
        The vertical and horizontal strides to apply when performing pooling (ignored if pool is None).
    pooling_kernel: typle (basetype: (int, int), default: (2, 2))
        The height and width size of the pooling kernel (ignored if pool is None)
    dropout: float (default: None)
        The dropout probability to apply on the conv. layer (None for no dropout)
    summarize: bool
        True for creating summaries on the various interesting components
    init: str
        The initialization method
    batch_norm: bool
        Batch normalization
    eps: float
        For numerical stability
    init_params: dict
        Variable initialization parameter
    Returns
    -------

    """
    # todo better initialization
    with tf.variable_scope(name):

        # build convolution layer
        in_channels = input.get_shape().as_list()[-1]
        _filter = _variable(
            [filter_size[0], filter_size[1], in_channels, depth],  # input dim is taken from the input's output dim
            dtype=dtype,
            name="filter",
            seed=seed,
            init=init,
            **init_params
        )
        conv = tf.nn.conv2d(
            input, _filter,
            strides=list(strides),
            padding="SAME",
            name="conv2d"
        )

        bias = None
        if batch_norm:  # normalize input
            to_activate = batch_norm_layer(
                conv, dtype=dtype, eps=eps,
                summarize=summarize, keep_last=True,
                name="batch_normalization"
            )
        else:
            bias = _variable([depth], dtype=dtype, name="biases")
            to_activate = tf.nn.bias_add(conv, bias)

        relu = tf.nn.relu(to_activate, "activation")

        if summarize:
            with tf.variable_scope("summaries"):
                if bias is not None:
                    _variable_summaries(bias, name="bias")
                _variable_summaries(_filter, name="filter")
                _activation_summary(relu, name="activation")  # Requires a feed dict in Trainer !

        if pool == "max":
            interm = tf.nn.max_pool(
                relu,
                ksize=[1, pooling_kernel[0], pooling_kernel[1], 1],
                strides=[1, pooling_stride[0], pooling_stride[1], 1],
                padding="SAME",
                name="max_pooling"
            )
        elif pool == "avg":  # mean pooling
            interm = tf.nn.avg_pool(
                relu,
                ksize=[1, pooling_kernel[0], pooling_kernel[1], 1],
                strides=[1, pooling_stride[0], pooling_stride[1], 1],
                padding="SAME",
                name="avg_pooling"
            )
        elif pool is None:
            interm = relu
        else:
            raise ValueError("Unknown pooling method '{}' for conv. layer '{}'".format(pool, name))

        if dropout is not None:  # add dropout if necessary
            return tf.nn.dropout(interm, dropout, seed=seed, name="dropout")
        else:
            return interm


def fc_layer(input, neurons, seed=0, dtype=tf.float32, dropout=None, name=None, summarize=False,
             init=INIT_NORMAL, batch_norm=False, eps=1e-8, **init_params):
    """Build a fully connected neuron layer.

    Parameters
    ----------
    input: tf.Tensor
        The input layer
    neurons: int
        Number of neurons in the fully connected layer
    seed: int (default: 0)
        Random seed for the layer initialization
    dtype: tf.DType (default: tf.float32)
        Type of the weight and bias variables
    dropout: float (default: None)
        The dropout probability to apply on the fully connected layer (None for no dropout)
    name: basestring
        Name of the layer
    summarize: bool
        True for creating summaries on the various interesting components
    init: str
        The initialization method
    batch_norm: bool
        Batch normalization
    eps: float
        For numerical stability
    init_params: dict
        Variable initialization parameter
    Returns
    -------

    """
    with tf.variable_scope(name):

        # Build fully connected layer
        in_shape = input.get_shape().as_list()[-1]
        weight = _variable(
            [in_shape, neurons],
            dtype=dtype,
            name="weights",
            seed=seed,
            init=init,
            **init_params
        )
        to_activate = tf.matmul(input, weight)
        bias = None
        # apply batch normalization on input
        if batch_norm:
            to_activate = batch_norm_layer(to_activate, dtype=dtype, eps=eps, summarize=summarize, name="batch_normalization")
        else:
            bias = _variable([neurons], dtype=dtype, name="biases")
            to_activate = tf.nn.bias_add(to_activate, bias)

        layer = tf.nn.relu(to_activate, name="activation")

        if summarize:
            with tf.variable_scope("summaries"):
                if bias is not None:
                    _variable_summaries(bias, name="bias")
                _variable_summaries(weight, name="weights")
                _activation_summary(layer, name="activation")  # Requires a feed dict in Trainer !

        if dropout is None:
            return layer

        return tf.nn.dropout(layer, dropout, seed=seed, name="dropout")


def deconv_layer(input, out_dim, batch_size=1, filter_size=(3, 3), stride=1, dtype=tf.float32, name=None, seed=0,
                 dropout=None, init=INIT_NORMAL, summarize=False, **init_params):
    """Builds a "deconvolution" layer (i.e. a transposed 2d convolution layer)

    Parameters
    ----------
    input: Tensor
        Input layer
    out_dim: tuple (int, int, int)
        Dimensions of the output layer: (height, width, channels)
    batch_size: int
        Batch size
    filter_size: tuple (int, int)
        Size of the filter
    stride: int
        Deconvolution stride (the same is applied vertically and horizontally
    dtype: int
        Filter type
    name: str
        The string
    seed: int
        Seed for variable initialization
    dropout: tf.placeholder or number
        The dropout probability
    init: str
        The initialization method
    init_params: dict
        Variable initialization parameter
    :return:
    """
    with tf.variable_scope(name):
        height, width, channels = out_dim
        in_dim = input.get_shape()[-1].value
        _filter = _variable(
            [filter_size[0], filter_size[1], channels, in_dim],
            dtype=dtype,
            name="filter",
            seed=seed,
            init=init,
            **init_params
        )
        deconv = tf.nn.conv2d_transpose(
            input, _filter,
            output_shape=tf.stack([batch_size, height, width, channels]),
            strides=[1, stride, stride, 1], padding="SAME",
            name="deconvolution"
        )

        if summarize:
            with tf.name_scope("summaries"):
                _variable_summaries(_filter, "filters")
                _activation_summary(deconv, "activation")

        if dropout is None:
            return deconv
        return tf.nn.dropout(deconv, keep_prob=dropout, seed=seed, name="dropout")


def optimizer(y_pred, y_true, opt="adam", sample_weights=None, summarize=False, name="optimizer", **opt_params):
    """Return an optimizer operator for the given prediction

    Parameters
    ----------
    y_pred: Tensor
        Must have a dimension [batch_size, [...], n_classes]
    y_true: Tensor
        Must have a dimension [batch_size, [...], n_classes]
    opt: basestring
        Optimization procedure among ("adam", "rmsprop")
    learning_rate: float
        Learning rate
    sample_weights: Tensor (default: None)
        The sample weights. Must have dimensions [batch_size, [...]]
    :return:
    """
    with tf.variable_scope(name):
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, name="softmax_cross_entropy")
        if sample_weights is not None:
            loss = tf.reduce_mean(softmax * sample_weights, name="loss")
        else:
            loss = tf.reduce_mean(softmax, name="loss")

        if summarize:
            tf.summary.scalar("loss", loss)   # Requires a feed dict in Trainer !

        if opt == "rmsprop":
            return loss, tf.train.RMSPropOptimizer(**opt_params).minimize(loss)
        else:  # adam
            return loss, tf.train.AdamOptimizer(**opt_params).minimize(loss)


def flatten_layer(layer, batch_size=None):
    """Flatten the given layer. If batch size is known it can be passed"""
    if batch_size is not None:
        return tf.reshape(layer, [batch_size, -1])
    else:
        shapes = layer.get_shape().as_list()[1:]
        return tf.reshape(layer, [-1, reduce(operator.mul, shapes)])


def current_model_complexity():
    """Complexity of the current graph in terms of number of trainable parameters"""
    return sum([reduce(operator.mul, variable.get_shape().as_list(), 1) for variable in tf.trainable_variables()])


def _cohens_kappa(cm, name="cohens_kappa"):
    """
    Given a confusion matrix, computes the cohen's kappa agreement score
    source: https://onlinecourses.science.psu.edu/stat509/node/162
    """
    with tf.name_scope(name):
        n = tf.reduce_sum(cm)
        p0 = tf.divide(tf.reduce_sum(tf.diag_part(cm)), n, name="p0")
        pe = tf.divide(tf.reduce_sum(tf.reduce_sum(cm, axis=0) * tf.reduce_sum(cm, axis=1)), n * n, name="pe")
        return tf.divide(p0 - pe, tf.subtract(tf.constant(1.0, dtype=pe.dtype), pe), name="kappa")


def evaluations(probas, y_true, summarize=False, name="evaluations", classic=True, segmentation=False):
    """Return a dict containing a bunch of evaluations metrics to be used for the given problem.
    Especially, the dictionary maps metric names with the metric op/tensor.
    If the metric has no inner local variable, then update_od = tensor.

    Binary
    ------
     * Classic metrics:
        - accuracy
        - recall
        - precision
        - f1-score
     * Segmentation metrics:
        - mean intersection over union (mean_iou)
        - freq. weighted intersection over union (weighted_mean_iou)
        - positive intersection over union (positive_iou)
        - negative intersection over union (negative_iou)
        - Cohen's kappa (cohens_kappa)

    Multiclass
    ----------
     * Classic metrics:
        - accuracy
     * Segmentation metrics:
        - Cohen's kappa (cohens_kappa)

    Parameters:
    -----------
    probas: tf.float64 Tensor
        Predicted probabilities
    y_true: tf.float64 Tensor
        Ground truth, can be a one hot class or probabilities encoding
    summarize: bool
        True for attaching a summary to the produced metrics
    """
    # Evaluate model
    with tf.variable_scope(name):
        # reshape to match metrics expected inputs
        last_dim = y_true.get_shape()[-1].value
        y_true = tf.reshape(y_true, [-1, last_dim])
        probas = tf.reshape(probas, [-1, last_dim])

        # compute metrics
        actual, pred = tf.argmax(y_true, axis=1), tf.argmax(probas, axis=1)
        cm = tf.confusion_matrix(actual, pred, num_classes=last_dim, name="confusion_matrix", dtype=actual.dtype)
        count = tf.reduce_sum(cm)
        correct = tf.reduce_sum(tf.diag_part(cm))
        accuracy = tf.divide(correct, count, name="accuracy")

        metrics_dict = dict()

        # Generic classic metrics
        if classic:
            if summarize:
                tf.summary.scalar("accuracy", accuracy)

            metrics_dict.update({
                "accuracy": accuracy
            })

        # Generic segmentation metrics
        if segmentation:
            cohens_kappa = _cohens_kappa(cm)

            if summarize:
                tf.summary.scalar("cohens_kappa", cohens_kappa)

            metrics_dict.update({
                "cohens_kappa": cohens_kappa
            })

        if last_dim == 2:
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            # Compute classic binary metrics
            if classic:
                recall = tf.divide(tp, tf.maximum(tp + fn, 1), name="recall")
                precision = tf.divide(tp, tf.maximum(tp + fp, 1), name="precision")
                f1 = tf.divide(2 * recall * precision, tf.maximum(recall + precision, 1e-8), name="f1-score")

                if summarize:
                    tf.summary.scalar("recall", recall)
                    tf.summary.scalar("precision", precision)
                    tf.summary.scalar("f1-score", precision)

                metrics_dict.update({
                    "recall": recall,
                    "precision": precision,
                    "f1-score": f1
                    # TODO ROC
                })

            # Compute binary segmentation metrics
            if segmentation:
                positive_iou = tf.divide(tp, tf.maximum(tp + fn + fp, 1), name="positive_iou")
                negative_iou = tf.divide(tn, tf.maximum(tn + fn + fp, 1), name="negative_iou")
                mean_iou = tf.multiply(positive_iou + negative_iou, 0.5, name="mean_iou")

                # counts
                total_count = tp + fn + fp + tn
                positive_count = tp + fn
                negative_count = tf.subtract(total_count, positive_count)

                weighted_mean_iou = tf.add(
                    positive_iou * (positive_count / total_count),
                    negative_iou * (negative_count / total_count),
                    name="weighted_mean_iou"
                )

                if summarize:
                    tf.summary.scalar("positive_iou", positive_iou)
                    tf.summary.scalar("negative_iou", negative_iou)
                    tf.summary.scalar("mean_iou", mean_iou)
                    tf.summary.scalar("weighted_mean_iou", weighted_mean_iou)

                metrics_dict.update({
                    "positive_iou": positive_iou,
                    "negative_iou": negative_iou,
                    "mean_iou": mean_iou,
                    "weighted_mean_iou": weighted_mean_iou
                })

        return metrics_dict


def streaming_evaluations(probas, y_true, name="streaming_evaluations", classic=True, segmentation=False):
    """Return a dict containing a bunch of evaluations metrics to be used for the given problem.
    Especially, the dictionary maps metric names with a tuple containing (tensor, update_op).
    If the metric has no inner local variable, then update_od = tensor.

    Binary
    ------
     * Classic metrics:
        - accuracy
        - recall
        - precision
        - f1-score
        - roc
     * Segmentation metrics:
        - mean intersection over union (mean_iou)
        - positive binary intersection over union (positive_iou)
        - negative binary intersection over union (negative_iou)
        - Cohen's kappa (cohens_kappa)

    Multiclass
    ----------
     * Classic metrics:
        - accuracy
     * Segmentation metrics:
        - mean intersection over union (mean_iou)

    Parameters:
    -----------
    probas: tf.float64 Tensor
        Predicted probabilities
    y_true: tf.float64 Tensor
        Ground truth, a one hot class encoding
    summarize: bool
        True for attaching a summary to the produced metrics
    """
    # Evaluate model
    with tf.variable_scope(name):
        # reshape to match metrics expected inputs
        last_dim = y_true.get_shape()[-1].value
        y_true = tf.reshape(y_true, [-1, last_dim])
        probas = tf.reshape(probas, [-1, last_dim])

        # compute metrics
        actual, pred = tf.argmax(y_true, axis=1), tf.argmax(probas, axis=1)
        acc, update_acc = tf.contrib.metrics.streaming_accuracy(pred, actual, name="accuracy")

        metrics_dict = dict()

        if classic:
            metrics_dict.update({
                "accuracy": (acc, update_acc)
            })

        if segmentation:
            mean_iou, update_miou = metrics.streaming_mean_iou(pred, actual, last_dim)
            metrics_dict.update({
                "mean_iou": (mean_iou, update_miou)
            })

        if last_dim == 2:
            actual, pred = tf.cast(actual, dtype=tf.bool), tf.cast(pred, dtype=tf.bool)
            tp, update_tp = metrics.streaming_true_positives(pred, actual, name="tp")
            fn, update_fn = metrics.streaming_false_negatives(pred, actual, name="fn")
            fp, update_fp = metrics.streaming_false_positives(pred, actual, name="fp")
            tn, update_tn = metrics.streaming_true_negatives(pred, actual, name="tn")
            cm = [[tn, fp], [fn, tp]]

            # Compute classic binary metrics
            if classic:
                roc, update_roc = metrics.streaming_auc(probas, tf.greater(y_true, 0.5), curve='ROC', name="roc_auc")
                recall = tf.divide(tp, tf.maximum(tp + fn, 1), name="recall")
                precision = tf.divide(tp, tf.maximum(tp + fp, 1), name="precision")
                f1 = tf.divide(2 * recall * precision, tf.maximum(recall + precision, 1e-8), name="f1-score")

                metrics_dict.update({
                    "recall": (recall, recall),
                    "precision": (precision, precision),
                    "f1-score": (f1, f1),
                    "roc_auc": (roc, update_roc),
                    "tp": (None, update_tp),
                    "tn": (None, update_tn),
                    "fp": (None, update_fp),
                    "fn": (None, update_fn)
                })

            # Compute binary segmentation metrics
            if segmentation:
                positive_iou = tf.divide(tp, tf.maximum(tp + fn + fp, 1), name="positive_iou")
                negative_iou = tf.divide(tn, tf.maximum(tn + fn + fp, 1), name="negative_iou")
                cohens_kappa = _cohens_kappa(cm)

                metrics_dict.update({
                    "positive_iou": (positive_iou, positive_iou),
                    "negative_iou": (negative_iou, negative_iou),
                    "cohens_kappa": (cohens_kappa, cohens_kappa)
                })

        return metrics_dict


def norm_input_layer(input, eps=1e-8, name="input_norm"):
    """
    Parameters
    ----------
    input: Tensor
        Input tensor
    axis: tuple
        Indexes of the axis to squeeze
    name: str
        Name of the scope

    Returns
    -------
    output: Tensor
    """
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(input, list(range(input.get_shape().ndims - 1)), keep_dims=True)
        stddev = tf.sqrt(var)
        adjusted_stddev = tf.maximum(stddev, eps)
        return tf.divide(input - mean, adjusted_stddev)


def batch_norm_layer(input, dtype=tf.float32, eps=1e-8, summarize=False, keep_last=False, name="batch_norm"):
    """Batch normalization error

    Parameters
    ----------
    input: tf.Tensor
        Input layer (activations from a previous layer
    dtype: tf.DType
        dtype of the scale and offset variables
    eps: float
        Epsilon for numerical stability
    summarize: bool
        True for adding a summary
    name: str
        Name of the layer

    Returns
    -------
    batch_norm: tf.Tensor
        Batch normalized activations of input
    """
    with tf.variable_scope(name):
        # compute mean and var
        input_shape = input.get_shape()
        n_dims_to_shrink = input_shape.ndims - (1 if keep_last else 0)
        axes = list(range(n_dims_to_shrink))
        mean, var = tf.nn.moments(input, axes=axes, name="moments", keep_dims=True)

        # define offset and scale
        shape = [input_shape[-1]] if keep_last else []
        offset = _variable(shape, dtype=dtype, name="offset", init=INIT_ZERO)
        scale = _variable(shape, dtype=dtype, name="scale", init=INIT_CONSTANT, value=1)

        # apply batch_norm
        batch_norm = tf.nn.batch_normalization(input, mean, var, offset, scale, eps, name="batch_norm")
        if summarize:
            with tf.name_scope("summaries"):
                if keep_last:
                    _minimal_variable_summary(scale, "scale")
                    _minimal_variable_summary(offset, "offset")
                else:
                    tf.summary.scalar("scale", scale)
                    tf.summary.scalar("offset", offset)
                _activation_summary(batch_norm, name="activation")
        return batch_norm
