from unittest import TestCase

import tensorflow as tf

import layers

class TestHelpers(TestCase):
    def setUp(self):
        tf.reset_default_graph()  # to make sure tests don't bloat a single graph object

    def testVariable(self):
        dtype = tf.float64
        shape = (5, 5)
        name = "var"
        var = layers._variable(shape, dtype=dtype, name=name, init=None, seed=0, summarize=False, std=0.1)
        self.assertEqual(tuple(var.get_shape().as_list()), (5, 5))
        self.assertEqual(var.dtype.base_dtype, dtype)

    def testConvLayer(self):
        dtype = tf.float64
        in_width, in_height, in_depth = 40, 40, 3
        x = tf.placeholder(dtype, shape=[None, in_width, in_height, in_depth])
        filter_width, filter_height, filter_depth = 3, 3, 25
        layer = layers.conv_layer(x, filter_depth, filter_size=(filter_height, filter_width), dtype=dtype, name="conv")
        self.assertEqual(tuple(layer.get_shape().as_list()), (None, in_height, in_width, filter_depth))
        self.assertEqual(layer.dtype.base_dtype, dtype)

    def testConvLayerWithBatchNormalization(self):
        dtype = tf.float64
        in_width, in_height, in_depth = 40, 40, 3
        x = tf.placeholder(dtype, shape=[None, in_width, in_height, in_depth])
        filter_width, filter_height, filter_depth = 3, 3, 25
        pool_width, pool_height = 3, 3
        layer = layers.conv_layer(x, filter_depth, filter_size=(filter_height, filter_width), dtype=dtype, name="conv",
                                  pool="avg", pooling_kernel=(pool_height, pool_width), batch_norm=True)
        self.assertEqual(tuple(layer.get_shape().as_list()), (None, in_height // 2, in_width // 2, filter_depth))
        self.assertEqual(layer.dtype.base_dtype, dtype)

    def testConvLayerWithPooling(self):
        dtype = tf.float64
        in_width, in_height, in_depth = 40, 40, 3
        x = tf.placeholder(dtype, shape=[None, in_width, in_height, in_depth])
        filter_width, filter_height, filter_depth = 3, 3, 25
        pool_width, pool_height = 3, 3
        layer = layers.conv_layer(x, filter_depth, filter_size=(filter_height, filter_width), dtype=dtype, name="conv",
                                  pool="avg", pooling_kernel=(pool_height, pool_width))
        self.assertEqual(tuple(layer.get_shape().as_list()), (None, in_height // 2, in_width // 2, filter_depth))
        self.assertEqual(layer.dtype.base_dtype, dtype)

    def testFullyConnectedLayer(self):
        dtype = tf.float64
        in_edges = 40
        x = tf.placeholder(dtype, shape=[None, in_edges])
        neurons = 100
        layer = layers.fc_layer(x, neurons, dtype=dtype, name="fcn")
        self.assertEqual(tuple(layer.get_shape().as_list()), (None, neurons))
        self.assertEqual(layer.dtype.base_dtype, dtype)

    def testDeconvLayer(self):
        dtype = tf.float64
        batch_size = 2
        in_width, in_height, in_depth = 40, 40, 3
        x = tf.placeholder(dtype, shape=[batch_size, in_width, in_height, in_depth])
        out_width, out_height, out_depth = 80, 80, 2
        layer = layers.deconv_layer(x, (out_height, out_width, out_depth), batch_size=batch_size, filter_size=(4, 4),
                                    stride=2, name="deconv", dtype=dtype)
        self.assertEqual(tuple(layer.get_shape().as_list()), (batch_size, out_height, out_width, out_depth))
        self.assertEqual(layer.dtype.base_dtype, dtype)

    def testFlatten(self):
        dtype = tf.float64
        batch_size, in_width, in_height, in_depth = 10, 40, 40, 3
        out_depth = 20
        x = tf.placeholder(dtype, shape=[batch_size, in_width, in_height, in_depth])
        layer = layers.conv_layer(x, out_depth, dtype=tf.float64)
        reshaped = layers.flatten_layer(layer, batch_size)
        self.assertEqual(tuple(reshaped.get_shape().as_list()), (batch_size, in_width * in_height * out_depth))
        self.assertEqual(reshaped.dtype.base_dtype, tf.float64)

    def testGetModelComplexity(self):
        dtype = tf.float64
        batch_size, in_width, in_height, in_depth = 10, 40, 40, 3
        out_depth1, out_depth2 = 20, 25
        input_ = tf.placeholder(dtype, shape=[batch_size, in_width, in_height, in_depth])
        x = layers.conv_layer(input_, out_depth1, dtype=tf.float64, filter_size=(3, 3))
        _ = layers.conv_layer(x, out_depth2, dtype=tf.float64, filter_size=(3, 3))

        # weights for two layers + bias
        expected_comp = (in_depth * out_depth1 + out_depth1 * out_depth2) * 9 + out_depth1 + out_depth2
        self.assertEqual(layers.current_model_complexity(), expected_comp)

    def testGetStreamingEvaluationsBinary(self):
        dtype = tf.float32
        batch_size, n_classes = 10, 2
        y_true = tf.placeholder(dtype, shape=[batch_size, n_classes])
        y_pred = tf.placeholder(dtype, shape=[batch_size, n_classes])
        eval = layers.streaming_evaluations(y_pred, y_true)
        self.assertTrue("accuracy" in eval)
        self.assertTrue("recall" in eval)
        self.assertTrue("precision" in eval)
        self.assertTrue("roc_auc" in eval)
        self.assertTrue(isinstance(eval["accuracy"][0], tf.Tensor))
        self.assertTrue(isinstance(eval["recall"][0], tf.Tensor))
        self.assertTrue(isinstance(eval["precision"][0], tf.Tensor))
        self.assertTrue(isinstance(eval["roc_auc"][0], tf.Tensor))
        self.assertTrue(isinstance(eval["accuracy"][1], tf.Tensor))
        self.assertTrue(isinstance(eval["recall"][1], tf.Tensor))
        self.assertTrue(isinstance(eval["precision"][1], tf.Tensor))
        self.assertTrue(isinstance(eval["roc_auc"][1], tf.Tensor))

    def testGetStreamingEvaluationsNotBinary(self):
        dtype = tf.float32
        batch_size, n_classes = 10, 3
        y_true = tf.placeholder(dtype, shape=[batch_size, n_classes])
        y_pred = tf.placeholder(dtype, shape=[batch_size, n_classes])
        eval = layers.streaming_evaluations(y_pred, y_true)
        self.assertTrue("accuracy" in eval)
        self.assertTrue(isinstance(eval["accuracy"][0], tf.Tensor))
        self.assertTrue(isinstance(eval["accuracy"][1], tf.Tensor))

    def testGetEvaluationsBinary(self):
        dtype = tf.float32
        batch_size, n_classes = 10, 2
        y_true = tf.placeholder(dtype, shape=[batch_size, n_classes])
        y_pred = tf.placeholder(dtype, shape=[batch_size, n_classes])
        eval = layers.evaluations(y_pred, y_true)
        self.assertTrue("accuracy" in eval)
        self.assertTrue("recall" in eval)
        self.assertTrue("precision" in eval)
        self.assertTrue("f1-score" in eval)
        self.assertTrue(isinstance(eval["accuracy"], tf.Tensor))
        self.assertTrue(isinstance(eval["recall"], tf.Tensor))
        self.assertTrue(isinstance(eval["precision"], tf.Tensor))
        self.assertTrue(isinstance(eval["f1-score"], tf.Tensor))

    def testGetEvaluationsBinaryWithSegmentationAndNoClassic(self):
        dtype = tf.float32
        batch_size, n_classes = None, 2
        y_true = tf.placeholder(dtype, shape=[batch_size, n_classes])
        y_pred = tf.placeholder(dtype, shape=[batch_size, n_classes])
        eval = layers.evaluations(y_pred, y_true, classic=False, segmentation=True)
        self.assertFalse("accuracy" in eval)
        self.assertFalse("recall" in eval)
        self.assertFalse("precision" in eval)
        self.assertFalse("f1-score" in eval)
        self.assertTrue("positive_iou" in eval)
        self.assertTrue("negative_iou" in eval)
        self.assertTrue("mean_iou" in eval)
        self.assertTrue("weighted_mean_iou" in eval)
        self.assertTrue(isinstance(eval["positive_iou"], tf.Tensor))
        self.assertTrue(isinstance(eval["negative_iou"], tf.Tensor))
        self.assertTrue(isinstance(eval["mean_iou"], tf.Tensor))
        self.assertTrue(isinstance(eval["weighted_mean_iou"], tf.Tensor))


    def testGetEvaluationsNotBinary(self):
        dtype = tf.float32
        batch_size, n_classes = 10, 3
        y_true = tf.placeholder(dtype, shape=[batch_size, n_classes])
        y_pred = tf.placeholder(dtype, shape=[batch_size, n_classes])
        eval = layers.evaluations(y_pred, y_true)
        self.assertTrue("accuracy" in eval)
        self.assertTrue(isinstance(eval["accuracy"], tf.Tensor))

    def testGetBatchNormalLayer(self):
        dtype = tf.float64
        in_edges = 40
        x = tf.placeholder(dtype, shape=[None, in_edges])
        layer = layers.batch_norm_layer(x, eps=1e-8, dtype=dtype, summarize=True)
        self.assertEqual(tuple(layer.get_shape().as_list()), tuple(x.get_shape().as_list()))
        self.assertEqual(layer.dtype.base_dtype, dtype)
