from argparse import ArgumentParser

import tensorflow as tf
from PIL import Image
from sklearn.utils import check_random_state

import nn
from generic.dataset import ImageSegmentationDataset


def get_model(height, width, channel, n_classes, batch_size=None, random_state=0):
    random_state = check_random_state(random_state)
    X_in = tf.placeholder(tf.float32, shape=[batch_size, height, width, channel], name="input")
    y_out = tf.placeholder(tf.float32, shape=[batch_size, height, width, n_classes], name="ground_truth")
    keep_proba = tf.placeholder(tf.float32, name="keep_proba")

    # large conv
    filters = [96, 96, 96]
    norm = nn.norm_input_layer(X_in)
    conv = nn.conv_layer(norm, filters[0], name="conv1", pool="avg", pooling_kernel=(3, 3), dropout=keep_proba, seed=nn.randint(random_state), summarize=True, stddev=0.05)
    conv = nn.conv_layer(conv, filters[1], name="conv2", pool="avg", pooling_kernel=(3, 3), dropout=keep_proba, seed=nn.randint(random_state), summarize=True, stddev=0.05)
    conv = nn.conv_layer(conv, filters[2], name="conv3", pool="avg", pooling_kernel=(3, 3), dropout=keep_proba, seed=nn.randint(random_state), summarize=True, stddev=0.05)
    deconv = nn.deconv_layer(conv, (height // 4, width // 4, n_classes), stride=2, batch_size=batch_size, name="deconv1", seed=nn.randint(random_state), dropout=keep_proba, summarize=True, stddev=0.05)
    deconv = nn.deconv_layer(deconv, (height // 2, width // 2, n_classes), stride=2, batch_size=batch_size, name="deconv2", seed=nn.randint(random_state), dropout=keep_proba, summarize=True, stddev=0.05)
    deconv = nn.deconv_layer(deconv, (height, width, n_classes), stride=2, batch_size=batch_size, name="deconv3", seed=nn.randint(random_state), dropout=keep_proba, summarize=True, stddev=0.05)
    return X_in, y_out, keep_proba, deconv


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--summary", dest="summary")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--override", dest="override", action="store_true")
    parser.add_argument("--n_iter", dest="n_iter", default=1, type=int)
    parser.add_argument("--name", dest="name")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.set_defaults(override=False)
    params, unknown = parser.parse_known_args(argv)
    dataset = ImageSegmentationDataset(params.dataset, dirs=["train", "validation", "test"], ignore_ext=True)

    # dataset parameters
    window_height, window_width = 256, 256
    batch_size = 16

    # build network
    X_in, y_out, keep_proba, deconv = get_model(window_height, window_width, 3, n_classes=2, batch_size=batch_size)
    learning_rate = tf.placeholder(tf.float64, name="learning_rate")
    loss, optimizer = nn.optimizer(deconv, y_out, opt="adam", learning_rate=learning_rate, summarize=True)
    softmax = tf.nn.softmax(deconv, name="softmax")
    evaluations = nn.evaluations(softmax, y_out, summarize=True, segmentation=True)
    print("Model complexity: {}".format(nn.current_model_complexity()))

    # image summaries
    tf.summary.image("mask", softmax[:, :, :, 1:2])
    tf.summary.image("input", X_in)
    tf.summary.image("y_out", y_out[:, :, :, 1:2])

    # create initializer and summary op
    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()

    random_state = check_random_state(params.seed)  # init for training

    with tf.Session() as session:
        session.run([init])
        writer = nn.get_summary_writer(session, params.summary, params.name, include_datetime=True)

        if params.override:
            print("Override if checkpoint exists!")
            nn.restore_session(session, params.model, params.name)

        for i in range(params.n_iter):
            x, y = dataset.windows_batch(
                batch_size,
                dims=(window_height, window_width),
                dirs=["train"],
                random_state=random_state,
                one_hot=True,
                classes=[0, 255],
                open_fn=Image.open
            )

            print("Step {}".format(i), end="")

            _loss, _, _biou, _acc = session.run(
                [loss, optimizer, evaluations["positive_iou"], evaluations["accuracy"]],
                feed_dict={
                    X_in: x,
                    y_out: y,
                    keep_proba: 0.6,
                    learning_rate: 1e-3
                }
            )

            print(" -> loss {} - biou {} - acc {}".format(_loss, _biou, _acc))

            if i % 10 == 0:
                _summaries, = session.run(
                    [summaries],
                    feed_dict={
                        X_in: x,
                        y_out: y,
                        keep_proba: 1.0
                    }
                )
                writer.add_summary(_summaries, i)

        if params.n_iter > 0:
            nn.save_session(session, params.model, params.name)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])