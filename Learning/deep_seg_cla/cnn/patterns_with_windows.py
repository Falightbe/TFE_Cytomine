from argparse import ArgumentParser
import os

from generic.dataset import ImageClassificationDataset, ImageProvider
from nn.subwindows import COLORSPACE_RGB
from nn.training import RandomSubwindowsTrainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from sklearn.utils import check_random_state

import nn


def randint(state):
    return state.randint(9999999)


def get_model(height, width, channel, n_classes, batch_size=None, random_state=0):
    random_state = check_random_state(random_state)
    keep_proba = tf.placeholder(tf.float32, name="keep_proba")
    X_batch = tf.placeholder(tf.float32, shape=(batch_size, height, width, channel), name="X_in")
    y_batch = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name="y_true")

    # build network
    # 128 -> 128 -> 42 -> 21 -> 10 -> 5
    norm = nn.norm_input_layer(X_batch)
    conv = nn.conv_layer(norm, 32, filter_size=(5, 5), seed=nn.randint(random_state), name="conv1", summarize=True, stddev=0.1)
    conv = nn.conv_layer(conv, 16, filter_size=(5, 5), pool="avg", pooling_kernel=(5, 5), pooling_stride=(3, 3),
                         seed=nn.randint(random_state), name="conv2", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 16, filter_size=(5, 5), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv3", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 32, filter_size=(3, 3), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv4", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 32, filter_size=(3, 3), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv5", summarize=True, stddev=0.1, dropout=keep_proba)

    flattened = nn.flatten_layer(conv, batch_size=batch_size)

    fc1 = nn.fc_layer(flattened, 32, seed=nn.randint(random_state), name="fc1", summarize=True, stddev=0.1, dropout=keep_proba)
    fc2 = nn.fc_layer(fc1, 16, seed=nn.randint(random_state), name="fc2", summarize=True, stddev=0.1, dropout=keep_proba)
    y_pred = nn.fc_layer(fc2, n_classes, seed=nn.randint(random_state), name="fc3", summarize=True, stddev=0.1)
    return X_batch, y_batch, y_pred, keep_proba


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--summary", dest="summary")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--override", dest="override", action="store_true")
    parser.add_argument("--n_epoch", dest="n_epoch", default=1, type=int)
    parser.add_argument("--epoch_offset", dest="epoch_offset", type=int, default=0)
    parser.add_argument("--name", dest="name", default="thyroid_pattern")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.set_defaults(override=False)
    params, unknown = parser.parse_known_args(argv)

    # create model
    seed = params.seed
    random_state = check_random_state(seed)
    height, width, channel = 48, 48, 3
    n_classes = 2
    batch_size = None

    X_batch, y_batch, y_pred, keep_proba = get_model(
        height, width, channel,
        n_classes, random_state=random_state,
        batch_size=batch_size
    )
    learning_rate = tf.placeholder(tf.float64, name="learning_rate")
    loss, optimizer = nn.optimizer(
        y_pred, y_batch,
        opt="adam",
        learning_rate=learning_rate,
        summarize=True
    )
    probas = tf.nn.softmax(y_pred)
    metrics = nn.evaluations(y_pred, y_batch, summarize=True)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    tf.summary.image(name="windows", tensor=X_batch, max_outputs=4)
    summaries = tf.summary.merge_all()

    print("Complexity: {}".format(nn.current_model_complexity()))
    print("Load dataset...")
    dataset = ImageClassificationDataset(params.dataset, dirs=["train", "test"])
    image_provider = ImageProvider(dataset, dirs=["train"], classes=[0, 1], one_hot=True, stratified=False)

    actual_batch_size = 64
    n_subwindows = 8
    sw_min_size = 0.4
    sw_max_size = 0.8

    print("Start session...")
    with tf.Session() as sess:
        sess.run([init_local, init_global])
        summary_writer = nn.get_summary_writer(sess, params.summary, params.name, include_datetime=True)

        if params.override:
            print("Init weights if checkpoint exists.")
            nn.restore_if_exists(sess, params.model, params.name)

        feed_dict = {
            keep_proba: 0.6,
            learning_rate: 1e-3
        }

        trainer = RandomSubwindowsTrainer(
            n_subwindows=n_subwindows, colorspace=COLORSPACE_RGB,
            min_size=sw_min_size, max_size=sw_max_size,
            target_width=width, target_height=height,
            n_jobs=7,
            image_provider=image_provider,
            session=sess,
            input=X_batch, output=y_batch,
            loss=loss, optimizer=optimizer,
            batch_size=actual_batch_size,
            feed_dict=feed_dict,
            summary_period=1, print_period=1
        )
        trainer.set_displayed_metric("accuracy", metrics["accuracy"])
        trainer.set_train_update_ops([probas])
        trainer.set_summaries(summary_writer, summaries, feed_dict=feed_dict)

        print("Start learning...")
        for epoch in range(params.n_epoch):
            print("Start training for epoch {}:".format(epoch))
            trainer.train_epoch(epoch + params.epoch_offset, random_state=random_state)
            nn.save_session(sess, params.model, params.name)

        print("Optimization Finished!")
        X_test, y_test = dataset.all(dirs=["test"], classes=[0, 1], one_hot=True, stratified=False)

        print("Run tests...")
        predicted = nn.predict_proba_with_windows(
            sess, X=X_batch,
            images=X_test, n_classes=2,
            probas=probas,
            n_subwindows=n_subwindows * 4,
            batch_size=20,
            feed_dict={
                keep_proba: 1
            },
            min_size=sw_min_size,
            max_size=sw_max_size,
            colorspace=COLORSPACE_RGB,
            n_jobs=7,
            random_state=40,  # use the same random seed for any testing !
        )

        nn.print_tf_scores(y_test, predicted, ["{}".format(i) for i in range(2)])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])