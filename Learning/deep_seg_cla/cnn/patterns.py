from argparse import ArgumentParser

import tensorflow as tf
from sklearn.utils import check_random_state

import nn
from generic.dataset import ImageClassificationDataset, ImageProvider
from nn.training import BatchTrainer


def get_model(height, width, channel, n_classes, batch_size=None, random_state=0):
    random_state = check_random_state(random_state)
    keep_proba = tf.placeholder(tf.float32, name="keep_proba")
    X_batch = tf.placeholder(tf.float32, shape=(batch_size, height, width, channel), name="X_in")
    y_batch = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name="y_true")

    # build network
    # 128 -> 128 -> 42 -> 21 -> 10 -> 5
    norm = nn.norm_input_layer(X_batch)
    conv = nn.conv_layer(norm, 32, filter_size=(5, 5), seed=nn.randint(random_state), name="conv1", summarize=True, stddev=0.1)
    conv = nn.conv_layer(conv, 32, filter_size=(5, 5), pool="avg", pooling_kernel=(5, 5), pooling_stride=(3, 3),
                         seed=nn.randint(random_state), name="conv2", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 32, filter_size=(5, 5), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv3", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 32, filter_size=(3, 3), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv4", summarize=True, stddev=0.1, dropout=keep_proba)
    conv = nn.conv_layer(conv, 32, filter_size=(3, 3), pool="avg", pooling_kernel=(3, 3),
                         seed=nn.randint(random_state), name="conv5", summarize=True, stddev=0.1, dropout=keep_proba)

    flattened = nn.flatten_layer(conv, batch_size=batch_size)

    fc1 = nn.fc_layer(flattened, 64, seed=nn.randint(random_state), name="fc1", summarize=True, stddev=0.1, dropout=keep_proba)
    fc2 = nn.fc_layer(fc1, 32, seed=nn.randint(random_state), name="fc2", summarize=True, stddev=0.1, dropout=keep_proba)
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
    parser.add_argument("--name", dest="name", default="thyroid_patterns_full")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.set_defaults(override=False)
    params, unknown = parser.parse_known_args(argv)

    seed = params.seed
    random_state = check_random_state(seed)
    height, width, channel = 128, 128, 3
    n_classes = 2
    batch_size = None

    X_batch, y_batch, y_pred, keep_proba = get_model(height, width, channel, n_classes, random_state=random_state, batch_size=batch_size)
    print("Complexity: {}".format(nn.current_model_complexity()))
    learning_rate = tf.placeholder(tf.float64, name="learning_rate")
    tf.summary.image("input", X_batch)
    # momentum = tf.placeholder(tf.float64, name="momentum")
    # decay = tf.placeholder(tf.float64, name="decay")
    loss, optimizer = nn.optimizer(
        y_pred, y_batch,
        opt="adam",
        learning_rate=learning_rate,
        # momentum=momentum,
        # decay=decay,
        summarize=True
    )
    probas = tf.nn.softmax(y_pred)
    metrics = nn.evaluations(probas, y_batch, summarize=True)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    summaries = tf.summary.merge_all()

    print("Load dataset...")
    dataset = ImageClassificationDataset(params.dataset, dirs=["train", "validation", "test"])
    image_provider = ImageProvider(dataset, dirs=["train", "validation"], classes=[0, 1], one_hot=True, stratified=False)

    n_epoch = params.n_epoch
    batch_size = 64

    print("Start session...")
    with tf.Session() as sess:
        sess.run([init, init_local])
        summary_writer = nn.get_summary_writer(sess, params.summary, params.name, include_datetime=True)

        if params.override:
            print("Override if checkpoint exists!")
            nn.restore_session(sess, params.model, params.name)

        # build the trainer
        feed_dict = {
            keep_proba: 0.5,
            # momentum: 0.0,
            # decay: 0.9,
            learning_rate: 1e-3
        }

        trainer = BatchTrainer(
            image_provider=image_provider,
            session=sess,
            input=X_batch, output=y_batch,
            loss=loss, optimizer=optimizer,
            batch_size=batch_size,
            feed_dict=feed_dict,
            summary_period=10, print_period=1
        )
        trainer.set_displayed_metric("accuracy", metrics["accuracy"])
        trainer.set_summaries(summary_writer, summaries, feed_dict=feed_dict)

        print("Start learning...")
        for epoch in range(n_epoch):
            print("Start training for epoch {}:".format(epoch))
            trainer.train_epoch(params.epoch_offset + epoch)
            nn.save_session(sess, params.model, params.name)

        print("Optimization Finished!")
        X_test, y_test = dataset.all(classes=[0, 1], dirs=["test"], one_hot=True, train=False)
        test_images = dataset.load_images(X_test)

        print("Run tests...")
        predicted = nn.predict_proba(
            sess, X_batch, y_batch,
            test_images, y_test, probas,
            batch_size=64,
            more_feed_dict={keep_proba: 1}
        )

        nn.print_tf_scores(y_test, predicted, ["{}".format(i) for i in range(2)])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])