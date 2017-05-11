import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from tensorflow.examples.tutorials.mnist import input_data
import nn


def get_model(height, width, channel, n_classes, batch_size=None, random_state=0):
    random_state = check_random_state(random_state)
    X_batch = tf.placeholder(tf.float32, shape=(batch_size, height, width, channel), name="X_train")
    y_batch = tf.placeholder(tf.int64, shape=(batch_size, n_classes), name="y_true")
    keep_prob = tf.placeholder(tf.float32, name="keep_proba")

    # build network
    conv1 = nn.conv_layer(X_batch, 5, filter_size=(3, 3), dropout=keep_prob, seed=nn.randint(random_state), name="conv1",
                          summarize=True, stddev=0.1, batch_norm=True)
    conv2 = nn.conv_layer(conv1, 5, filter_size=(3, 3), dropout=keep_prob, seed=nn.randint(random_state), name="conv2",
                          summarize=True, stddev=0.1, batch_norm=True)

    flattened = nn.flatten_layer(conv2, batch_size=batch_size)

    fc1 = nn.fc_layer(flattened, 100, seed=nn.randint(random_state), name="fc1", summarize=True, batch_norm=True)
    fc2 = nn.fc_layer(fc1, 20, seed=nn.randint(random_state), name="fc2", summarize=True, batch_norm=True)
    y_pred = nn.fc_layer(fc2, n_classes, seed=nn.randint(random_state), name="output", summarize=True, batch_norm=True)
    return X_batch, y_batch, y_pred, keep_prob


def main():
    # Load mnist data
    mnist = input_data.read_data_sets("C:/data/", one_hot=True)

    seed = 0
    random_state = check_random_state(seed)
    height, width, channel = 28, 28, 1
    n_classes = 10
    name = "mnist"

    X_batch, y_batch, y_pred, keep_proba = get_model(height, width, channel, n_classes, random_state=random_state)
    loss, optimizer = nn.optimizer(y_pred, y_batch, opt="rmsprop", learning_rate=1e-4, summarize=True)
    probas = tf.nn.softmax(y_pred, name="softmax")
    metrics = nn.evaluations(y_pred, y_batch, summarize=True)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    summaries = tf.summary.merge_all()

    BATCH_SIZE = 128
    TRAIN_ITER = 1500000
    with tf.Session() as sess:

        writer = nn.get_summary_writer(sess, "D:/data/tf/logdir/mnist/new", "mnist", include_datetime=True)
        sess.run([init, init_local])
        step = 1
        # Keep training until reach max iterations
        while step * BATCH_SIZE < TRAIN_ITER:
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = [np.reshape(img, (28, 28, 1)) for img in batch_x]
            # Run optimization op (backprop)
            results = sess.run(
                [optimizer, summaries, loss, metrics["accuracy"]],
                feed_dict={
                    X_batch: batch_x,
                    y_batch: batch_y,
                    keep_proba: 0.8
                }
            )

            _loss, _accuracy, _summaries = results[2], results[3], results[1]
            if step % 50 == 0:
                # Calculate batch loss and accuracy
                print("Iter {}, Minibatch Loss={:.6f}, Training Accuracy= {:.5f}".format(step * BATCH_SIZE, _loss, _accuracy))
                writer.add_summary(_summaries, step)

            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 256 mnist test images

        predicted, = sess.run([probas], feed_dict={
            X_batch: [np.reshape(img, (28, 28, 1)) for img in mnist.test.images],
            y_batch: mnist.test.labels,
            keep_proba: 1
        })

        nn.print_tf_scores(mnist.test.labels, predicted, ["{}".format(i) for i in range(10)])
        nn.save_session(sess, "D:/data/models/", name)

if __name__ == "__main__":
    main()