import os
import tensorflow as tf

from generic import custom_iso


def save_session(sess, path, name):
    fullpath = os.path.join(path, "{}.ckpt".format(name))
    if not os.path.exists(os.path.dirname(fullpath)):
        os.makedirs(os.path.dirname(fullpath))
    tf.train.Saver().save(sess, fullpath)


def restore_session(sess, path, name):
    fullpath = os.path.join(path, "{}.ckpt".format(name))
    tf.train.Saver().restore(sess, fullpath)


def restore_if_exists(sess, path, name):
    filename = "{}.ckpt".format(name)
    files = next(os.walk(path))[2]
    if any([file.startswith(filename) for file in files]):
        restore_session(sess, path, name)


def get_summary_writer(sess, path, name, include_datetime=False):
    fullpath = os.path.join(path, name).replace("\\", "/")
    if include_datetime:
        fullpath = os.path.join(fullpath, custom_iso("_")).replace("\\", "/")
    return tf.summary.FileWriter(fullpath, sess.graph)
