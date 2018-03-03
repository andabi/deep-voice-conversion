# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse

import tensorflow as tf

from data_load import get_batch, phns, load_vocab
from hparam import hparam as hp
from models import Model
from utils import plot_confusion_matrix


def eval(logdir, writer, queue=False):
    # Load graph
    model = Model(mode="test1", batch_size=hp.test1.batch_size, queue=queue)

    # Accuracy
    acc_op = model.acc_net1()

    # Loss
    loss_op = model.loss_net1()

    # confusion matrix
    y_ppg_1d = tf.reshape(model.y_ppg, shape=(tf.size(model.y_ppg),))
    pred_ppg_1d = tf.reshape(model.pred_ppg, shape=(tf.size(model.pred_ppg),))

    # Summary
    tf.summary.scalar('net1/eval/acc', acc_op)
    tf.summary.scalar('net1/eval/loss', loss_op)
    summ_op = tf.summary.merge_all()

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 0},
    )
    with tf.Session(config=session_conf) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'train1', logdir=logdir)

        if queue:
            summ, acc, loss, y_ppg_1d, pred_ppg_1d = sess.run([summ_op, acc_op, loss_op, y_ppg_1d, pred_ppg_1d])
        else:
            mfcc, ppg = get_batch(model.mode, model.batch_size)
            summ, acc, loss, y_ppg_1d, pred_ppg_1d = sess.run([summ_op, acc_op, loss_op, y_ppg_1d, pred_ppg_1d],
                                                              feed_dict={model.x_mfcc: mfcc, model.y_ppg: ppg})

        # plot confusion matrix
        _, idx2phn = load_vocab()
        y_ppg_1d = [idx2phn[i] for i in y_ppg_1d]
        pred_ppg_1d = [idx2phn[i] for i in pred_ppg_1d]
        cm_summ = plot_confusion_matrix(y_ppg_1d, pred_ppg_1d, phns)

        writer.add_summary(summ)
        writer.add_summary(cm_summ)

        print("acc:", acc)
        print("loss:", loss)
        print('\n')

        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case)
    logdir = '{}/train1'.format(hp.logdir)
    writer = tf.summary.FileWriter(logdir)
    eval(logdir=logdir, writer=writer)
    writer.close()

    print("Done")
