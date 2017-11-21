# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse

import tensorflow as tf
from hparam import Hparam
from data_load import get_batch
from hparam import logdir_path
from models import Model


def eval(logdir, writer, queue=False):
    hp = Hparam.get_global_hparam()

    # Load graph
    model = Model(mode="test1", batch_size=hp.test1.batch_size, hp=hp, queue=queue)

    # Accuracy
    acc_op = model.acc_net1()

    # Loss
    loss_op = model.loss_net1()

    # Summary
    summ_op = summaries(acc_op, loss_op)

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
            summ, acc, loss = sess.run([summ_op, acc_op, loss_op])
        else:
            mfcc, ppg = get_batch(model.mode, model.batch_size)
            summ, acc, loss = sess.run([summ_op, acc_op, loss_op], feed_dict={model.x_mfcc: mfcc, model.y_ppgs: ppg})

        writer.add_summary(summ)

        print("acc:", acc)
        print("loss:", loss)
        print('\n')

        coord.request_stop()
        coord.join(threads)


def summaries(acc, loss):
    tf.summary.scalar('net1/eval/acc', acc)
    tf.summary.scalar('net1/eval/loss', loss)
    return tf.summary.merge_all()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    case = args.case
    logdir = '{}/{}/train1'.format(logdir_path, case)
    Hparam(case).set_as_global_hparam()

    writer = tf.summary.FileWriter(logdir)
    eval(logdir=logdir, writer=writer)
    writer.close()

    print("Done")
