# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

from data_load import *
from models import Model


def eval(logdir='logdir/train1'):
    # Load graph
    model = Model(mode="test1")

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
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load_variables(sess, 'train1', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        summ, acc, loss = sess.run([summ_op, acc_op, loss_op])

        writer.add_summary(summ)

        writer.close()

        coord.request_stop()
        coord.join(threads)

        print("acc:", acc)
        print("loss:", loss)


def summaries(acc, loss):
    tf.summary.scalar('net1/eval/acc', acc)
    tf.summary.scalar('net1/eval/loss', loss)
    return tf.summary.merge_all()


if __name__ == '__main__':
    eval(logdir='logdir_preem/train1')
    print("Done")
