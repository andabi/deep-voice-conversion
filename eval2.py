# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

from data_load import *
from models import Model


def eval():
    # Load graph
    model = Model(mode="test2")

    # Loss
    loss_op = model.loss_net2()

    # Summary
    summ_op = summaries(loss_op)

    sv = tf.train.Supervisor(logdir="logdir/train2",
                             save_model_secs=0)
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        summ, loss = sess.run([summ_op, loss_op])
        sv.summary_computed(sess, summ)
        print("loss:", loss)


def summaries(loss):
    tf.summary.scalar('net2/eval/loss', loss)
    return tf.summary.merge_all()


if __name__ == '__main__':
    eval()
    print("Done")
