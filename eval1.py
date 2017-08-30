# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

from data_load import *
from models import Model


def eval():
    # Load graph
    model = Model(mode="test1")

    # Accuracy
    acc_op = model.acc_net1()

    # Summary
    summ_op = summaries(acc_op)

    sv = tf.train.Supervisor()

    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=session_conf) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint("logdir/train1"))

        summ, acc = sess.run([summ_op, acc_op])
        sv.summary_computed(sess, summ)

        print("acc:", acc)


def summaries(acc):
    tf.summary.scalar('net1/eval/acc', acc)
    return tf.summary.merge_all()


if __name__ == '__main__':
    eval()
    print("Done")
