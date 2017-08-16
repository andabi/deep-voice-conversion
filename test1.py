# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vc
'''

from __future__ import print_function
from utils import load_vocab
from tqdm import tqdm

from train1 import Model
from data_load import *


def test():
    # Load graph
    g = Model(mode="test1"); print("Graph loaded")

    # Load vocabulary
    phn2idx, idx2phn = load_vocab()

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint("logdir1")); print("Restored!")

        # Get model name
        mname = open('logdir1/checkpoint', 'r').read().split('"')[1]; print("model:", mname)  # model name

        total_num_hits, total_num_targets = 0, 0
        for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            preds, y, num_hits, num_targets = sess.run([g.preds_ppgs, g.ppgs, g.num_hits, g.num_targets])
            total_num_hits += num_hits
            total_num_targets += num_targets
        print("acc:", total_num_hits / total_num_targets)


def summaries(acc):
    tf.summary.scalar('acc', acc)

    return tf.summary.merge_all()


if __name__ == '__main__':
    test()
    print("Done")


