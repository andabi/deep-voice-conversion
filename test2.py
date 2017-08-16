# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/vc
'''

from __future__ import print_function
import codecs
import copy
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from models import Model
from data_load import *
from tqdm import tqdm


def eval():
    # Load graph
    model = Model(mode="test")
    print("Graph loaded")

    # Load vocabulary
    phn2idx, idx2phn = load_vocab()

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        print("Restored!")

        # Get model name
        mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

        print("model:", mname)
        total_num_hits, total_num_targets = 0, 0
        for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
            preds, y, num_hits, num_targets = sess.run([model.preds_spec, model.y_spec, model.num_hits, model.num_targets])
            total_num_hits += num_hits
            total_num_targets += num_targets
        print("acc:", total_num_hits / total_num_targets)


def summaries(acc):
    tf.summary.scalar('acc', acc)

    return tf.summary.merge_all()


if __name__ == '__main__':
    eval()
    print("Done")


