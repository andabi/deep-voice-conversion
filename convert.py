# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vc
'''

from __future__ import print_function

from scipy.io.wavfile import write
from utils import *
from models import Model
from tqdm import tqdm


def convert():
    # Load graph
    model = Model(mode="convert")
    print("Graph loaded")

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint("logdir2"))
        print("Restored!")

        # Get model name
        mname = open('logdir2/checkpoint', 'r').read().split('"')[1]; print("model:", mname)  # model name

        for step in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
            specs = sess.run(model.logits)
            for s in specs:
                audio = spectrogram2wav(np.power(np.e, s))
                write('outputs/{}_{}'.format(mname, i), hp.sr, audio)

if __name__ == '__main__':
    convert()
    print("Done")


