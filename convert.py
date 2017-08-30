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
        sv.saver.restore(sess, tf.train.latest_checkpoint("logdir/train2"))

        # Get model name
        mname = open('logdir/train2/checkpoint', 'r').read().split('"')[1]

        specs, y_specs = sess.run([model(), model.y_spec])
        for i, (spec, y_spec) in enumerate(zip(specs, y_specs)):
            audio = spectrogram2wav(spec)
            write('outputs/{}_{}.wav'.format(mname, i), hp.sr, audio)

            y_audio = spectrogram2wav(y_spec)
            write('outputs/{}_{}_gt.wav'.format(mname, i), hp.sr, y_audio)

if __name__ == '__main__':
    convert()
    print("Done")