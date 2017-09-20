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


def convert(logdir='logdir/train2'):
    # Load graph
    model = Model(mode="convert")

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load_variables(sess, 'convert', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get model name
        mname = open('{}/checkpoint'.format(logdir), 'r').read().split('"')[1]

        specs, y_specs = sess.run([model(), model.y_spec])
        for i, (spec, y_spec) in enumerate(zip(specs, y_specs)):
            audio = spectrogram2wav(spec)
            y_audio = spectrogram2wav(y_spec)

            write('outputs/{}_{}.wav'.format(mname, i), hp.sr, audio)
            write('outputs/{}_{}_gt.wav'.format(mname, i), hp.sr, y_audio)

        # TODO
        # Write the result
        # tf.summary.audio('pred', audio, hp.sr, max_outputs=hp.batch_size)
        # tf.summary.audio('gt', y_audio, hp.sr, max_outputs=hp.batch_size)

        writer.add_summary(sess.run(tf.summary.merge_all()))
        writer.close()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    convert()
    print("Done")
