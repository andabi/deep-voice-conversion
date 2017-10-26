# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse

from data_load import get_batch_per_wav
from models import Model
from utils import *
from hparams import logdir_path
import datetime
import tensorflow as tf


def convert(logdir='logdir/default/train2', queue=False):
    # Load graph
    model = Model(mode="convert", batch_size=hp.convert.batch_size, queue=queue)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 0},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6
        ),
    )
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, 'convert', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        gs = Model.get_global_step(logdir)

        if queue:
            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_log_spec, model.ppgs])
        else:
            x, y = get_batch_per_wav(model.mode, model.batch_size)
            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_log_spec, model.ppgs], feed_dict={model.x_mfcc: x, model.y_spec: y})

        # Convert log of magnitude to magnitude
        if model.log_mag:
            pred_specs, y_specs = np.e ** pred_log_specs, np.e ** y_log_spec
        else:
            pred_specs = np.where(pred_log_specs < 0, 0., pred_log_specs)
            y_specs = np.where(pred_specs < 0, 0., y_log_spec)

        # Emphasize the magnitude
        pred_specs = np.power(pred_specs, hp.convert.emphasis_magnitude)
        y_specs = np.power(y_specs, hp.convert.emphasis_magnitude)

        # Spectrogram to waveform
        audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), pred_specs))
        y_audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), y_specs))

        # Apply inverse pre-emphasis
        audio = inv_preemphasis(audio, coeff=hp.preemphasis)
        y_audio = inv_preemphasis(y_audio, coeff=hp.preemphasis)


        # Visualize PPGs
        heatmap = np.expand_dims(ppgs, 3)  # channel=1
        tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])

        # Concatenate to a wav
        y_audio = np.reshape(y_audio, (1, y_audio.size), order='C')
        audio = np.reshape(audio, (1, audio.size), order='C')

        # Write the result
        tf.summary.audio('A', y_audio, hp.sr, max_outputs=hp.convert.batch_size)
        tf.summary.audio('B', audio, hp.sr, max_outputs=hp.convert.batch_size)


        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)
        writer.close()

        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='case')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    case = args.case
    logdir = '{}/{}/train2'.format(logdir_path, case)

    print('case: {}, logdir: {}'.format(case, logdir))

    s = datetime.datetime.now()

    convert(logdir=logdir)

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))