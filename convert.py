# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse

from data_load import get_wav_batch
from models import Model
import numpy as np
from utils import spectrogram2wav, inv_preemphasis
from hparams import logdir_path
import datetime
import tensorflow as tf
from hparams import Default as hp_default
import hparams as hp


def convert(logdir='logdir/default/train2', queue=False):

    # Load graph
    model = Model(mode="convert", batch_size=hp.Convert.batch_size, queue=queue)

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
            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_spec, model.ppgs])
        else:
            mfcc, spec, mel = get_wav_batch(model.mode, model.batch_size)
            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_spec, model.ppgs], feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

        # Denormalizatoin
        # pred_log_specs = hp.mean_log_spec + hp.std_log_spec * pred_log_specs
        # y_log_spec = hp.mean_log_spec + hp.std_log_spec * y_log_spec
        # pred_log_specs = hp.min_log_spec + (hp.max_log_spec - hp.min_log_spec) * pred_log_specs
        # y_log_spec = hp.min_log_spec + (hp.max_log_spec - hp.min_log_spec) * y_log_spec

        # Convert log of magnitude to magnitude
        pred_specs, y_specs = np.e ** pred_log_specs, np.e ** y_log_spec

        # Emphasize the magnitude
        pred_specs = np.power(pred_specs, hp.Convert.emphasis_magnitude)
        y_specs = np.power(y_specs, hp.Convert.emphasis_magnitude)

        # Spectrogram to waveform
        audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp_default.n_fft, hp_default.win_length, hp_default.hop_length, hp_default.n_iter), pred_specs))
        y_audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp_default.n_fft, hp_default.win_length, hp_default.hop_length, hp_default.n_iter), y_specs))

        # Apply inverse pre-emphasis
        audio = inv_preemphasis(audio, coeff=hp_default.preemphasis)
        y_audio = inv_preemphasis(y_audio, coeff=hp_default.preemphasis)

        if not queue:
            # Concatenate to a wav
            y_audio = np.reshape(y_audio, (1, y_audio.size), order='C')
            audio = np.reshape(audio, (1, audio.size), order='C')

        # Write the result
        tf.summary.audio('A', y_audio, hp_default.sr, max_outputs=hp.Convert.batch_size)
        tf.summary.audio('B', audio, hp_default.sr, max_outputs=hp.Convert.batch_size)

        # Visualize PPGs
        heatmap = np.expand_dims(ppgs, 3)  # channel=1
        tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)
        writer.close()

        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
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