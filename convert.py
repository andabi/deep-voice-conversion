# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse

from data_load import get_wav_batch, get_batch
from models import Model
import numpy as np
from audio import spectrogram2wav, inv_preemphasis
from hparam import logdir_path
import datetime
import tensorflow as tf
from hparam import Hparam


def convert(logdir, step, writer, queue=False):
    hp = Hparam.get_global_hparam()

    # Load graph
    model = Model(mode="convert", batch_size=hp.convert.batch_size, hp=hp, queue=queue)

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
        model.load(sess, 'convert', logdir=logdir, step=step)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        epoch, gs = Model.get_epoch_and_global_step(logdir, step=step)

        if queue:
            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_spec, model.ppgs])
        else:
            if hp.convert.one_full_wav:
                mfcc, spec, mel = get_wav_batch(model.mode, model.batch_size)
            else:
                mfcc, spec, mel = get_batch(model.mode, model.batch_size)

            pred_log_specs, y_log_spec, ppgs = sess.run([model(), model.y_spec, model.ppgs], feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel})

        # Denormalizatoin
        # pred_log_specs = hp.mean_log_spec + hp.std_log_spec * pred_log_specs
        # y_log_spec = hp.mean_log_spec + hp.std_log_spec * y_log_spec
        # pred_log_specs = hp.min_log_spec + (hp.max_log_spec - hp.min_log_spec) * pred_log_specs
        # y_log_spec = hp.min_log_spec + (hp.max_log_spec - hp.min_log_spec) * y_log_spec

        # Convert log of magnitude to magnitude
        pred_specs, y_specs = np.e ** pred_log_specs, np.e ** y_log_spec

        # Emphasize the magnitude
        pred_specs = np.power(pred_specs, hp.convert.emphasis_magnitude)
        y_specs = np.power(y_specs, hp.convert.emphasis_magnitude)

        # Spectrogram to waveform
        audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length, hp.default.n_iter), pred_specs))
        y_audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length, hp.default.n_iter), y_specs))

        # Apply inverse pre-emphasis
        audio = inv_preemphasis(audio, coeff=hp.default.preemphasis)
        y_audio = inv_preemphasis(y_audio, coeff=hp.default.preemphasis)

        if hp.convert.one_full_wav:
            # Concatenate to a wav
            y_audio = np.reshape(y_audio, (1, y_audio.size), order='C')
            audio = np.reshape(audio, (1, audio.size), order='C')

        # Write the result
        tf.summary.audio('A', y_audio, hp.default.sr, max_outputs=hp.convert.batch_size)
        tf.summary.audio('B', audio, hp.default.sr, max_outputs=hp.convert.batch_size)

        # Visualize PPGs
        heatmap = np.expand_dims(ppgs, 3)  # channel=1
        tf.summary.image('PPG', heatmap, max_outputs=ppgs.shape[0])

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)

        coord.request_stop()
        coord.join(threads)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, help='experiment case name')
    parser.add_argument('-step', type=int, help='checkpoint step to load', nargs='?', default=None)
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    case = args.case
    logdir = '{}/{}/train2'.format(logdir_path, case)
    Hparam(case).set_as_global_hparam()

    print('case: {}, logdir: {}'.format(case, logdir))

    s = datetime.datetime.now()

    writer = tf.summary.FileWriter(logdir)
    convert(logdir=logdir, step=args.step, writer=writer)
    writer.close()

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))