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


def convert(logdir='logdir/train2', queue=True):
    # Load graph
    model = Model(mode="convert", batch_size=hp.convert.batch_size, queue=queue)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=1
        ),
    )
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load_variables(sess, 'convert', logdir=logdir)

        writer = tf.summary.FileWriter(logdir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        model_name = open('{}/checkpoint'.format(logdir), 'r').read().split('"')[1]
        gs = int(model_name.split('_')[3])

        pred_specs, y_specs = model.convert(sess)
        specs = np.where(pred_specs < 0, 0., pred_specs)
        audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), specs))
        y_audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), y_specs))

        # Write the result
        tf.summary.audio('orig', y_audio, hp.sr, max_outputs=hp.convert.batch_size)
        tf.summary.audio('pred', audio, hp.sr, max_outputs=hp.convert.batch_size)
        #     write('outputs/{}_{}.wav'.format(mname, i), hp.sr, audio)
        #     write('outputs/{}_{}_origin.wav'.format(mname, i), hp.sr, y_audio)

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)
        writer.close()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    convert(logdir='logdir_relu/train2')
    print("Done")
