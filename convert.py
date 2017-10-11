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
from data_load import get_batch


def convert(logdir='logdir/train2', queue=True):
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
            pred_specs, y_specs = sess.run([model(), model.y_spec])
        else:
            x, y = get_batch(model.mode, model.batch_size)
            pred_specs, y_specs = sess.run([model(), model.y_spec], feed_dict={model.x_mfcc: x, model.y_spec: y})

        # Convert log of magnitude to magnitude
        if hp.log_mag:
            pred_specs, y_specs = np.e ** pred_specs, np.e ** y_specs
        else:
            pred_specs = np.where(pred_specs < 0, 0., pred_specs)
            y_specs = np.where(pred_specs < 0, 0., y_specs)

        pred_specs, y_specs = model.convert(sess)
        audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), pred_specs))
        y_audio = np.array(map(lambda spec: spectrogram2wav(spec.T, hp.n_fft, hp.win_length, hp.hop_length, hp.n_iter), y_specs))

        # Write the result
        tf.summary.audio('A', y_audio, hp.sr, max_outputs=hp.convert.batch_size)
        tf.summary.audio('B', audio, hp.sr, max_outputs=hp.convert.batch_size)
        #     write('outputs/{}_{}.wav'.format(mname, i), hp.sr, audio)
        #     write('outputs/{}_{}_origin.wav'.format(mname, i), hp.sr, y_audio)

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gs)
        writer.close()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    convert(logdir='logdir/train2')
    print("Done")
