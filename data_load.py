# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vc
'''

from utils import *
import glob
import tensorflow as tf

phn2idx, idx2phn = load_vocab()
V = len(phn2idx)

def load_data(mode="train1"):
    '''Loads the list of sound files.
    mode: A string. One of the phases below:
      `train1`: TIMIT TRAIN waveform -> mfccs (inputs) -> PGGs -> phones (target) (ce loss)
      `test1`: TIMIT TEST waveform -> mfccs (inputs) -> PGGs -> phones (target) (accuracy)
      `train2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(l2 loss)
      `test2`: ARCTIC SLT waveform -> mfccs -> PGGs (inputs) -> spectrogram (target)(accuracy)
      `convert`: ARCTIC BDL waveform -> mfccs (inputs) -> PGGs -> spectrogram -> waveform (output)
    '''
    if mode == "train1":
        wav_files = glob.glob('datasets/timit/TIMIT/TRAIN/*/*/*.wav')
    elif mode == "test1":
        wav_files = glob.glob('datasets/timit/TIMIT/TEST/*/*/*.wav')
    elif mode == "train2":  # target speaker arctic.slt (female)
        wav_files = glob.glob('datasets/arctic/slt/*.wav')[:-10]
    elif mode == "test2": # target speaker arctic.slt (female)
        wav_files = glob.glob('datasets/arctic/slt/*.wav')[-10:]
    elif mode == "convert":  # source speaker arctic.bdl (male)
        wav_files = glob.glob('datasets/arctic/bdl/*.wav')
    return wav_files


def get_batch(mode="train1"):
    '''Loads data and put them in mini batch queues.
    mode: A string. Either `train1` | `test1` | `train2` | `test2` | `convert`.
    '''
    with tf.device('/cpu:0'):
        # Load data
        wav_files = load_data(mode=mode)

        # calc total batch count
        num_batch = len(wav_files) // hp.batch_size

        # Convert to tensor
        wav_files = tf.convert_to_tensor(wav_files)

        if mode in ('train1', 'test1', 'train2', 'test2', 'convert'):
            # Create Queues
            wav_file, = tf.train.slice_input_producer([wav_files, ], shuffle=True)

            if mode in ('train1', 'test1'):
                # Get inputs and target
                x, y = get_mfccs_and_phones(inputs=wav_file,
                                            dtypes=[tf.float32, tf.int32],
                                            capacity=128,
                                            num_threads=32)

                # create batch queues
                x, y = tf.train.batch([x, y],
                                      shapes=[(None, hp.n_mfcc), (None,)],
                                      num_threads=32,
                                      batch_size=hp.batch_size,
                                      capacity=hp.batch_size * 32,
                                      dynamic_pad=True)

            else:
                # Get inputs and target
                x, y = get_mfccs_and_spectrogram(inputs=wav_file,
                                              dtypes=[tf.float32, tf.float32],
                                              capacity=128,
                                              num_threads=32)

                # create batch queues
                x, y = tf.train.batch([x, y],
                                    shapes=[(None, hp.n_mfcc), (None, 1+hp.n_fft//2)],
                                    num_threads=32,
                                    batch_size=hp.batch_size,
                                    capacity=hp.batch_size * 32,
                                    dynamic_pad=True)

            return x, y, num_batch